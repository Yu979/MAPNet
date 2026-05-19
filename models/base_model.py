import torch
import torch.nn as nn
from transformers import AutoModel
from .embeddings import MoleculeEmbedding, ProteinEmbedding
from .fusion import FusionFactory
from utils.pool import get_pooling_layer

class AdamantanePeptideModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        local_files_only = config.get('model.local_files_only', False)
        
        # Load pretrained models.
        print(f"Loading Molformer: {config.get('model.mol_model_name')} ...")
        self.mol_model = AutoModel.from_pretrained(
            config.get('model.mol_model_name'), 
            trust_remote_code=True, 
            deterministic_eval=True,
            local_files_only=local_files_only
        )
        self.mol_dim = self.mol_model.config.hidden_size
        
        print(f"Loading ESM-2: {config.get('model.prot_model_name')} ...")
        self.prot_model = AutoModel.from_pretrained(
            config.get('model.prot_model_name'),
            local_files_only=local_files_only
        )
        self.prot_dim = self.prot_model.config.hidden_size
        
        for param in self.mol_model.parameters():
            param.requires_grad = False
    
        for param in self.prot_model.parameters():
            param.requires_grad = False
        
        self.mol_eos_id = 1 
        self.prot_eos_id = 2
        
        fusion_dim = config.get('model.fusion_dim')
        
        # Small-molecule embedding extractor.
        mol_emb_type = config.get('model.mol_embedding.type', 'mean')
        self.mol_embedding = MoleculeEmbedding(mol_emb_type, self.mol_dim)
        
        # Protein embedding extractor.
        prot_emb_type = config.get('model.prot_embedding.type', 'transformer')
        self.prot_embedding = ProteinEmbedding(prot_emb_type, hidden_dim=self.prot_dim)
        
        # Projection layers.
        if config.get('model.mol_embedding.use_projection', True):
            self.mol_projector = nn.Linear(self.mol_dim, fusion_dim)
        else:
            self.mol_projector = nn.Identity()
            
        if config.get('model.prot_embedding.use_projection', True):
            self.prot_projector = nn.Linear(self.prot_dim, fusion_dim)
        else:
            self.prot_projector = nn.Identity()
        
        # Fusion module.
        fusion_type = config.get('model.fusion.type', 'transformer')
        self.fusion = FusionFactory.get_fusion_module(
            fusion_type,
            fusion_dim,
            n_layers=config.get('model.n_transformer_layers', 2),
            n_heads=config.get('model.n_heads', 4)
        )
        
        # Pooling layer.
        pooling_type = config.get('model.pooling.type', 'attention')
        self.pooling = get_pooling_layer(pooling_type, fusion_dim)
        
        # Compute the regressor input dimension.
        if pooling_type == 'max_mean':
            regressor_input_dim = fusion_dim * 2
        else:
            regressor_input_dim = fusion_dim
        
        # Build the regressor.
        self.regressor = self._build_regressor(
            regressor_input_dim,
            config.get('model.regressor.hidden_dims', [64]),
            config.get('model.regressor.dropout', 0.1),
            config.get('model.regressor.activation', 'relu')
        )
    
    def _build_regressor(self, input_dim, hidden_dims, dropout, activation):
        """Build the regressor dynamically."""
        layers = []
        prev_dim = input_dim
        
        activation_fn = {
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'tanh': nn.Tanh
        }[activation]
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation_fn(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        # layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)
    
    
    def create_clean_mask(self, input_ids, attention_mask, eos_id):
        """
        Create a clean mask by removing [CLS] (index 0) and [EOS].
        """
        # Copy the original mask (Batch, Seq_Len).
        clean_mask = attention_mask.clone()
        
        # 1. Remove the leading [CLS] (Batch, 0) = 0.
        clean_mask[:, 0] = 0
        
        # 2. Remove [EOS].
        if eos_id is not None:
            is_eos = (input_ids == eos_id)
            clean_mask.masked_fill_(is_eos, 0)
            
        return clean_mask
    
    
    def forward(self, mol_input_ids, mol_attention_mask, prot_input_ids, prot_attention_mask):
        # 1. Extract raw features.
        mol_outputs = self.mol_model(
            input_ids=mol_input_ids, 
            attention_mask=mol_attention_mask
        )
        prot_outputs = self.prot_model(
            input_ids=prot_input_ids, 
            attention_mask=prot_attention_mask
        )
        
        mol_clean_mask = self.create_clean_mask(mol_input_ids, mol_attention_mask, self.mol_eos_id)
        prot_clean_mask = self.create_clean_mask(prot_input_ids, prot_attention_mask, self.prot_eos_id)
        
        # 2. Apply embedding extraction strategies.
        mol_emb = self.mol_embedding(mol_outputs, mol_clean_mask)
        prot_emb = self.prot_embedding(prot_outputs, prot_clean_mask)
        
        # 3. Project to a shared dimension.
        if len(mol_emb.shape) == 2:  # (B, D)
            mol_feat = self.mol_projector(mol_emb).unsqueeze(1)  # (B, 1, D)
        else:  # (B, L, D)
            mol_feat = self.mol_projector(mol_emb)
        
        if len(prot_emb.shape) == 2:  # (B, D)
            prot_feat = self.prot_projector(prot_emb).unsqueeze(1)  # (B, 1, D)
        else:  # (B, L, D)
            prot_feat = self.prot_projector(prot_emb)
        
        # 4. Fuse features.
        combined_feat = torch.cat([mol_feat, prot_feat], dim=1)
        batch_size = mol_feat.shape[0]
        
        mol_valid_mask = torch.ones(batch_size, mol_feat.shape[1]).to(mol_feat.device)
        
        combined_mask = torch.cat([mol_valid_mask, prot_clean_mask], dim=1)
        
        fused_feat = self.fusion(combined_feat, combined_mask)
        # fused_feat = self.fusion(mol_feat, prot_feat, combined_mask)
        
        # 5. Pooling.
        pooled_output = self.pooling(fused_feat, mask=combined_mask)
        
        # 6. Prediction.
        prediction = self.regressor(pooled_output)
        return prediction
    
    


class AdamantanePeptideEmb(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        local_files_only = config.get('model.local_files_only', False)
        
        # Load pretrained models.
        print(f"Loading Molformer: {config.get('model.mol_model_name')} ...")
        self.mol_model = AutoModel.from_pretrained(
            config.get('model.mol_model_name'), 
            trust_remote_code=True, 
            deterministic_eval=True,
            local_files_only=local_files_only
        )
        self.mol_dim = self.mol_model.config.hidden_size
        
        print(f"Loading ESM-2: {config.get('model.prot_model_name')} ...")
        self.prot_model = AutoModel.from_pretrained(
            config.get('model.prot_model_name'),
            local_files_only=local_files_only
        )
        self.prot_dim = self.prot_model.config.hidden_size
        
        for param in self.mol_model.parameters():
            param.requires_grad = False
    
        for param in self.prot_model.parameters():
            param.requires_grad = False
        
        self.mol_eos_id = 1 
        self.prot_eos_id = 2
        
        fusion_dim = config.get('model.fusion_dim')
        
        # Small-molecule embedding extractor.
        mol_emb_type = config.get('model.mol_embedding.type', 'mean')
        self.mol_embedding = MoleculeEmbedding(mol_emb_type, self.mol_dim)
        
        # Protein embedding extractor.
        prot_emb_type = config.get('model.prot_embedding.type', 'transformer')
        self.prot_embedding = ProteinEmbedding(prot_emb_type, hidden_dim=self.prot_dim)
        
        # Projection layers.
        if config.get('model.mol_embedding.use_projection', True):
            self.mol_projector = nn.Linear(self.mol_dim, fusion_dim)
        else:
            self.mol_projector = nn.Identity()
            
        if config.get('model.prot_embedding.use_projection', True):
            self.prot_projector = nn.Linear(self.prot_dim, fusion_dim)
        else:
            self.prot_projector = nn.Identity()
        
        # Fusion module.
        fusion_type = config.get('model.fusion.type', 'transformer')
        self.fusion = FusionFactory.get_fusion_module(
            fusion_type,
            fusion_dim,
            n_layers=config.get('model.n_transformer_layers', 2),
            n_heads=config.get('model.n_heads', 4)
        )
        
        # Pooling layer.
        pooling_type = config.get('model.pooling.type', 'attention')
        self.pooling = get_pooling_layer(pooling_type, fusion_dim)
        
        # Compute the regressor input dimension.
        if pooling_type == 'max_mean':
            regressor_input_dim = fusion_dim * 2
        else:
            regressor_input_dim = fusion_dim
        
        # Build the regressor.
        self.regressor = self._build_regressor(
            regressor_input_dim,
            config.get('model.regressor.hidden_dims', [64]),
            config.get('model.regressor.dropout', 0.1),
            config.get('model.regressor.activation', 'relu')
        )
    
    def _build_regressor(self, input_dim, hidden_dims, dropout, activation):
        """Build the regressor dynamically."""
        layers = []
        prev_dim = input_dim
        
        activation_fn = {
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'tanh': nn.Tanh
        }[activation]
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation_fn(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        # layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)
    
    
    def create_clean_mask(self, input_ids, attention_mask, eos_id):
        """
        Create a clean mask by removing [CLS] (index 0) and [EOS].
        """
        # Copy the original mask (Batch, Seq_Len).
        clean_mask = attention_mask.clone()
        
        # 1. Remove the leading [CLS] (Batch, 0) = 0.
        clean_mask[:, 0] = 0
        
        # 2. Remove [EOS].
        if eos_id is not None:
            is_eos = (input_ids == eos_id)
            clean_mask.masked_fill_(is_eos, 0)
            
        return clean_mask
    
    
    def forward(self, mol_input_ids, mol_attention_mask, prot_input_ids, prot_attention_mask):
        # 1. Extract raw features.
        mol_outputs = self.mol_model(
            input_ids=mol_input_ids, 
            attention_mask=mol_attention_mask
        )
        prot_outputs = self.prot_model(
            input_ids=prot_input_ids, 
            attention_mask=prot_attention_mask
        )
        
        mol_clean_mask = self.create_clean_mask(mol_input_ids, mol_attention_mask, self.mol_eos_id)
        prot_clean_mask = self.create_clean_mask(prot_input_ids, prot_attention_mask, self.prot_eos_id)
        
        # 2. Apply embedding extraction strategies.
        mol_emb = self.mol_embedding(mol_outputs, mol_clean_mask)
        prot_emb = self.prot_embedding(prot_outputs, prot_clean_mask)
        
        # 3. Project to a shared dimension.
        if len(mol_emb.shape) == 2:  # (B, D)
            mol_feat = self.mol_projector(mol_emb).unsqueeze(1)  # (B, 1, D)
        else:  # (B, L, D)
            mol_feat = self.mol_projector(mol_emb)
        
        if len(prot_emb.shape) == 2:  # (B, D)
            prot_feat = self.prot_projector(prot_emb).unsqueeze(1)  # (B, 1, D)
        else:  # (B, L, D)
            prot_feat = self.prot_projector(prot_emb)
        
        # 4. Fuse features.
        combined_feat = torch.cat([mol_feat, prot_feat], dim=1)
        batch_size = mol_feat.shape[0]
        
        mol_valid_mask = torch.ones(batch_size, mol_feat.shape[1]).to(mol_feat.device)
        
        combined_mask = torch.cat([mol_valid_mask, prot_clean_mask], dim=1)
        
        fused_feat = self.fusion(combined_feat, combined_mask)
        # fused_feat = self.fusion(mol_feat, prot_feat, combined_mask)
        
        # 5. Pooling.
        pooled_output = self.pooling(fused_feat, mask=combined_mask)
        
        # 6. Prediction.
        prediction = self.regressor(pooled_output)
        return prediction, pooled_output
