import torch
import torch.nn as nn
from typing import Optional

class MoleculeEmbedding(nn.Module):
    """Small-molecule embedding extractor."""
    def __init__(self, embedding_type: str = "mean", hidden_dim: Optional[int] = None):
        super().__init__()
        self.embedding_type = embedding_type
        
        if embedding_type == "attention":
            assert hidden_dim is not None, "hidden_dim required for attention pooling"
            self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, model_output, attention_mask=None):
        """
        Args:
            model_output: transformer output containing last_hidden_state, etc.
            attention_mask: (Batch, Seq_Len)
        Returns:
            embedding: (Batch, Hidden_Dim)
        """
        if self.embedding_type == "cls":
            # Use the [CLS] token (first position).
            return model_output.last_hidden_state[:, 0, :]
        
        elif self.embedding_type == "mean":
            # Mean pooling with mask support.
            hidden_states = model_output.last_hidden_state
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                return sum_embeddings / sum_mask
            else:
                return hidden_states.mean(dim=1)
        
        elif self.embedding_type == "max":
            # Max pooling.
            hidden_states = model_output.last_hidden_state
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                hidden_states = hidden_states.clone()
                hidden_states[mask_expanded == 0] = -1e9
            return torch.max(hidden_states, dim=1)[0]
        
        elif self.embedding_type == "attention":
            # Attention-weighted pooling.
            hidden_states = model_output.last_hidden_state
            attention_scores = self.attention(hidden_states).squeeze(-1)
            
            if attention_mask is not None:
                attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
            
            attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(-1)
            return torch.sum(hidden_states * attention_weights, dim=1)
        
        else:
            raise ValueError(f"Unknown embedding type: {self.embedding_type}")

class ProteinEmbedding(nn.Module):
    """Protein sequence embedding extractor."""
    def __init__(self, embedding_type: str = "transformer", hidden_dim: Optional[int] = None):
        super().__init__()
        self.embedding_type = embedding_type
        
        if embedding_type == "mean":
            self.pooling_layer = MoleculeEmbedding("mean")
        elif embedding_type == "max":
            self.pooling_layer = MoleculeEmbedding("max")
        elif embedding_type == "attention":  # Add attention pooling support.
            self.pooling_layer = MoleculeEmbedding("attention", hidden_dim=hidden_dim)
        # The transformer type does not need a pooling layer, so it is not initialized here.
    
    def forward(self, model_output, attention_mask=None):
        """
        Returns:
            embedding: (Batch, Seq_Len, Hidden_Dim) or (Batch, Hidden_Dim)
        """
        if self.embedding_type == "transformer":
            # Return full sequence representations for downstream transformer processing.
            return model_output.last_hidden_state
        
        elif self.embedding_type in ["mean", "max", "attention"]:
            # Reuse the pre-initialized pooling layer with no extra overhead.
            return self.pooling_layer(model_output, attention_mask)
        
        else:
            raise ValueError(f"Unknown protein embedding type: {self.embedding_type}")
