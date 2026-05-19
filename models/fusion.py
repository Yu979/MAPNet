import torch
import torch.nn as nn

class FusionFactory:
    @staticmethod
    def get_fusion_module(fusion_type: str, fusion_dim: int, **kwargs):
        if fusion_type == "transformer":
            return TransformerFusion(fusion_dim, **kwargs)
        elif fusion_type == "concat":
            return ConcatFusion()
        elif fusion_type == "bilinear":
            return BilinearFusion(fusion_dim)
        elif fusion_type == "cross_attention":
            return CrossAttentionFusion(fusion_dim, **kwargs)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        
# class LearnablePositionalEncoding(nn.Module):
#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 50):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
#         # Initialize with a truncated normal distribution to help early training converge stably.
#         nn.init.trunc_normal_(self.pe, std=0.02) 

#     def forward(self, x):
#         """
#         x shape: (batch_size, seq_len, d_model)
#         """
#         # Slice positional encodings to the current batch sequence length and add them to features.
#         x = x + self.pe[:, :x.size(1), :]
#         return self.dropout(x)


class TransformerFusion(nn.Module):
    def __init__(self, fusion_dim: int, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        
        # self.pos_encoder = LearnablePositionalEncoding(d_model=fusion_dim, max_len=50)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim, 
            nhead=n_heads, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
    
    def forward(self, combined_feat, mask=None):
        # combined_feat = self.pos_encoder(combined_feat)
        if mask is not None:
            key_padding_mask = (mask == 0).bool()
        else:
            key_padding_mask = None
        
        return self.transformer(combined_feat, src_key_padding_mask=key_padding_mask)

class ConcatFusion(nn.Module):
    """Simple concatenation."""
    def forward(self, mol_feat, prot_feat, mask=None):
        return torch.cat([mol_feat, prot_feat], dim=1)

class BilinearFusion(nn.Module):
    """Bilinear fusion."""
    def __init__(self, dim: int):
        super().__init__()
        self.bilinear = nn.Bilinear(dim, dim, dim)
    
    def forward(self, mol_feat, prot_feat, mask=None):
        # mol_feat: (B, 1, D), prot_feat: (B, L, D)
        mol_expanded = mol_feat.expand(-1, prot_feat.size(1), -1)
        fused = self.bilinear(mol_expanded, prot_feat)
        return fused

class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion."""
    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
    
    def forward(self, mol_feat, prot_feat, mask=None):
        # Use the molecule as query, and the protein as key and value.
        attn_output, _ = self.cross_attn(
            mol_feat, prot_feat, prot_feat,
            key_padding_mask=(mask == 0).bool() if mask is not None else None
        )
        return torch.cat([attn_output, prot_feat], dim=1)
