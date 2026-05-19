import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(-1)  # (B, L, 1)
            x_masked = x * mask
            sum_embeddings = x_masked.sum(dim=1)
            sum_mask = mask.sum(dim=1).clamp(min=1e-9)
            return sum_embeddings / sum_mask
        else:
            return x.mean(dim=1)


class AttentionPooling(nn.Module):
    """
    Attention Pooling.
    Instead of simple averaging, the model learns which sequence positions matter more.
    Useful when key features appear only at local positions, such as a binding-critical amino acid site.
    """
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        # A simple linear layer for computing each position's importance score.
        self.attention_weights = nn.Linear(input_dim, 1)

    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len, input_dim)
        # mask shape: (batch_size, seq_len) -> 1 for valid, 0 for padding
        
        # 1. Compute raw scores: (batch_size, seq_len, 1)
        scores = self.attention_weights(x)
        
        # 2. If a mask exists, set padding scores to negative infinity so softmax weights become 0.
        if mask is not None:
            # Expand mask dimensions to match scores.
            mask = mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 3. Compute normalized weights (softmax): (batch_size, seq_len, 1)
        attn_weights = F.softmax(scores, dim=1)
        
        # 4. Weighted sum: (batch_size, input_dim)
        # sum( (batch, seq, dim) * (batch, seq, 1) ) -> (batch, dim)
        pooled = torch.sum(x * attn_weights, dim=1)
        
        return pooled


class MaxMeanPooling(nn.Module):
    """
    Concatenated max and mean pooling.
    Keeps both background information (mean) and the most salient features (max).
    Useful when both a global summary and extreme feature values matter.
    Note: the output dimension doubles and usually needs a downstream Linear layer for reduction.
    """
    def __init__(self):
        super(MaxMeanPooling, self).__init__()

    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len, input_dim)
        
        if mask is not None:
            mask = mask.unsqueeze(-1) # (batch_size, seq_len, 1)
            # For mean pooling, set padding to 0.
            x_masked_mean = x * mask
            # Compute true lengths for averaging.
            lengths = mask.sum(dim=1)
            mean_pooled = x_masked_mean.sum(dim=1) / lengths.clamp(min=1e-9)
            
            # For max pooling, set padding to a very small negative value.
            x_masked_max = x.masked_fill(mask == 0, -1e9)
            max_pooled = x_masked_max.max(dim=1)[0]
        else:
            mean_pooled = x.mean(dim=1)
            max_pooled = x.max(dim=1)[0]
            
        # Concatenate: (batch_size, input_dim * 2)
        return torch.cat([mean_pooled, max_pooled], dim=1)


class CLSPooling(nn.Module):
    """
    CLS token pooling.
    Directly takes the first sequence vector.
    Assumes the first sequence position is a special token representing global information,
    such as when 'mol_feat' is placed first.
    Useful in Transformer architectures where the first position already integrates later sequence information through attention.
    """
    def __init__(self):
        super(CLSPooling, self).__init__()

    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len, input_dim)
        # Directly take the first position: (batch_size, input_dim)
        return x[:, 0, :]


# Factory function for convenient use.
def get_pooling_layer(method_name, input_dim):
    if method_name == 'mean':
        return MeanPooling()
    elif method_name == 'attention':
        return AttentionPooling(input_dim)
    elif method_name == 'max_mean':
        return MaxMeanPooling()
    elif method_name == 'cls':
        return CLSPooling()
    else:
        raise ValueError(f"Unknown pooling method: {method_name}")
