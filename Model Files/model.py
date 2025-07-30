"""
Malware Classifier Model Module
------------------------------
Implements transformer-based neural network models for malware classification.

This module includes:
- MalwareClassifier: Main classification model with transformer architecture
- FunctionEncoder: Encodes individual function instruction sequences
- BinaryAggregator: Aggregates function embeddings to binary-level
- OpcodeTokenizer: Tokenizes instruction sequences
- PositionalEncoding: Adds positional information to sequences

Dependencies:
- torch
- torch.nn
- torch.nn.functional
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional

# Configuration constants
DEFAULT_D_MODEL = 256
DEFAULT_NHEAD = 8
DEFAULT_NUM_LAYERS = 2
DEFAULT_MAX_FUNC_LEN = 64
DEFAULT_MAX_FUNCS = 64
DEFAULT_DROPOUT = 0.2
DEFAULT_MAX_LEN = 5000
DEFAULT_PADDING_TOKEN = 0
DEFAULT_POS_ENCODING = 'sinusoidal'

# Helper to select device (GPU if available)
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from "Attention is All You Need"
    """
    def __init__(self, d_model: int, max_len: int = DEFAULT_MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch, d_model)
        return x + self.pe[:x.size(0), :]

class OpcodeTokenizer:
    """
    Simple tokenizer for opcode sequences.
    Maps opcodes to integer tokens.
    """
    def __init__(self, opcode_vocab: List[str]):
        self.opcode2idx = {op: i+1 for i, op in enumerate(opcode_vocab)}  # 0 is reserved for padding
        self.idx2opcode = {i+1: op for i, op in enumerate(opcode_vocab)}
        self.pad_token = DEFAULT_PADDING_TOKEN
        # Find <UNK> token index (should always exist)
        if '<UNK>' not in self.opcode2idx:
            raise ValueError("Vocabulary must contain '<UNK>' token!")
        self.unk_token = self.opcode2idx['<UNK>']

    def encode(self, opcodes: List[str], max_len: int) -> List[int]:
        tokens = [self.opcode2idx.get(op, self.unk_token) for op in opcodes]
        if len(tokens) < max_len:
            tokens += [self.pad_token] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
        return tokens

    def vocab_size(self):
        return len(self.opcode2idx) + 1  # +1 for padding

class MyTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_attn_weights = None

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = src
        attn_output, attn_weights = self.self_attn(
            x, x, x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        self.last_attn_weights = attn_weights
        x = attn_output
        x = self.dropout1(x)
        x = self.norm1(src + x)
        x = self.norm2(x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x))))))
        return x

class FunctionEncoder(nn.Module):
    """
    Encodes a sequence of opcodes (tokens) for a single function using a Transformer encoder.
    Now includes positional encoding!
    """
    def __init__(self, vocab_size: int, d_model: int = DEFAULT_D_MODEL, nhead: int = DEFAULT_NHEAD, num_layers: int = DEFAULT_NUM_LAYERS, max_len: int = DEFAULT_MAX_FUNC_LEN, pos_encoding: str = DEFAULT_POS_ENCODING):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding_type = pos_encoding
        if pos_encoding == 'sinusoidal':
            self.pos_enc = PositionalEncoding(d_model, max_len)
        elif pos_encoding == 'learned':
            self.pos_embedding = nn.Embedding(max_len, d_model)
        elif pos_encoding == 'none':
            self.pos_enc = None
        else:
            raise ValueError(f"Unknown pos_encoding: {pos_encoding}. Use 'sinusoidal', 'learned', or 'none'")
        encoder_layer = MyTransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.max_len = max_len
        self.d_model = d_model

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)  # (batch, seq_len, d_model)
        if self.pos_encoding_type == 'sinusoidal':
            emb = emb.transpose(0, 1)  # (seq_len, batch, d_model)
            emb = self.pos_enc(emb)
            emb = emb.transpose(0, 1)  # back to (batch, seq_len, d_model)
        elif self.pos_encoding_type == 'learned':
            batch_size, seq_len = x.shape
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)  # (batch, seq_len)
            pos_emb = self.pos_embedding(positions)  # (batch, seq_len, d_model)
            emb = emb + pos_emb
        mask = (x == 0)  # padding mask
        out = self.transformer(emb, src_key_padding_mask=mask)
        out = out.masked_fill(mask.unsqueeze(-1), 0)
        lengths = (~mask).sum(dim=1, keepdim=True)
        pooled = out.sum(dim=1) / lengths.clamp(min=1)
        return pooled  # (batch, d_model)

class BinaryAggregator(nn.Module):
    """
    Aggregates function embeddings for a binary into a single embedding.
    Supports mean pooling or attention.
    """
    def __init__(self, d_model: int, use_attention: bool = False):
        super().__init__()
        self.use_attention = use_attention
        if use_attention:
            self.attn = nn.Linear(d_model, 1)

    def forward(self, func_embs, mask=None):
        # func_embs: (batch, num_funcs, d_model)
        if self.use_attention:
            attn_scores = self.attn(func_embs).squeeze(-1)  # (batch, num_funcs)
            if mask is not None:
                attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
                all_masked = (mask.sum(dim=1) == 0)  # (batch,)
                if all_masked.any():
                    attn_scores[all_masked] = 0.0
            attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # (batch, num_funcs, 1)
            agg = (func_embs * attn_weights).sum(dim=1)
        else:
            if mask is not None:
                func_embs = func_embs * mask.unsqueeze(-1)
                lengths = mask.sum(dim=1, keepdim=True)
            else:
                lengths = func_embs.new_full((func_embs.size(0), 1), func_embs.size(1))
            agg = func_embs.sum(dim=1) / lengths.clamp(min=1)
        return agg  # (batch, d_model)

class MalwareClassifier(nn.Module):
    """
    Full model: opcode tokenization, function encoding, binary aggregation, and classification.
    """
    def __init__(self, opcode_vocab: List[str],
                 d_model: int = DEFAULT_D_MODEL,
                 nhead: int = DEFAULT_NHEAD,
                 num_layers: int = DEFAULT_NUM_LAYERS,
                 max_func_len: int = DEFAULT_MAX_FUNC_LEN,
                 max_funcs: int = DEFAULT_MAX_FUNCS,
                 use_attention: bool = False,
                 dropout: float = DEFAULT_DROPOUT,
                 pos_encoding: str = DEFAULT_POS_ENCODING):
        super().__init__()
        self.tokenizer = OpcodeTokenizer(opcode_vocab)
        tokenizer_vocab_size = self.tokenizer.vocab_size()
        self.func_encoder = FunctionEncoder(tokenizer_vocab_size, d_model, nhead, num_layers, max_func_len, pos_encoding)
        self.aggregator = BinaryAggregator(d_model, use_attention)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        self.max_func_len = max_func_len
        self.max_funcs = max_funcs

    def forward(self, batch_funcs, device: Optional[torch.device] = None):
        # batch_funcs can be either:
        # 1. List[List[List[str]]] - raw instructions (for inference)
        # 2. torch.Tensor - already tokenized (from DataLoader)
        
        if isinstance(batch_funcs, torch.Tensor):
            # Already tokenized by collate_batch
            func_tensors = batch_funcs
            if device is not None:
                func_tensors = func_tensors.to(device)
            # Create mask (1 for non-empty functions, 0 for padded)
            func_masks = (func_tensors.sum(dim=-1) > 0).float()
        else:
            # Raw string data - tokenize here
            batch_size = len(batch_funcs)
            func_tensors = []
            func_masks = []
            for funcs in batch_funcs:
                # Pad/truncate to max_funcs
                original_len = len(funcs)
                if len(funcs) < self.max_funcs:
                    funcs = funcs + [[]] * (self.max_funcs - len(funcs))
                    func_mask = [1] * original_len + [0] * (self.max_funcs - original_len)
                else:
                    funcs = funcs[:self.max_funcs]
                    func_mask = [1] * self.max_funcs
                # Tokenize each function
                func_tokens = [self.tokenizer.encode(f, self.max_func_len) for f in funcs]
                func_tensors.append(torch.tensor(func_tokens, dtype=torch.long))
                func_masks.append(torch.tensor(func_mask, dtype=torch.float))
            func_tensors = torch.stack(func_tensors)  # (batch, max_funcs, max_func_len)
            func_masks = torch.stack(func_masks)  # (batch, max_funcs)
            if device is not None:
                func_tensors = func_tensors.to(device)
                func_masks = func_masks.to(device)
        # Encode each function
        B, F, L = func_tensors.shape
        func_tensors = func_tensors.view(B * F, L)
        func_embs = self.func_encoder(func_tensors)  # (B*F, d_model)
        func_embs = func_embs.view(B, F, -1)
        # Aggregate to binary-level embedding
        binary_embs = self.aggregator(func_embs, mask=func_masks)
        # Classify
        logits = self.classifier(binary_embs).squeeze(-1)
        return logits  # (batch,)

