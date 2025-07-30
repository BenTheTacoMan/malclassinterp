"""
Malware Dataset Module
---------------------
Provides dataset classes and utilities for loading and processing malware disassembly data.

This module includes:
- MalwareDataset: PyTorch Dataset for loading disassembled malware JSONs
- Hierarchical sampling functions for function selection
- Caching mechanisms for improved performance
- Tokenization integration with build_vocabulary module

Dependencies:
- torch
- tqdm
- build_vocabulary (for tokenization)
"""
import json
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from functools import lru_cache
import random
from build_vocabulary import tokenize_instruction_sequence

# Configuration constants
DEFAULT_MAX_FUNCS = 64
DEFAULT_MAX_FUNC_LEN = 128
DEFAULT_N_BUCKETS = 4
DEFAULT_SEED = 42
DEFAULT_LABEL = 0
DEFAULT_PADDING_INDEX = -1

# Hierarchical sampling for function selection
def hierarchical_sample(funcs, max_funcs=DEFAULT_MAX_FUNCS, n_buckets=DEFAULT_N_BUCKETS, seed=DEFAULT_SEED):
    # Set seed for reproducible sampling
    random.seed(seed)
    
    N = len(funcs)
    if N == 0:
        return [[] for _ in range(max_funcs)]  # pad with empty if no functions
    bucket_size = N // n_buckets
    per_bucket = max_funcs // n_buckets
    sampled = []
    for i in range(n_buckets):
        start = i * bucket_size
        end   = (i+1)*bucket_size if i < n_buckets-1 else N
        bucket = funcs[start:end]
        if len(bucket) <= per_bucket:
            sampled.extend(bucket)
        else:
            sampled.extend(random.sample(bucket, per_bucket))
    # If we undershot max_funcs, pad with random from remaining
    if len(sampled) < max_funcs:
        remaining = list(set(tuple(f) for f in funcs) - set(tuple(f) for f in sampled))
        if remaining:
            sampled.extend(random.sample(remaining, min(len(remaining), max_funcs - len(sampled))))
    return sampled[:max_funcs]

def hierarchical_sample_with_mapping(functions, max_funcs=DEFAULT_MAX_FUNCS, n_buckets=DEFAULT_N_BUCKETS, seed=DEFAULT_SEED):
    """
    Like hierarchical_sample, but returns both:
      - sampled_funcs: list of function dicts/lists
      - mapping: list of original JSON indices (length = max_funcs, padded with -1)
    """
    # Set seed for reproducible sampling
    random.seed(seed)
    
    N = len(functions)
    if N == 0:
        return [[] for _ in range(max_funcs)], [-1] * max_funcs
    
    # 1) Build a list of indices [0,1,2,...,len(functions)-1]
    all_idxs = list(range(N))
    
    # 2) Apply hierarchical sampling to indices (reuse the existing logic)
    sampled_idxs = hierarchical_sample(all_idxs, max_funcs, n_buckets, seed)
    
    # 3) Grab the actual functions using the sampled indices
    sampled_funcs = [functions[i] for i in sampled_idxs]
    
    # 4) Pad mapping with -1 so it's always length max_funcs
    padded_mapping = sampled_idxs + [-1] * (max_funcs - len(sampled_idxs))
    
    return sampled_funcs, padded_mapping

class MalwareDataset(Dataset):
    def __init__(self, file_list, tokenizer, max_func_len=DEFAULT_MAX_FUNC_LEN, max_funcs=DEFAULT_MAX_FUNCS, cache_in_memory=True, max_cache_size=None, n_buckets=DEFAULT_N_BUCKETS, use_split_tokens=True, use_boundaries=True, seed=DEFAULT_SEED):
        self.file_list = file_list
        self.tokenizer = tokenizer
        self.max_func_len = max_func_len
        self.max_funcs = max_funcs
        self.cache_in_memory = cache_in_memory
        self.max_cache_size = max_cache_size
        self.n_buckets = n_buckets
        self.use_split_tokens = use_split_tokens
        self.use_boundaries = use_boundaries
        self.seed = seed
        # Full memory caching
        if self.cache_in_memory and self.max_cache_size is None:
            print(f"[INFO] Loading {len(file_list)} files into memory...")
            self.cached_data = []
            for file_path in tqdm(file_list, desc="Caching files"):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    functions = [func['instructions'] for func in data.get('functions', [])]
                    # Apply atomic tokenization if enabled
                    if self.use_split_tokens:
                        functions = [tokenize_instruction_sequence(func, self.use_boundaries) for func in functions]
                    # Hierarchical sample if more than max_funcs
                    if len(functions) > self.max_funcs:
                        functions, mapping = hierarchical_sample_with_mapping(functions, self.max_funcs, self.n_buckets, seed=self.seed)
                    elif len(functions) < self.max_funcs:
                        functions = functions + [[]] * (self.max_funcs - len(functions))
                        mapping = list(range(len(functions))) + [-1] * (self.max_funcs - len(functions))
                    else:
                        mapping = list(range(len(functions)))
                    self.cached_data.append((functions, data.get('label', 0), mapping))
                except Exception as e:
                    print(f"[ERROR] Could not read {file_path}: {e}")
                    # Add dummy data for failed files
                    self.cached_data.append(([ [] for _ in range(self.max_funcs) ], 0, [-1] * self.max_funcs))
            print(f"[INFO] Successfully cached {len(self.cached_data)} files in memory!")
        # Partial LRU caching
        elif self.cache_in_memory and self.max_cache_size is not None:
            print(f"[INFO] Using LRU cache with max size: {self.max_cache_size}")
            self.cached_data = None
            # Create LRU cached loading function
            @lru_cache(maxsize=self.max_cache_size)
            def _load_file_cached(file_path):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    functions = [func['instructions'] for func in data.get('functions', [])]
                    # Apply atomic tokenization if enabled
                    if self.use_split_tokens:
                        functions = [tokenize_instruction_sequence(func, self.use_boundaries) for func in functions]
                    if len(functions) > self.max_funcs:
                        functions, mapping = hierarchical_sample_with_mapping(functions, self.max_funcs, self.n_buckets, seed=self.seed)
                    elif len(functions) < self.max_funcs:
                        functions = functions + [[]] * (self.max_funcs - len(functions))
                        mapping = list(range(len(functions))) + [-1] * (self.max_funcs - len(functions))
                    else:
                        mapping = list(range(len(functions)))
                    label = data.get('label', 0)
                    return (functions, label, mapping)
                except Exception as e:
                    print(f"[ERROR] Could not read {file_path}: {e}")
                    return ([ [] for _ in range(self.max_funcs) ], 0, [-1] * self.max_funcs)
            self._load_file_cached = _load_file_cached
        else:
            self.cached_data = None

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Backwards compatibility check
        cache_in_memory = getattr(self, 'cache_in_memory', False)
        cached_data = getattr(self, 'cached_data', None)
        if cache_in_memory and cached_data is not None:
            # Return fully cached data (no HDD access!)
            return cached_data[idx]
        elif cache_in_memory and hasattr(self, '_load_file_cached'):
            # Return LRU cached data (some HDD access)
            file_path = self.file_list[idx]
            return self._load_file_cached(file_path)
        else:
            # Original behavior - load from disk
            file_path = self.file_list[idx]
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[ERROR] Could not read {file_path}: {e}")
                # Return a dummy sample (empty functions, label 0)
                return [ [] for _ in range(self.max_funcs) ], 0, [-1] * self.max_funcs
            functions = [func['instructions'] for func in data.get('functions', [])]
            # Apply atomic tokenization if enabled
            if self.use_split_tokens:
                functions = [tokenize_instruction_sequence(func, self.use_boundaries) for func in functions]
            # Hierarchical sample if more than max_funcs
            if len(functions) > self.max_funcs:
                functions, mapping = hierarchical_sample_with_mapping(functions, self.max_funcs, self.n_buckets, seed=getattr(self, 'seed', 42))
            elif len(functions) < self.max_funcs:
                functions = functions + [[]] * (self.max_funcs - len(functions))
                mapping = list(range(len(functions))) + [-1] * (self.max_funcs - len(functions))
            else:
                mapping = list(range(len(functions)))
            label = data.get('label', 0)
            return functions, label, mapping

def collate_batch(batch, tokenizer, max_func_len=128, max_funcs=64):
    batch_funcs, batch_labels, batch_mappings = zip(*batch)
    # Pad/truncate functions per binary
    padded_funcs = []
    for funcs in batch_funcs:
        if len(funcs) < max_funcs:
            funcs = funcs + [[]] * (max_funcs - len(funcs))
        else:
            funcs = funcs[:max_funcs]
        
        # Tokenize each function
        func_tokens = [tokenizer.encode(f, max_func_len) for f in funcs]
        padded_funcs.append(func_tokens)
    
    batch_tensor = torch.tensor(padded_funcs, dtype=torch.long)
    batch_labels = torch.tensor(batch_labels, dtype=torch.float)
    batch_mappings = torch.tensor(batch_mappings, dtype=torch.long)
    return batch_tensor, batch_labels, batch_mappings 