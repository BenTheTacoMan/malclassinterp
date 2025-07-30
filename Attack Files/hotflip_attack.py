"""
HotFlip Adversarial Attack for Malware Classifier
================================================

This module implements white-box HotFlip adversarial attacks on Transformer-based
malware classifiers. HotFlip is a token-level attack that computes gradients with
respect to input embeddings and flips tokens to maximize the loss.

Key Features:
- Token-level adversarial attacks with saliency-based token selection
- Support for different attack strengths (gentle, medium, aggressive)
- Curriculum-based attack configurations for consistency with training
- Comprehensive attack statistics and reporting
- Single file and batch processing modes
- Integration with hierarchical function sampling

Attack Configurations:
- Gentle: 5% of tokens flipped (low success rate, precise changes)
- Medium: 10% of tokens flipped (moderate success rate)
- Aggressive: 20% of tokens flipped (high success rate, extensive changes)

Usage:
    # Single file attack
    python hotflip_attack.py --model model.pt --vocab_path vocab.json \
           --input_json sample.json --output_json adv_sample.json
    
    # Batch attack on test set
    python hotflip_attack.py --model model.pt --vocab_path vocab.json \
           --data_glob "data/*.json" --adv_output_dir results/

Dependencies:
    - torch, numpy, tqdm
    - model.py, dataset.py, build_vocabulary.py
"""

import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
import os
import glob
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Model Files'))
from model import MalwareClassifier, OpcodeTokenizer, get_device
from dataset import MalwareDataset, hierarchical_sample
from tqdm import tqdm
import csv
import random
from build_vocabulary import tokenize_instruction_sequence

def set_all_seeds(seed):
    """Set seeds for all random number generators for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make cuDNN deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_hardcoded_attack_configs():
    """Get the hardcoded HotFlip attack configurations"""
    return [
        {'k_percent': 0.05, 'name': 'gentle', 'description': 'Gentle attack - 5% of tokens flipped'},
        {'k_percent': 0.10, 'name': 'medium', 'description': 'Medium attack - 10% of tokens flipped'},
        {'k_percent': 0.20, 'name': 'aggressive', 'description': 'Aggressive attack - 20% of tokens flipped'}
    ]

def hotflip_attack(model, tokenizer, func_tensor, label, k, debug=False):
    """
    Perform HotFlip attack on token sequences with masking for PAD, int3, <UNK> and enforced nop as only deletion.
    
    Args:
        model: The malware classifier model
        tokenizer: Tokenizer with vocab mapping
        func_tensor: Function token tensor (B, F, L) where B=batch, F=funcs, L=length
        label: True label (0 or 1) for untargeted attack
        k: Number of tokens to flip
        debug: Whether to print debug information
    
    Returns:
        adv_tensor: Adversarial token tensor (B, F, L)
        adv_logits: New logits after attack
        flipped_positions: List of (batch_idx, func_idx, token_idx) tuples that were flipped
        attack_info: Dictionary with attack statistics
    """
    B, F, L = func_tensor.shape
    device = func_tensor.device
    
    # CRITICAL: Validate that the input tensor has valid functions
    has_valid_functions = (func_tensor > 0).any()
    if not has_valid_functions:
        raise ValueError(f"HotFlip attack received empty tensor with no valid functions (all zeros). Shape: {func_tensor.shape}")
    
    # Retrieve banned token IDs and nop
    ban_pad = 0  # By convention, PAD is always 0
    ban_unk = tokenizer.opcode2idx['<UNK>']
    ban_int3 = tokenizer.opcode2idx['int3']
    nop_id = tokenizer.opcode2idx['nop']
    
    # Step 1: Forward pass to compute loss and gradients
    func_tensor_flat = func_tensor.view(B * F, L)
    func_tensor_flat.requires_grad_(False)  # Tokens don't need gradients
    
    # Get embeddings and enable gradients
    embedding_layer = model.func_encoder.embedding
    
    # Get raw embeddings and apply positional encoding first
    raw_emb = embedding_layer(func_tensor_flat)  # (B*F, L, D)
    if model.func_encoder.pos_encoding_type == 'sinusoidal':
        emb_t = raw_emb.transpose(0, 1)
        emb_pos = model.func_encoder.pos_enc(emb_t)
        emb = emb_pos.transpose(0, 1)
    elif model.func_encoder.pos_encoding_type == 'learned':
        batch_size, seq_len = func_tensor_flat.shape
        positions = torch.arange(seq_len, device=func_tensor_flat.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = model.func_encoder.pos_embedding(positions)
        emb = raw_emb + pos_emb
    else:
        # No positional encoding: use raw embeddings directly
        emb = raw_emb

    # Forward pass through transformer
    mask = (func_tensor_flat == 0)
    emb_transposed = emb.transpose(0, 1)  # (L, B*F, D)
    out = model.func_encoder.transformer(emb_transposed, src_key_padding_mask=mask)
    out = out.transpose(0, 1)  # (B*F, L, D)
    out = out.masked_fill(mask.unsqueeze(-1), 0)
    lengths = (~mask).sum(dim=1, keepdim=True)
    pooled = out.sum(dim=1) / lengths.clamp(min=1)
    
    # Reshape for binary aggregation
    pooled = pooled.view(B, F, -1)
    func_tensor_reshaped = func_tensor_flat.view(B, F, L)
    func_masks = (func_tensor_reshaped.sum(dim=-1) > 0).float()
    binary_emb = model.aggregator(pooled, mask=func_masks)
    logits = model.classifier(binary_emb).squeeze(-1)
    
    # Compute loss (untargeted attack - maximize loss to push away from true label)
    label_tensor = torch.full((B,), label, dtype=torch.float, device=device)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, label_tensor)
    
    # Step 2: Get gradients w.r.t. embeddings using autograd.grad (more reliable than backward + .grad)
    # This avoids leaf tensor issues and gives us exactly ∂loss/∂emb
    embedding_grads, = torch.autograd.grad(loss, emb, retain_graph=False)  # (B*F, L, D)
    
    # Step 3: Compute saliency scores (L2 norm of gradient vectors)
    saliency_scores = embedding_grads.norm(p=2, dim=-1)  # (B*F, L)
    saliency_scores = saliency_scores.view(B, F, L)
    
    # Mask out padding positions (don't flip padding tokens)
    token_mask = (func_tensor > 0).float()
    saliency_scores = saliency_scores * token_mask
    
    # Step 4: Rank all positions by saliency and select top k
    saliency_flat = saliency_scores.view(B, -1)  # (B, F*L)
    token_flat = func_tensor.view(B, -1)  # (B, F*L)
    
    # Get top k positions for each batch element
    _, top_k_indices = torch.topk(saliency_flat, k, dim=1)  # (B, k)
    
    # Convert flat indices back to (func_idx, token_idx)
    top_k_func_indices = top_k_indices // L  # (B, k)
    top_k_token_indices = top_k_indices % L   # (B, k)
    
    # Step 5: For each selected position, find best replacement token
    vocab_size = len(tokenizer.opcode2idx)
    embedding_weight = embedding_layer.weight.detach()  # (V, D)
    
    # Prepare adversarial tensor (start with original)
    adv_tensor = func_tensor.clone()
    flipped_positions = []
    
    attack_info = {
        'initial_loss': loss.item(),
        'k_flips_requested': k,
        'k_flips_applied': 0,
        'avg_saliency': saliency_scores.mean().item(),
        'max_saliency': saliency_scores.max().item()
    }
    
    for batch_idx in range(B):
        for pos_idx in range(k):
            func_idx = top_k_func_indices[batch_idx, pos_idx].item()
            token_idx = top_k_token_indices[batch_idx, pos_idx].item()
            
            # Skip if this is a padding position
            if func_tensor[batch_idx, func_idx, token_idx].item() == 0:
                continue
            
            # Get gradient at this position
            grad_pos = embedding_grads.view(B, F, L, -1)[batch_idx, func_idx, token_idx]  # (D,)
            
            # Step 6: Compute dot products with all vocab embeddings
            # For untargeted attack, we want to maximize loss, so use positive gradient
            dot_products = torch.matmul(grad_pos, embedding_weight.T)  # (V,)
            
            # --- Masking: Set PAD, int3, and <UNK> candidate scores to -inf so they are never chosen ---
            dot_products[ban_pad] = -float('inf')
            dot_products[ban_unk] = -float('inf')
            dot_products[ban_int3] = -float('inf')
            
            # Find the token that gives maximum increase in loss
            best_token_id = dot_products.argmax().item()
            
            # --- Enforce nop as only allowed deletion ---
            # If the best candidate is PAD or int3, flip to nop instead
            if best_token_id == ban_pad or best_token_id == ban_int3:
                best_token_id = nop_id
            
            # Skip if it's the same as current token (no change needed)
            current_token_id = func_tensor[batch_idx, func_idx, token_idx].item()
            if best_token_id == current_token_id:
                continue
            
            # Apply the flip
            adv_tensor[batch_idx, func_idx, token_idx] = best_token_id
            flipped_positions.append((batch_idx, func_idx, token_idx))
            attack_info['k_flips_applied'] += 1
    
    # Step 7: Forward pass with adversarial tokens to get new predictions
    with torch.no_grad():
        adv_logits = model(adv_tensor)
    
    attack_info['final_loss'] = None  # Could compute if needed
    attack_info['positions_flipped'] = len(flipped_positions)
    
    if debug:
        print(f"Requested {k} flips, applied {attack_info['k_flips_applied']}")
        print(f"Average saliency: {attack_info['avg_saliency']:.4f}")
        print(f"Max saliency: {attack_info['max_saliency']:.4f}")
        print(f"Positions flipped: {len(flipped_positions)}")
    
    return adv_tensor, adv_logits, flipped_positions, attack_info

# --- Unit test for hotflip_attack (commented out to avoid production issues) ---
# def _unit_test_hotflip():
#     class DummyTokenizer:
#         def __init__(self):
#             self.opcode2idx = {'<PAD>': 0, '<UNK>': 1, 'int3': 2, 'nop': 3, 'mov': 4, 'add': 5, 'sub': 6}
#             self.idx2opcode = {v: k for k, v in self.opcode2idx.items()}
#     class DummyModel(torch.nn.Module):
#         def __init__(self, vocab_size, d=8):
#             super().__init__()
#             self.func_encoder = type('', (), {})()
#             self.func_encoder.embedding = torch.nn.Embedding(vocab_size, d)
#             self.func_encoder.pos_encoding_type = 'none'
#             self.func_encoder.transformer = torch.nn.Identity()
#             self.aggregator = lambda x, mask=None: x.sum(dim=1)
#             self.classifier = torch.nn.Linear(d, 1)
#         def forward(self, x):
#             # x: (B, F, L)
#             B, F, L = x.shape
#             emb = self.func_encoder.embedding(x.view(B*F, L))
#             pooled = emb.mean(dim=1).view(B, F, -1)
#             agg = self.aggregator(pooled)
#             return self.classifier(agg)
#     # Dummy data
#     tokenizer = DummyTokenizer()
#     model = DummyModel(vocab_size=len(tokenizer.opcode2idx))
#     B, F, L = 2, 3, 4
#     # All tokens are 'mov', 'add', 'sub', except for some 'nop', 'int3', '<UNK>'
#     func_tensor = torch.tensor([
#         [[4, 5, 6, 3], [4, 5, 6, 0], [4, 5, 6, 0]],
#         [[4, 5, 6, 1], [4, 2, 6, 0], [4, 5, 6, 0]]
#     ])
#     label = 1
#     k = 3
#     adv_tensor, adv_logits, flipped_positions, attack_info = hotflip_attack(model, tokenizer, func_tensor, label, k, debug=True)
#     # Check that no <UNK> or int3 in output
#     banned_ids = {tokenizer.opcode2idx['<UNK>'], tokenizer.opcode2idx['int3']}
#     assert not any(tok in banned_ids for tok in adv_tensor.flatten().tolist()), "Banned tokens <UNK> or int3 found in output!"
#     # Check that if any pad (0) appears, it was not a flip (should only be original padding)
#     # Check that nop is present as a deletion
#     print("Unit test passed: No banned tokens in adversarial output, nop used for deletion.")

def split_dataset(all_files, train_ratio=0.6, val_ratio=0.2, seed=42):
    # Use Python's random module for consistency with training scripts
    import random as py_random
    py_random.seed(seed)
    all_files = list(all_files)
    py_random.shuffle(all_files)
    n = len(all_files)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    train_files = all_files[:n_train]
    val_files = all_files[n_train:n_train+n_val]
    test_files = all_files[n_train+n_val:]
    return train_files, val_files, test_files

def filter_no_func(files, no_func_path):
    if not os.path.exists(no_func_path):
        return files
    with open(no_func_path) as f:
        no_func_files = set(line.strip().replace('\\', '/') for line in f if line.strip())
    return [f for f in files if f.replace('\\', '/') not in no_func_files]

def run_single_file(args, model, tokenizer, device):
    with open(args.input_json, 'r') as f:
        data = json.load(f)
    functions = [func['instructions'] for func in data.get('functions', [])]
    # Use tokenization consistent with training
    if args.use_split_tokens:
        functions = [tokenize_instruction_sequence(instrs, use_boundaries=True) for instrs in functions]
    # else: keep original instruction format (merged tokens)
    label = data.get('label', 0)
    
    # Check for empty functions
    non_empty_functions = [f for f in functions if f and len(f) > 0]
    if len(non_empty_functions) == 0:
        print(f"ERROR: File has no valid functions: {args.input_json}")
        return
    
    # Use hierarchical sampling (consistent with training)
    seed = getattr(args, 'seed', 42)
    if len(functions) > args.max_funcs:
        functions = hierarchical_sample(functions, args.max_funcs, n_buckets=4, seed=seed)
    elif len(functions) < args.max_funcs:
        functions = functions + [[]] * (args.max_funcs - len(functions))
    print(f"Using {len([f for f in functions if f])} non-empty functions out of {args.max_funcs} slots")
    func_tokens = [tokenizer.encode(f, args.max_func_len) for f in functions]
    func_tensor = torch.tensor(func_tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        orig_logits = model(func_tensor)
        orig_prob = torch.sigmoid(orig_logits).item()
        orig_pred = int(orig_prob > 0.5)
    print(f"Original prediction: {orig_prob:.4f} (label: {orig_pred})")
    
    if label == 0:
        print("Benign file: no attack performed.")
        return
    
    # Calculate k based on percentage of valid tokens
    total_valid_tokens = (func_tensor > 0).sum().item()
    k = max(1, int(args.k_percent * total_valid_tokens))
    print(f"HotFlip attack: flipping {k} tokens out of {total_valid_tokens} valid tokens ({args.k_percent:.1%})")
    
    # HotFlip attack for malicious files
    print(f"Performing HotFlip attack with k={k}...")
    adv_tensor, adv_logits, flipped_positions, attack_info = hotflip_attack(
        model, tokenizer, func_tensor, label, k, debug=True
    )
    
    # Evaluate attack success
    with torch.no_grad():
        adv_prob = torch.sigmoid(adv_logits).item()
        adv_pred = int(adv_prob > 0.5)
    
    print(f"\nHotFlip attack completed:")
    print(f"Initial loss: {attack_info['initial_loss']:.6f}")
    print(f"Tokens flipped: {attack_info['k_flips_applied']}/{k}")
    print(f"Adversarial prediction: {adv_prob:.4f} (label: {adv_pred})")
    print(f"Label flipped: {orig_pred != adv_pred}")
    
    if orig_pred != adv_pred:
        # Only save adversarial files if not disabled
        if not args.no_save_files:
            # Convert back to opcodes and save
            idx2opcode = tokenizer.idx2opcode
            adv_functions = []
            for func_idx in range(args.max_funcs):
                adv_func = []
                for tok_idx in range(args.max_func_len):
                    tok_id = adv_tensor[0, func_idx, tok_idx].item()
                    if tok_id == 0:
                        continue
                    opcode = idx2opcode.get(tok_id, '<UNK>')
                    adv_func.append(opcode)
                adv_functions.append(adv_func)
            
            adv_json = data.copy()
            for i, func in enumerate(adv_json.get('functions', [])):
                func['instructions'] = adv_functions[i] if i < len(adv_functions) else []
            
            with open(args.output_json, 'w') as f:
                json.dump(adv_json, f, indent=2)
            print(f"Adversarial example saved to {args.output_json}")
            
            # Save attack info
            attack_info_path = args.output_json.replace('.json', '_attack_info.json')
            attack_info['flipped_positions'] = flipped_positions
            with open(attack_info_path, 'w') as f:
                json.dump(attack_info, f, indent=2)
            print(f"Attack information saved to {attack_info_path}")
        else:
            print("Attack successful (label flipped) but files not saved due to --no_save_files flag")
    else:
        print("Attack failed: label did not flip.")

def run_batch(args, model, tokenizer, device):
    all_files = glob.glob(args.data_glob)
    seed = getattr(args, 'seed', 42)
    train_files, val_files, test_files = split_dataset(all_files, seed=seed)
    test_files = filter_no_func(test_files, args.no_func_path)
    
    os.makedirs(args.adv_output_dir, exist_ok=True)
    
    # Only create saliency directory if we're saving files
    if not args.no_save_files:
        saliency_npy_dir = args.saliency_npy_dir or os.path.join(args.adv_output_dir, 'saliency_npys')
        os.makedirs(saliency_npy_dir, exist_ok=True)
    else:
        print("[INFO] --no_save_files enabled: Skipping adversarial JSON and saliency file creation to save disk space")
    
    # Single pass: process files and attack only those that are malicious AND correctly predicted
    print("=== SINGLE PASS: Processing files and attacking set S ===")
    print("S = {files where true_label == 1 AND orig_pred == 1}")
    
    # Set model to eval mode
    model.eval()
    
    summary = []
    benign_count = 0
    malicious_but_wrong_pred = 0
    malicious_correct_pred = 0
    total_attacked = 0  # |S|
    successful = 0  # number of flips within S
    debug_every = 100  # Show progress every 100 files
    
    for idx, file_path in enumerate(tqdm(test_files, desc="Processing and attacking")):
        # Process each file and decide whether to attack it
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            functions = [func['instructions'] for func in data.get('functions', [])]
            # Use tokenization consistent with training
            if args.use_split_tokens:
                functions = [tokenize_instruction_sequence(instrs, use_boundaries=True) for instrs in functions]
            label = data.get('label', 0)
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"Warning: Skipping {file_path} due to file error: {e}")
            continue
        
        # Skip files with no valid functions (all empty)
        non_empty_functions = [f for f in functions if f and len(f) > 0]
        if len(non_empty_functions) == 0:
            continue  # Skip silently for cleaner output
        
        # Use hierarchical sampling (consistent with training)
        if len(functions) > args.max_funcs:
            functions = hierarchical_sample(functions, args.max_funcs, n_buckets=4, seed=seed)
        elif len(functions) < args.max_funcs:
            functions = functions + [[]] * (args.max_funcs - len(functions))
        func_tokens = [tokenizer.encode(f, args.max_func_len) for f in functions]
        func_tensor = torch.tensor(func_tokens, dtype=torch.long, device=device).unsqueeze(0)
        
        # Get original prediction (no gradients needed)
        try:
            with torch.no_grad():
                orig_logits = model(func_tensor)
                orig_prob = torch.sigmoid(orig_logits).item()
                orig_pred = int(orig_prob > 0.5)
        except (RuntimeError, ValueError) as e:
            print(f"Warning: Skipping {file_path} due to model inference error: {e}")
            continue
        
        # Initialize result structure
        adv_result = {
            'file': file_path,
            'label': label,
            'orig_prob': orig_prob,
            'orig_pred': orig_pred,
            'adv_prob': None,
            'adv_pred': None,
            'label_flipped': None,
            'tokens_flipped': None,
            'k_requested': None,
            'adv_file': None,
            'attack_info': None,
            'hotflip_k_percent': args.k_percent,
            'config_name': args.config_name
        }
        
        # Categorize and handle each file type
        if label == 0:
            # Benign file - no attack needed
            benign_count += 1
            adv_result['adv_prob'] = orig_prob
            adv_result['adv_pred'] = orig_pred
            adv_result['label_flipped'] = False
            adv_result['tokens_flipped'] = 0
            adv_result['k_requested'] = 0
            summary.append(adv_result)  # Record benign files
            continue
        
        elif label == 1:
            if orig_pred != 1:
                # Malicious but model predicted wrong - not in set S
                malicious_but_wrong_pred += 1
                adv_result['adv_prob'] = orig_prob
                adv_result['adv_pred'] = orig_pred
                adv_result['label_flipped'] = False
                adv_result['tokens_flipped'] = 0
                adv_result['k_requested'] = 0
                adv_result['attack_info'] = {
                    'k_flips_requested': 0,
                    'k_flips_applied': 0,
                    'skipped_reason': 'model_prediction_mismatch'
                }
                summary.append(adv_result)  # Record mismatched files
                continue
            
            else:
                # This file is in set S: malicious AND correctly predicted as malicious
                malicious_correct_pred += 1
                total_attacked += 1  # Count this as an attack attempt
                
                # Calculate k based on percentage of valid tokens
                total_valid_tokens = (func_tensor > 0).sum().item()
                k = max(1, int(args.k_percent * total_valid_tokens))
                adv_result['k_requested'] = k
                
                # HotFlip attack for files in set S
                try:
                    adv_tensor, adv_logits, flipped_positions, attack_info = hotflip_attack(
                        model, tokenizer, func_tensor, label, k, debug=(idx % debug_every == 0)
                    )
                    
                    # Evaluate attack success
                    with torch.no_grad():
                        adv_prob = torch.sigmoid(adv_logits).item()
                        adv_pred = int(adv_prob > 0.5)
                    
                    adv_result['adv_prob'] = adv_prob
                    adv_result['adv_pred'] = adv_pred
                    adv_result['label_flipped'] = (orig_pred != adv_pred)
                    adv_result['tokens_flipped'] = attack_info['k_flips_applied']
                    adv_result['attack_info'] = attack_info
                    
                    # Track success
                    if orig_pred != adv_pred:
                        successful += 1
                        
                        # Only save adversarial files if not disabled
                        if not args.no_save_files:
                            # Convert back to opcodes and save
                            idx2opcode = tokenizer.idx2opcode
                            adv_functions = []
                            for func_idx in range(args.max_funcs):
                                adv_func = []
                                for tok_idx in range(args.max_func_len):
                                    tok_id = adv_tensor[0, func_idx, tok_idx].item()
                                    if tok_id == 0:
                                        continue
                                    opcode = idx2opcode.get(tok_id, '<UNK>')
                                    adv_func.append(opcode)
                                adv_functions.append(adv_func)
                            
                            adv_json = data.copy()
                            for i, func in enumerate(adv_json.get('functions', [])):
                                func['instructions'] = adv_functions[i] if i < len(adv_functions) else []
                            
                            adv_filename = os.path.join(args.adv_output_dir, os.path.basename(file_path))
                            with open(adv_filename, 'w') as f:
                                json.dump(adv_json, f, indent=2)
                            adv_result['adv_file'] = adv_filename
                        else:
                            adv_result['adv_file'] = "not_saved"
                
                except (RuntimeError, ValueError) as e:
                    print(f"Warning: HotFlip attack failed for {file_path}: {e}")
                    adv_result['adv_prob'] = orig_prob
                    adv_result['adv_pred'] = orig_pred
                    adv_result['label_flipped'] = False
                    adv_result['tokens_flipped'] = 0
                    adv_result['attack_info'] = {
                        'k_flips_requested': k,
                        'k_flips_applied': 0,
                        'error': str(e)
                    }
        
        # Progress reporting
        if idx % debug_every == 0:
            print(f"Processed {idx}/{len(test_files)} files | Attacks: {total_attacked} | Successes: {successful}")
        
        summary.append(adv_result)
    
    summary_path = os.path.join(args.adv_output_dir, 'hotflip_attack_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")
    
    print("\n=== HotFlip Batch Attack Summary ===")
    print(f"Total test files: {len(test_files)}")
    print(f"Benign files: {benign_count}")
    print(f"Malicious but wrong prediction: {malicious_but_wrong_pred}")
    print(f"Malicious and correct prediction: {malicious_correct_pred}")
    print(f"Attack set S size: |S| = {total_attacked}")
    print(f"HotFlip attacks attempted: {total_attacked}")
    print(f"Successful attacks (label flipped): {successful}")
    
    if total_attacked > 0:
        # Calculate robust accuracy and attack success rate
        attack_success_rate = successful / total_attacked
        robust_accuracy = 1.0 - attack_success_rate
        
        print(f"Attack success rate: {attack_success_rate:.2%}")
        print(f"Robust accuracy: {robust_accuracy:.2%}")
        
        # Save results
        success_rate_path = os.path.join(args.adv_output_dir, 'success_rate.txt')
        with open(success_rate_path, 'w') as f:
            f.write(f"attack_success_rate:{attack_success_rate:.4f}\n")
            f.write(f"robust_accuracy:{robust_accuracy:.4f}\n")
        print(f"Results saved to: {success_rate_path}")
    else:
        print("No attacks attempted! (total_attacked == 0)")
        print("This means no malicious files were correctly predicted by the model.")
        # Save 0 success rate if no attacks
        success_rate_path = os.path.join(args.adv_output_dir, 'success_rate.txt')
        with open(success_rate_path, 'w') as f:
            f.write("attack_success_rate:0.0000\n")
            f.write("robust_accuracy:1.0000\n")
        print(f"Results saved to: {success_rate_path}")
    print(f"Summary file: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="HotFlip adversarial attack for Transformer malware classifier")
    parser.add_argument('--model', type=str, required=True, help='Path to trained .pt model')
    parser.add_argument('--vocab_path', type=str, required=True, help='Path to vocab file (json)')
    parser.add_argument('--data_glob', type=str, help='Glob for all JSON files (for test split)')
    parser.add_argument('--no_func_path', type=str, default='no_functions_files.txt', help='Path to no_functions_files.txt')
    parser.add_argument('--adv_output_dir', type=str, help='Where to save adversarial JSONs (malicious only)')
    parser.add_argument('--k_percent', type=float, default=0.10, help='Percentage of tokens to flip (0.0 to 1.0)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for split')
    parser.add_argument('--max_func_len', type=int, default=64)
    parser.add_argument('--max_funcs', type=int, default=64)
    parser.add_argument('--input_json', type=str, help='(Optional) Single file mode: run on one JSON file')
    parser.add_argument('--output_json', type=str, help='(Optional) Output for single file mode')
    parser.add_argument('--saliency_npy_dir', type=str, help='Directory to save saliency npy files (default: adv_output_dir/saliency_npys)')
    parser.add_argument('--use_attention', action='store_true', help='Use attention in MalwareClassifier (default: mean pooling)')
    parser.add_argument('--pos_encoding', type=str, default='sinusoidal', choices=['sinusoidal', 'learned', 'none'],
                        help='Type of positional encoding to use')
    parser.add_argument('--ablation_name', type=str, default=None, help='Ablation name for logging and output organization')
    parser.add_argument('--use_split_tokens', dest='use_split_tokens', action='store_true', help='Use atomic token splitting approach (recommended for better generalization)')
    parser.add_argument('--no-use_split_tokens', dest='use_split_tokens', action='store_false', help='Use merged tokens (original approach)')
    parser.set_defaults(use_split_tokens=True)
    parser.add_argument('--no_save_files', action='store_true', help='Skip saving adversarial JSONs and saliency files to save disk space (only keep summary and success rate)')
    parser.add_argument('--use_manual_params', action='store_true', help='Use manual parameters instead of curriculum configs (for testing arbitrary k values)')
    args = parser.parse_args()

    # Set all seeds for reproducibility
    set_all_seeds(args.seed)
    print(f"Set all random seeds to {args.seed}")

    # Print attack config
    ablation_name = args.ablation_name or f"pos-{args.pos_encoding}_attn-{'attn' if args.use_attention else 'mean'}"
    print(f"Positional Encoding: {args.pos_encoding}, Aggregator: {'Attention' if args.use_attention else 'Mean'}")
    print(f"Ablation Name: {ablation_name}")
    
    # Check if we should use curriculum configs or respect manual parameters
    if args.use_manual_params:
        print(f"Using manual parameters (--use_manual_params flag set)")
        print(f"  Manual: k_percent={args.k_percent}")
        config_name = f"manual_k{args.k_percent:.3f}"
    else:
        # HARDCODED: Always use curriculum configs for consistency with training
        # Override manual parameters with curriculum configurations
        curriculum_configs = get_hardcoded_attack_configs()
        
        # Find the config that matches the manual parameters (for backward compatibility)
        # If manual params don't match any curriculum config, use the medium config
        matching_config = None
        for config in curriculum_configs:
            if abs(config['k_percent'] - args.k_percent) < 0.001:
                matching_config = config
                break
        
        if matching_config:
            print(f"Manual parameters match curriculum config: {matching_config['name']}")
            print(f"  Using: k_percent={matching_config['k_percent']}")
            # Use the matching config
            args.k_percent = matching_config['k_percent']
            config_name = matching_config['name']
        else:
            print(f"Manual parameters (k_percent={args.k_percent}) don't match curriculum")
            print(f"OVERRIDING with curriculum MEDIUM config for consistency")
            medium_config = curriculum_configs[1]  # Medium is index 1
            args.k_percent = medium_config['k_percent']
            config_name = medium_config['name']
            print(f"  Using: k_percent={args.k_percent} ({config_name})")
        
        print(f"This ensures consistency with adversarial training curriculum")
        print(f"Expected result: Correct attack strength vs success rate relationship")
    
    # Store config name for later use
    args.config_name = config_name
    
    # DEBUG: Print final configuration
    print(f"Final HotFlip config -> k_percent={args.k_percent}")
    
    print(f"Using vocab file: {args.vocab_path}")
    print(f"Using seed: {args.seed} (deterministic mode enabled)")
    print(f"Tokenization style: {'atomic/split' if args.use_split_tokens else 'merged/original'}")
    if ('atomic' in args.vocab_path and not args.use_split_tokens) or ('merged' in args.vocab_path and args.use_split_tokens):
        print("Warning: Vocab file and tokenization style may be mismatched! Check your settings.")

    device = get_device()
    print(f"Using device: {device}")
    print(f"Model use_attention: {args.use_attention}")
    print(f"Model pos_encoding: {args.pos_encoding}")

    with open(args.vocab_path, 'r') as f:
        vocab = json.load(f)
    # Handle vocab format: if it's a dict, convert to list
    if isinstance(vocab, dict):
        print("Warning: vocab.json contains a dict, converting to list")
        opcode_list = list(vocab.keys()) if vocab else []
    else:
        print("Info: vocab.json contains a list (expected format)")
        opcode_list = vocab
    print(f"Vocabulary size: {len(opcode_list)}")
    tokenizer = OpcodeTokenizer(opcode_list)
    
    model = MalwareClassifier(
        opcode_list,
        max_func_len=args.max_func_len,
        max_funcs=args.max_funcs,
        use_attention=args.use_attention,
        pos_encoding=args.pos_encoding
    )
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()

    # Ensure output directory exists
    adv_output_dir = args.adv_output_dir
    if not adv_output_dir:
        adv_output_dir = os.path.join('hotflip_results', f"{ablation_name}_{args.config_name}")
        print(f"--adv_output_dir not provided. Using default: {adv_output_dir}")
    else:
        # Append config name to provided directory for clarity
        adv_output_dir = f"{adv_output_dir}_{args.config_name}"
        print(f"Appending config name to output directory: {adv_output_dir}")
    os.makedirs(adv_output_dir, exist_ok=True)
    args.adv_output_dir = adv_output_dir
    if args.saliency_npy_dir:
        os.makedirs(args.saliency_npy_dir, exist_ok=True)
    else:
        args.saliency_npy_dir = os.path.join(adv_output_dir, 'saliency_npys')
        os.makedirs(args.saliency_npy_dir, exist_ok=True)

    if args.input_json:
        args.output_json = args.output_json or os.path.join(args.adv_output_dir or '.', 'hotflip_example.json')
        run_single_file(args, model, tokenizer, device)
    else:
        assert args.data_glob and args.adv_output_dir, 'Batch mode requires --data_glob and --adv_output_dir'
        run_batch(args, model, tokenizer, device)

if __name__ == "__main__":
    main() 