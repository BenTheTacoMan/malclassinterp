"""
Transfer Attack Analysis for Malware Classifier
=============================================

This module implements transfer attack analysis on Transformer-based malware classifiers.
It evaluates model robustness by testing adversarial examples generated from one model
against other models to measure transferability.

Usage:
    # Single file transfer attack
    python transfer_attack.py --source_model model1.pt --target_model model2.pt \
           --vocab_path vocab.json --input_json sample.json --output_json adv_sample.json
    
    # Batch transfer attack on test set
    python transfer_attack.py --source_model model1.pt --target_model model2.pt \
           --vocab_path vocab.json --data_glob "data/*.json" --adv_output_dir results/

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
from hotflip_attack import hotflip_attack, get_hardcoded_attack_configs

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_hotflip_model(model_path, use_attention, pos_encoding, max_func_len, max_funcs, device):
    vocab = load_vocab('bigVocab.json')
    model = MalwareClassifier(
        vocab,
        d_model=256,
        nhead=8,
        num_layers=2,
        max_func_len=max_func_len,
        max_funcs=max_funcs,
        dropout=0.2,
        use_attention=use_attention,
        pos_encoding=pos_encoding
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    tokenizer = OpcodeTokenizer(vocab)
    model.tokenizer = tokenizer
    return model, tokenizer

def run_transfer_attack(args):
    device = get_device()
    print(f"Using device: {device}")
    print(f"Loading source model from: {args.source_model}")
    source_model, source_tokenizer = load_hotflip_model(
        args.source_model, args.source_use_attention, args.source_pos_encoding,
        args.max_func_len, args.max_funcs, device
    )
    print(f"Loading target model from: {args.target_model}")
    target_model, target_tokenizer = load_hotflip_model(
        args.target_model, args.target_use_attention, args.target_pos_encoding,
        args.max_func_len, args.max_funcs, device
    )
    all_files = sorted(glob.glob(args.data_glob))
    print(f"Found {len(all_files)} files before filtering.")
    no_func_path = os.path.join(os.path.dirname(__file__), 'no_functions_files.txt')
    if os.path.exists(no_func_path):
        with open(no_func_path, 'r') as nf:
            no_func_files = set(line.strip().replace('\\', '/') for line in nf if line.strip())
        def filter_no_func(files):
            return [f for f in files if f.replace('\\', '/') not in no_func_files]
        all_files = filter_no_func(all_files)
        print(f"After filtering: {len(all_files)} files remain.")
    else:
        print(f"Warning: {no_func_path} not found. No files filtered.")
    
    def split_dataset(all_files, train_ratio=0.6, val_ratio=0.2, seed=42):
        random.seed(seed)
        random.shuffle(all_files)
        n = len(all_files)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)
        train_files = all_files[:n_train]
        val_files = all_files[n_train:n_train+n_val]
        test_files = all_files[n_train+n_val:]
        return train_files, val_files, test_files
    
    train_files, val_files, test_files = split_dataset(all_files, seed=args.seed)
    test_files = test_files[:args.max_test_files] if args.max_test_files else test_files
    print(f"Testing transfer attack on {len(test_files)} files")
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = []
    source_success = 0
    target_success = 0
    transfer_success = 0
    malicious_count = 0
    benign_count = 0
    debug_every = 10
    attack_configs = get_hardcoded_attack_configs()
    total_malicious = 0
    
    for idx, file_path in enumerate(tqdm(test_files, desc="Transfer Attack")):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except:
            continue
        
        functions = [func['instructions'] for func in data.get('functions', [])]
        functions = [tokenize_instruction_sequence(instrs, use_boundaries=True) for instrs in functions]
        label = data.get('label', 0)
        non_empty_functions = [f for f in functions if f and len(f) > 0]
        if len(non_empty_functions) == 0:
            continue
        
        if len(functions) > args.max_funcs:
            functions = hierarchical_sample(functions, args.max_funcs, n_buckets=4, seed=args.seed)
        elif len(functions) < args.max_funcs:
            functions = functions + [[]] * (args.max_funcs - len(functions))
        
        func_tokens = [source_tokenizer.encode(f, args.max_func_len) for f in functions]
        func_tensor = torch.tensor(func_tokens, dtype=torch.long, device=device).unsqueeze(0)
        
        with torch.no_grad():
            source_orig_logits = source_model(func_tensor)
            source_orig_prob = torch.sigmoid(source_orig_logits).item()
            source_orig_pred = int(source_orig_prob > 0.5)
            target_orig_logits = target_model(func_tensor)
            target_orig_prob = torch.sigmoid(target_orig_logits).item()
            target_orig_pred = int(target_orig_prob > 0.5)
        
        result = {
            'file': file_path,
            'label': label,
            'source_orig_prob': source_orig_prob,
            'source_orig_pred': source_orig_pred,
            'target_orig_prob': target_orig_prob,
            'target_orig_pred': target_orig_pred,
            'source_adv_prob': None,
            'source_adv_pred': None,
            'target_adv_prob': None,
            'target_adv_pred': None,
            'source_attack_success': False,
            'target_attack_success': False,
            'transfer_success': False,
            'attack_info': None
        }
        
        if label == 1:
            malicious_count += 1
            total_malicious += 1
            attack_config = attack_configs[(total_malicious - 1) % len(attack_configs)]
            total_valid_tokens = (func_tensor > 0).sum().item()
            k = max(1, int(attack_config['k_percent'] * total_valid_tokens))
            
            try:
                adv_tensor, adv_logits, flipped_positions, attack_info = hotflip_attack(
                    source_model, source_tokenizer, func_tensor, 1, k, debug=False
                )
                
                with torch.no_grad():
                    adv_pred = (torch.sigmoid(adv_logits) > 0.5).float()
                
                if source_orig_pred == 1 and adv_pred.item() == 0:
                    source_success += 1
                    
                    # Test on target model
                    with torch.no_grad():
                        target_adv_logits = target_model(adv_tensor)
                        target_adv_prob = torch.sigmoid(target_adv_logits).item()
                        target_adv_pred = int(target_adv_prob > 0.5)
                    
                    target_attack_success = (target_orig_pred != target_adv_pred)
                    if target_attack_success:
                        target_success += 1
                    
                    transfer_attack_success = (target_attack_success and (source_orig_pred != adv_pred.item()))
                    if transfer_attack_success:
                        transfer_success += 1
                    
                    result.update({
                        'source_adv_prob': torch.sigmoid(adv_logits).item(),
                        'source_adv_pred': int(adv_pred.item()),
                        'target_adv_prob': target_adv_prob,
                        'target_adv_pred': target_adv_pred,
                        'source_attack_success': True,
                        'target_attack_success': target_attack_success,
                        'transfer_success': transfer_attack_success,
                        'attack_info': attack_info
                    })
                else:
                    result['attack_info'] = attack_info
            except Exception as e:
                result['attack_info'] = {'error': str(e)}
        else:
            benign_count += 1
            if not args.attack_benign:
                results.append(result)
                continue
        
        if idx % debug_every == 0:
            print(f"File {idx}: Source {source_orig_prob:.3f}->{result['source_adv_prob']}, "
                  f"Target {target_orig_prob}->{result['target_adv_prob']}, Transfer: {result['transfer_success']}")
        
        results.append(result)
    
    summary_path = os.path.join(args.output_dir, 'transfer_attack_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    total_attacked = malicious_count + (benign_count if args.attack_benign else 0)
    if total_attacked > 0:
        source_success_rate = source_success / total_attacked
        target_success_rate = target_success / total_attacked
        transfer_success_rate = transfer_success / total_attacked
        transfer_rate_conditional = transfer_success / source_success if source_success > 0 else 0.0
        
        print(f"\n=== TRANSFER ATTACK RESULTS ===")
        print(f"Total files attacked: {total_attacked}")
        print(f"Malicious files: {malicious_count}, Benign files: {benign_count if args.attack_benign else 0}")
        print(f"Source model attack success: {source_success}/{total_attacked} ({source_success_rate:.4f})")
        print(f"Target model attack success: {target_success}/{total_attacked} ({target_success_rate:.4f})")
        print(f"Transfer attack success: {transfer_success}/{total_attacked} ({transfer_success_rate:.4f})")
        print(f"Transfer rate (given source success): {transfer_success}/{source_success} ({transfer_rate_conditional:.4f})")
        
        stats = {
            'total_attacked': total_attacked,
            'malicious_count': malicious_count,
            'benign_count': benign_count if args.attack_benign else 0,
            'source_success_rate': source_success_rate,
            'target_success_rate': target_success_rate,
            'transfer_success_rate': transfer_success_rate,
            'transfer_rate_conditional': transfer_rate_conditional,
            'source_model': args.source_model,
            'target_model': args.target_model,
            'attack_benign': args.attack_benign,
            'max_test_files': args.max_test_files,
            'total_test_files': len(test_files)
        }
        stats_path = os.path.join(args.output_dir, 'transfer_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Summary saved to: {summary_path}")
        print(f"Statistics saved to: {stats_path}")
    else:
        print("No files found to attack!")

def main():
    parser = argparse.ArgumentParser(description="HotFlip-based transfer attack for malware classifier")
    parser.add_argument('--source_model', type=str, required=True, help='Path to source model .pt file (HotFlip)')
    parser.add_argument('--target_model', type=str, required=True, help='Path to target model .pt file (HotFlip)')
    parser.add_argument('--source_use_attention', action='store_true', help='Source model uses attention')
    parser.add_argument('--source_pos_encoding', type=str, required=True, choices=['sinusoidal', 'learned', 'none'], help='Source model pos encoding')
    parser.add_argument('--target_use_attention', action='store_true', help='Target model uses attention')
    parser.add_argument('--target_pos_encoding', type=str, required=True, choices=['sinusoidal', 'learned', 'none'], help='Target model pos encoding')
    parser.add_argument('--data_glob', type=str, required=True, help='Glob for all JSON files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--max_test_files', type=int, default=None, help='Maximum number of test files to use (for faster execution)')
    parser.add_argument('--max_func_len', type=int, default=64)
    parser.add_argument('--max_funcs', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--attack_benign', action='store_true', help='Also attack benign files (benign->malicious)')
    args = parser.parse_args()
    
    set_all_seeds(args.seed)
    print(f"Set all random seeds to {args.seed}")
    print(f"Transfer Attack Configuration:")
    print(f"Source Model: {args.source_model}")
    print(f"  - Attention: {args.source_use_attention}")
    print(f"  - Pos Encoding: {args.source_pos_encoding}")
    print(f"Target Model: {args.target_model}")
    print(f"  - Attention: {args.target_use_attention}")
    print(f"  - Pos Encoding: {args.target_pos_encoding}")
    if args.max_test_files:
        print(f"Max test files: {args.max_test_files} (sampling for speed)")
    
    run_transfer_attack(args)

if __name__ == "__main__":
    main() 