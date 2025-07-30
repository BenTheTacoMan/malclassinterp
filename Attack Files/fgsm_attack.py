"""
FGSM (Fast Gradient Sign Method) Attack for Malware Classifier
=============================================================

This module implements white-box FGSM adversarial attacks on Transformer-based
malware classifiers. It evaluates model robustness by computing gradients with
respect to input embeddings and perturbing them in the direction that maximizes
the loss.

Usage:
    # Single file attack
    python fgsm_attack.py --model model.pt --vocab_path vocab.json \
           --input_json sample.json --output_json adv_sample.json
    
    # Batch attack on test set
    python fgsm_attack.py --model model.pt --vocab_path vocab.json \
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

# Helper: Map perturbed embeddings back to closest vocab tokens
@torch.no_grad()
def embeddings_to_tokens(perturbed_emb, embedding_weight):
    # perturbed_emb: (num_tokens, d_model)
    # embedding_weight: (vocab_size, d_model)
    # Returns: (num_tokens,) int tensor of closest vocab indices
    normed_weight = F.normalize(embedding_weight, dim=-1)
    sim = torch.matmul(normed_emb, normed_weight.T)  # (num_tokens, vocab_size)
    closest = sim.argmax(dim=-1)
    return closest.cpu().tolist()

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
    # FGSM attack for malicious
    embedding_layer = model.func_encoder.embedding
    B, num_funcs, num_tokens = func_tensor.shape
    func_tensor_flat = func_tensor.view(B * num_funcs, num_tokens)
    emb = embedding_layer(func_tensor_flat)
    
    # CRITICAL: Apply positional encoding (was missing!)
    if model.func_encoder.pos_encoding_type == 'sinusoidal':
        emb_transposed = emb.transpose(0, 1)  # (num_tokens, B*num_funcs, d_model)
        emb_with_pos = model.func_encoder.pos_enc(emb_transposed)
        emb = emb_with_pos.transpose(0, 1)  # back to (B*num_funcs, num_tokens, d_model)
        print("Applied sinusoidal positional encoding for attack")
    elif model.func_encoder.pos_encoding_type == 'learned':
        batch_size, seq_len = func_tensor_flat.shape
        positions = torch.arange(seq_len, device=func_tensor_flat.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = model.func_encoder.pos_embedding(positions)
        emb = emb + pos_emb
        print("Applied learned positional encoding for attack")
    else:
        print("No positional encoding applied for attack (pos_encoding='none')")
    
    emb.requires_grad_()
    emb.retain_grad()
    mask = (func_tensor_flat == 0)
    out = model.func_encoder.transformer(emb, src_key_padding_mask=mask)
    out = out.masked_fill(mask.unsqueeze(-1), 0)
    lengths = (~mask).sum(dim=1, keepdim=True)
    pooled = out.sum(dim=1) / lengths.clamp(min=1)
    pooled = pooled.view(B, num_funcs, -1)
    func_masks = (func_tensor.sum(dim=-1) > 0).float()
    binary_emb = model.aggregator(pooled, mask=func_masks)
    logits = model.classifier(binary_emb).squeeze(-1)
    label_tensor = torch.tensor([label], dtype=torch.float, device=device)
    # Ensure float32 for numerical stability
    logits = logits.float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, label_tensor)
    # Clear gradients before backward pass
    model.zero_grad()
    loss.backward()
    # Get embedding gradients (don't clip model gradients)
    grad = emb.grad.detach()  # (B*num_funcs, num_tokens, d_model)
    # --- Token Saliency Scoring ---
    saliency = grad.norm(p=2, dim=-1)  # (B*num_funcs, num_tokens)
    saliency = saliency.view(B, num_funcs, num_tokens)
    saliency_np = saliency.squeeze(0).cpu().numpy()  # (num_funcs, num_tokens)
    max_func_idx, max_tok_idx = np.unravel_index(np.argmax(saliency_np), saliency_np.shape)
    most_salient_token_id = func_tokens[max_func_idx][max_tok_idx]
    most_salient_opcode = tokenizer.idx2opcode.get(most_salient_token_id, '<UNK>')
    print(f"Most salient token: function {max_func_idx}, token {max_tok_idx}, opcode '{most_salient_opcode}' (saliency={saliency_np[max_func_idx, max_tok_idx]:.4f})")
    if getattr(args, 'saliency_output', None):
        np.save(args.saliency_output, saliency_np)
        print(f"Full saliency matrix saved to {args.saliency_output}")
    perturbed_emb = emb + args.epsilon * grad.sign()
    with torch.no_grad():
        out_adv = model.func_encoder.transformer(perturbed_emb, src_key_padding_mask=mask)
        out_adv = out_adv.masked_fill(mask.unsqueeze(-1), 0)
        pooled_adv = out_adv.sum(dim=1) / lengths.clamp(min=1)
        pooled_adv = pooled_adv.view(B, num_funcs, -1)
        binary_emb_adv = model.aggregator(pooled_adv, mask=func_masks)
        logits_adv = model.classifier(binary_emb_adv).squeeze(-1)
        prob_adv = torch.sigmoid(logits_adv).item()
        pred_adv = int(prob_adv > 0.5)
    print(f"Perturbed prediction: {prob_adv:.4f} (label: {pred_adv})")
    print(f"Label flipped: {orig_pred != pred_adv}")
    perturb_norm = (args.epsilon * grad.sign()).norm().item()
    print(f"Perturbation L2 norm: {perturb_norm:.4f}")
    embedding_weight = embedding_layer.weight.detach()
    perturbed_emb_flat = perturbed_emb.view(-1, perturbed_emb.shape[-1])
    new_token_ids = embeddings_to_tokens(perturbed_emb_flat, embedding_weight)
    new_token_ids = np.array(new_token_ids).reshape(num_funcs, num_tokens)
    idx2opcode = tokenizer.idx2opcode
    adv_functions = []
    for func_idx in range(args.max_funcs):
        adv_func = []
        for tok_idx in range(args.max_func_len):
            tok_id = new_token_ids[func_idx, tok_idx]
            if tok_id == 0:
                continue
            opcode = idx2opcode.get(tok_id, '<UNK>')
            adv_func.append(opcode)
        adv_functions.append(adv_func)
    if orig_pred != pred_adv:
        adv_json = data.copy()
        for i, func in enumerate(adv_json.get('functions', [])):
            func['instructions'] = adv_functions[i] if i < len(adv_functions) else []
        with open(args.output_json, 'w') as f:
            json.dump(adv_json, f, indent=2)
        print(f"Adversarial example saved to {args.output_json}")
    else:
        print("Attack failed: label did not flip. No adversarial file saved.")

def run_batch(args, model, tokenizer, device):
    all_files = glob.glob(args.data_glob)
    seed = getattr(args, 'seed', 42)
    train_files, val_files, test_files = split_dataset(all_files, seed=seed)
    test_files = filter_no_func(test_files, args.no_func_path)
    os.makedirs(args.adv_output_dir, exist_ok=True)
    saliency_npy_dir = args.saliency_npy_dir or os.path.join(args.adv_output_dir, 'saliency_npys')
    os.makedirs(saliency_npy_dir, exist_ok=True)
    summary = []
    benign_count = 0
    malicious_count = 0
    attack_success = 0
    debug_every = 10
    for idx, file_path in enumerate(tqdm(test_files, desc="FGSM Test Attack")):
        with open(file_path, 'r') as f:
            data = json.load(f)
        functions = [func['instructions'] for func in data.get('functions', [])]
        # Use tokenization consistent with training
        if args.use_split_tokens:
            functions = [tokenize_instruction_sequence(instrs, use_boundaries=True) for instrs in functions]
        # else: keep original instruction format (merged tokens)
        label = data.get('label', 0)
        
        # Skip files with no valid functions (all empty)
        non_empty_functions = [f for f in functions if f and len(f) > 0]
        if len(non_empty_functions) == 0:
            if idx % debug_every == 0:
                print(f"Skipping file with no valid functions: {file_path}")
            continue
        
        # Use hierarchical sampling (consistent with training)
        if len(functions) > args.max_funcs:
            functions = hierarchical_sample(functions, args.max_funcs, n_buckets=4, seed=seed)
        elif len(functions) < args.max_funcs:
            functions = functions + [[]] * (args.max_funcs - len(functions))
        func_tokens = [tokenizer.encode(f, args.max_func_len) for f in functions]
        func_tensor = torch.tensor(func_tokens, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            orig_logits = model(func_tensor)
            orig_prob = torch.sigmoid(orig_logits).item()
            orig_pred = int(orig_prob > 0.5)
        adv_result = {
            'file': file_path,
            'label': label,
            'orig_prob': orig_prob,
            'orig_pred': orig_pred,
            'adv_prob': None,
            'adv_pred': None,
            'label_flipped': None,
            'perturb_norm': None,
            'adv_file': None,
            'saliency_file': None,
            'max_saliency': None,
            'max_func_idx': None,
            'max_tok_idx': None,
            'max_opcode': None,
            'mean_saliency': None,
            'top5_opcodes': None
        }
        if label == 0:
            benign_count += 1
            adv_result['adv_prob'] = orig_prob
            adv_result['adv_pred'] = orig_pred
            adv_result['label_flipped'] = False
            summary.append(adv_result)
            if idx % debug_every == 0:
                print(f"Benign file: {file_path}, orig_prob={orig_prob:.4f}, pred={orig_pred}")
            continue
        malicious_count += 1
        embedding_layer = model.func_encoder.embedding
        B, num_funcs, num_tokens = func_tensor.shape
        func_tensor_flat = func_tensor.view(B * num_funcs, num_tokens)
        emb = embedding_layer(func_tensor_flat)
        
        # CRITICAL: Apply positional encoding (was missing!)
        if model.func_encoder.pos_encoding_type == 'sinusoidal':
            emb_transposed = emb.transpose(0, 1)  # (num_tokens, B*num_funcs, d_model)
            emb_with_pos = model.func_encoder.pos_enc(emb_transposed)
            emb = emb_with_pos.transpose(0, 1)  # back to (B*num_funcs, num_tokens, d_model)
        elif model.func_encoder.pos_encoding_type == 'learned':
            batch_size, seq_len = func_tensor_flat.shape
            positions = torch.arange(seq_len, device=func_tensor_flat.device).unsqueeze(0).expand(batch_size, -1)
            pos_emb = model.func_encoder.pos_embedding(positions)
            emb = emb + pos_emb
        # No debug prints in batch mode to avoid spam
        
        emb.requires_grad_()
        emb.retain_grad()
        mask = (func_tensor_flat == 0)
        out = model.func_encoder.transformer(emb, src_key_padding_mask=mask)
        out = out.masked_fill(mask.unsqueeze(-1), 0)
        lengths = (~mask).sum(dim=1, keepdim=True)
        pooled = out.sum(dim=1) / lengths.clamp(min=1)
        pooled = pooled.view(B, num_funcs, -1)
        func_masks = (func_tensor.sum(dim=-1) > 0).float()
        binary_emb = model.aggregator(pooled, mask=func_masks)
        logits = model.classifier(binary_emb).squeeze(-1)
        label_tensor = torch.tensor([label], dtype=torch.float, device=device)
        # Ensure float32 for numerical stability
        logits = logits.float()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, label_tensor)
        # Clear gradients before backward pass
        model.zero_grad()
        loss.backward()
        # Get embedding gradients (don't clip model gradients)
        grad = emb.grad.detach()
        saliency = grad.norm(p=2, dim=-1)
        saliency = saliency.view(B, num_funcs, num_tokens)
        saliency_np = saliency.squeeze(0).cpu().numpy()
        max_func_idx, max_tok_idx = np.unravel_index(np.argmax(saliency_np), saliency_np.shape)
        most_salient_token_id = func_tokens[max_func_idx][max_tok_idx]
        most_salient_opcode = tokenizer.idx2opcode.get(most_salient_token_id, '<UNK>')
        max_saliency = float(saliency_np[max_func_idx, max_tok_idx])
        mean_saliency = float(saliency_np.mean())
        flat_indices = saliency_np.flatten().argsort()[::-1][:5]
        top5_opcodes = []
        for flat_idx in flat_indices:
            func_idx = flat_idx // num_tokens
            tok_idx = flat_idx % num_tokens
            tok_id = func_tokens[func_idx][tok_idx]
            opcode = tokenizer.idx2opcode.get(tok_id, '<UNK>')
            top5_opcodes.append([opcode, float(saliency_np[func_idx, tok_idx])])
        saliency_file = os.path.join(saliency_npy_dir, os.path.basename(file_path) + '.saliency.npy')
        np.save(saliency_file, saliency_np)
        adv_result['saliency_file'] = saliency_file
        adv_result['max_saliency'] = max_saliency
        adv_result['max_func_idx'] = int(max_func_idx)
        adv_result['max_tok_idx'] = int(max_tok_idx)
        adv_result['max_opcode'] = most_salient_opcode
        adv_result['mean_saliency'] = mean_saliency
        adv_result['top5_opcodes'] = top5_opcodes
        if idx % debug_every == 0:
            print(f"Malicious file: {file_path}, orig_prob={orig_prob:.4f}, pred={orig_pred}, max_saliency={max_saliency:.4f}, mean_saliency={mean_saliency:.4f}, max_func={max_func_idx}, max_tok={max_tok_idx}, opcode={most_salient_opcode}")
            print(f"Top-5 opcodes: {top5_opcodes}")
        perturbed_emb = emb + args.epsilon * grad.sign()
        with torch.no_grad():
            out_adv = model.func_encoder.transformer(perturbed_emb, src_key_padding_mask=mask)
            out_adv = out_adv.masked_fill(mask.unsqueeze(-1), 0)
            pooled_adv = out_adv.sum(dim=1) / lengths.clamp(min=1)
            pooled_adv = pooled_adv.view(B, num_funcs, -1)
            binary_emb_adv = model.aggregator(pooled_adv, mask=func_masks)
            logits_adv = model.classifier(binary_emb_adv).squeeze(-1)
            prob_adv = torch.sigmoid(logits_adv).item()
            pred_adv = int(prob_adv > 0.5)
        adv_result['adv_prob'] = prob_adv
        adv_result['adv_pred'] = pred_adv
        adv_result['label_flipped'] = (orig_pred != pred_adv)
        adv_result['perturb_norm'] = (args.epsilon * grad.sign()).norm().item()
        if orig_pred != pred_adv:
            attack_success += 1
            adv_json = data.copy()
            adv_functions = []
            embedding_weight = embedding_layer.weight.detach()
            perturbed_emb_flat = perturbed_emb.view(-1, perturbed_emb.shape[-1])
            new_token_ids = embeddings_to_tokens(perturbed_emb_flat, embedding_weight)
            new_token_ids = np.array(new_token_ids).reshape(num_funcs, num_tokens)
            idx2opcode = tokenizer.idx2opcode
            for func_idx in range(args.max_funcs):
                adv_func = []
                for tok_idx in range(args.max_func_len):
                    tok_id = new_token_ids[func_idx, tok_idx]
                    if tok_id == 0:
                        continue
                    opcode = idx2opcode.get(tok_id, '<UNK>')
                    adv_func.append(opcode)
                adv_functions.append(adv_func)
            for i, func in enumerate(adv_json.get('functions', [])):
                func['instructions'] = adv_functions[i] if i < len(adv_functions) else []
            adv_filename = os.path.join(args.adv_output_dir, os.path.basename(file_path))
            with open(adv_filename, 'w') as f:
                json.dump(adv_json, f, indent=2)
            adv_result['adv_file'] = adv_filename
        summary.append(adv_result)
    summary_path = os.path.join(args.adv_output_dir, 'fgsm_attack_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")
    print("\n=== FGSM Batch Attack Summary ===")
    print(f"Total test files: {len(test_files)}")
    print(f"Benign files: {benign_count}")
    print(f"Malicious files: {malicious_count}")
    print(f"Successful attacks (label flipped): {attack_success}")
    if malicious_count > 0:
        print(f"Attack success rate: {attack_success/malicious_count:.2%}")
    print(f"Summary file: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="White-box FGSM adversarial attack for Transformer malware classifier (batch test mode)")
    parser.add_argument('--model', type=str, required=True, help='Path to trained .pt model')
    parser.add_argument('--vocab_path', type=str, required=True, help='Path to vocab file (json)')
    parser.add_argument('--data_glob', type=str, help='Glob for all JSON files (for test split)')
    parser.add_argument('--no_func_path', type=str, default='no_functions_files.txt', help='Path to no_functions_files.txt')
    parser.add_argument('--adv_output_dir', type=str, help='Where to save adversarial JSONs (malicious only)')
    parser.add_argument('--epsilon', type=float, default=0.1, help='FGSM epsilon (perturbation scale)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for split')
    parser.add_argument('--max_func_len', type=int, default=64)
    parser.add_argument('--max_funcs', type=int, default=64)
    parser.add_argument('--input_json', type=str, help='(Optional) Single file mode: run on one JSON file')
    parser.add_argument('--output_json', type=str, help='(Optional) Output for single file mode')
    parser.add_argument('--saliency_npy_dir', type=str, help='Directory to save all saliency npy files (default: adv_output_dir/saliency_npys)')
    parser.add_argument('--use_attention', action='store_true', help='Use attention in MalwareClassifier (default: mean pooling)')
    parser.add_argument('--pos_encoding', type=str, default='sinusoidal', choices=['sinusoidal', 'learned', 'none'],
                        help='Type of positional encoding to use')
    parser.add_argument('--ablation_name', type=str, default=None, help='Ablation name for logging and output organization')
    parser.add_argument('--use_split_tokens', dest='use_split_tokens', action='store_true', help='Use atomic token splitting approach (recommended for better generalization)')
    parser.add_argument('--no-use_split_tokens', dest='use_split_tokens', action='store_false', help='Use merged tokens (original approach)')
    parser.set_defaults(use_split_tokens=True)
    args = parser.parse_args()

    # Print ablation config
    ablation_name = args.ablation_name or f"pos-{args.pos_encoding}_attn-{'attn' if args.use_attention else 'mean'}"
    print(f"Positional Encoding: {args.pos_encoding}, Aggregator: {'Attention' if args.use_attention else 'Mean'}")
    print(f"Ablation Name: {ablation_name}")
    print(f"Using vocab file: {args.vocab_path}")
    print(f"Using seed: {args.seed}")
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
    
    # Warning about attention mechanism
    if args.use_attention:
        print("Warning: Attack with attention aggregation: gradients only flow through embeddings, not attention weights")
        print("Warning: This may result in suboptimal attacks for attention-based models")

    # Ensure output directory exists
    adv_output_dir = args.adv_output_dir
    if not adv_output_dir:
        adv_output_dir = os.path.join('fgsm_results', ablation_name)
        print(f"--adv_output_dir not provided. Using default: {adv_output_dir}")
    os.makedirs(adv_output_dir, exist_ok=True)
    args.adv_output_dir = adv_output_dir
    if args.saliency_npy_dir:
        os.makedirs(args.saliency_npy_dir, exist_ok=True)
    else:
        args.saliency_npy_dir = os.path.join(adv_output_dir, 'saliency_npys')
        os.makedirs(args.saliency_npy_dir, exist_ok=True)

    if args.input_json:
        args.output_json = args.output_json or os.path.join(args.adv_output_dir or '.', 'adv_example.json')
        run_single_file(args, model, tokenizer, device)
    else:
        assert args.data_glob and args.adv_output_dir, 'Batch mode requires --data_glob and --adv_output_dir'
        run_batch(args, model, tokenizer, device)

if __name__ == "__main__":
    main() 