"""
Malware Classifier Test Script
-----------------------------
Tests a trained malware classifier on a test dataset and generates performance metrics.

Dependencies:
- Python 3.6+
- torch
- numpy
- tqdm
- scikit-learn
- model.py, dataset.py (must be present in the same directory)

Usage:
    python test_model.py --model model.pt --vocab_path vocab.json --data_glob './data/*/disassembled/*.json'
"""
import os
import glob
import random
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Model Files'))
from model import MalwareClassifier, get_device, OpcodeTokenizer
from dataset import MalwareDataset, collate_batch
from tqdm import tqdm
import time
from functools import partial
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np
import argparse

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

def collate_fn(batch, tokenizer, max_func_len=64, max_funcs=64):
    return collate_batch(batch, tokenizer, max_func_len, max_funcs)

def test_model(data_glob, use_attention=False, pos_encoding='sinusoidal', log_path=None, ablation_name=None, vocab_path=None, use_split_tokens=True, results_dir='.', seed=42):
    device = get_device()
    all_files = sorted(glob.glob(data_glob))
    no_func_path = os.path.join(os.path.dirname(__file__), 'no_functions_files.txt')
    if os.path.exists(no_func_path):
        with open(no_func_path, 'r') as nf:
            no_func_files = set(line.strip().replace('\\', '/') for line in nf if line.strip())
        def filter_no_func(files):
            return [f for f in files if f.replace('\\', '/') not in no_func_files]
        all_files = filter_no_func(all_files)
    train_files, val_files, test_files = split_dataset(all_files, seed=seed)
    assert set(train_files).isdisjoint(test_files), "Train/Test overlap!"
    assert set(val_files).isdisjoint(test_files),   "Val/Test overlap!"
    if vocab_path:
        if not os.path.exists(vocab_path):
            print(f"ERROR: Vocabulary file {vocab_path} not found!")
            return
        with open(vocab_path, 'r') as vf:
            opcode_vocab = json.load(vf)
    else:
        ablation_name = ablation_name or f"pos-{pos_encoding}_attn-{'attn' if use_attention else 'mean'}"
        vocab_path = f'vocab_{ablation_name}.json'
        if not os.path.exists(vocab_path):
            print(f"ERROR: Vocabulary file {vocab_path} not found!")
            return
        with open(vocab_path, 'r') as vf:
            opcode_vocab = json.load(vf)
    tokenizer = OpcodeTokenizer(opcode_vocab)
    max_func_len = 64
    max_funcs = 64
    batch_size = 64
    test_dataset = MalwareDataset(test_files, tokenizer, max_func_len, max_funcs, cache_in_memory=False, use_split_tokens=use_split_tokens, use_boundaries=True, seed=seed)
    collate_fn_with_tokenizer = partial(collate_batch, tokenizer=tokenizer, max_func_len=max_func_len, max_funcs=max_funcs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_with_tokenizer, num_workers=8)
    if ablation_name:
        best_model_path = os.path.join(results_dir, f'best_model_{ablation_name}.pt')
    else:
        best_model_path = os.path.join(results_dir, 'best_model.pt')
    if not os.path.exists(best_model_path):
        print(f"ERROR: Best model file {best_model_path} not found!")
        return
    model = MalwareClassifier(opcode_vocab, d_model=256, nhead=8, num_layers=2, 
                             max_func_len=max_func_len, max_funcs=max_funcs, dropout=0.2, use_attention=use_attention, pos_encoding=pos_encoding).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    start_time = time.time()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    test_loss = 0
    criterion = torch.nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for batch_idx, (batch_funcs, batch_labels) in enumerate(tqdm(test_loader, desc="Testing")):
            batch_labels = batch_labels.to(device)
            logits = model(batch_funcs, device=device)
            loss = criterion(logits, batch_labels)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).float()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            test_loss += loss.item() * batch_labels.size(0)
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    avg_test_loss = test_loss / len(test_dataset)
    test_time = time.time() - start_time
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Test Loss:     {avg_test_loss:.6f}")
    print(f"Accuracy:      {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision:     {precision:.4f}")
    print(f"Recall:        {recall:.4f}")
    print(f"F1-Score:      {f1:.4f}")
    print(f"Test Time:     {test_time:.2f} seconds")
    print(f"Samples/sec:   {len(test_dataset)/test_time:.2f}")
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 0     1")
    print(f"Actual    0   {conf_matrix[0,0]:4d}  {conf_matrix[0,1]:4d}")
    print(f"          1   {conf_matrix[1,0]:4d}  {conf_matrix[1,1]:4d}")
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    print(f"\nDetailed Metrics:")
    print(f"True Positives:  {tp}")
    print(f"True Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"Specificity:     {specificity:.4f}")
    print(f"NPV:             {npv:.4f}")
    benign_count = int(np.sum(all_labels == 0))
    malicious_count = int(np.sum(all_labels == 1))
    print(f"\nTest Set Distribution:")
    print(f"Benign samples:    {benign_count} ({benign_count/len(all_labels)*100:.1f}%)")
    print(f"Malicious samples: {malicious_count} ({malicious_count/len(all_labels)*100:.1f}%)")
    print(f"\nProbability Analysis:")
    print(f"Min probability:   {all_probabilities.min():.4f}")
    print(f"Max probability:   {all_probabilities.max():.4f}")
    print(f"Mean probability:  {all_probabilities.mean():.4f}")
    print(f"Std probability:   {all_probabilities.std():.4f}")
    confident_predictions = np.sum((all_probabilities < 0.1) | (all_probabilities > 0.9))
    print(f"Confident predictions (p<0.1 or p>0.9): {confident_predictions}/{len(all_probabilities)} ({confident_predictions/len(all_probabilities)*100:.1f}%)")
    results = {
        'test_loss': avg_test_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix.tolist(),
        'test_time': test_time,
        'samples_per_second': len(test_dataset)/test_time,
        'total_samples': len(test_dataset),
        'benign_samples': benign_count,
        'malicious_samples': malicious_count,
        'specificity': specificity,
        'npv': npv,
        'probability_stats': {
            'min': float(all_probabilities.min()),
            'max': float(all_probabilities.max()),
            'mean': float(all_probabilities.mean()),
            'std': float(all_probabilities.std())
        }
    }
    results_path = os.path.join(results_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {results_path}")
    print("\n" + "="*60)
    print("Classification Report:")
    print("="*60)
    print(classification_report(all_labels, all_predictions, target_names=['Benign', 'Malicious']))
    failures = []
    for idx, (pred, label, prob) in enumerate(zip(all_predictions, all_labels, all_probabilities)):
        if pred != label:
            if idx < len(test_files):
                file_path = test_files[idx]
            else:
                file_path = f"unknown_file_{idx}"
            reason = "misclassification"
            failures.append({
                "file": file_path,
                "true_label": int(label),
                "predicted_label": int(pred),
                "probability": float(prob),
                "reason": reason
            })
    failures_path = os.path.join(results_dir, "failures.json")
    with open(failures_path, 'w') as f:
        json.dump(failures, f, indent=2)
    print(f"\nFailure details saved to {failures_path}")
    if log_path:
        log_entry = {
            'ablation': ablation_name,
            'test_loss': avg_test_loss,
            'test_acc': accuracy
        }
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                log_data = json.load(f)
        else:
            log_data = []
        log_data.append(log_entry)
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Malware Classifier")
    parser.add_argument('--data_glob', type=str, required=True, help='Glob pattern for test data files')
    parser.add_argument('--use_attention', action='store_true', help='Use attention in binary aggregator')
    parser.add_argument('--pos_encoding', type=str, default='sinusoidal', choices=['sinusoidal', 'learned', 'none'],
                        help='Type of positional encoding to use (sinusoidal, learned, or none)')
    parser.add_argument('--log_path', type=str, default=None, help='Path to JSON log file')
    parser.add_argument('--ablation_name', type=str, default=None, help='Ablation name for logging')
    parser.add_argument('--vocab_path', type=str, default=None, help='Path to vocabulary file (overrides ablation default)')
    parser.add_argument('--use_split_tokens', dest='use_split_tokens', action='store_true', help='Use atomic token splitting approach (recommended for better generalization)')
    parser.add_argument('--no-use_split_tokens', dest='use_split_tokens', action='store_false', help='Use merged tokens (original approach)')
    parser.set_defaults(use_split_tokens=True)
    parser.add_argument('--results_dir', type=str, default='.', help='Directory to save results for this ablation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    if args.vocab_path:
        vocab_path = args.vocab_path
    else:
        ablation_name = args.ablation_name or f"pos-{args.pos_encoding}_attn-{'attn' if args.use_attention else 'mean'}"
        vocab_path = os.path.join(args.results_dir, f'vocab_{ablation_name}.json')
    if ('atomic' in vocab_path and not args.use_split_tokens) or ('merged' in vocab_path and args.use_split_tokens):
        print("Warning: Vocab file and tokenization style may be mismatched! Check your settings.")
    test_model(args.data_glob, use_attention=args.use_attention, pos_encoding=args.pos_encoding, log_path=args.log_path, ablation_name=args.ablation_name, vocab_path=vocab_path, use_split_tokens=args.use_split_tokens, results_dir=args.results_dir, seed=args.seed) 