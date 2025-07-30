"""
Unipacker Runner Script
----------------------
Runs Unipacker on a list of packed files (from JSON), saves unpacked files, and tracks progress.

Dependencies:
- Python 3.6+
- Unipacker must be installed and available in PATH

Usage:
    python unipackerRunner.py packed_files.json -o unpacked_dir -j 4
"""
import os
import json
import subprocess
from pathlib import Path
import multiprocessing

def run_unipacker(input_file, output_dir):
    """
    Runs unipacker on the input_file, saves output to output_dir.
    Returns (success, message, unpacked_file_path or None)
    """
    temp_dir = output_dir / f"unipacker_temp_{Path(input_file).stem}"
    temp_dir.mkdir(exist_ok=True)
    try:
        result = subprocess.run(
            ['unipacker', str(input_file), '-d', str(temp_dir)],
            capture_output=True, text=True, timeout=45
        )
        if result.returncode == 0:
            # Look for unpacked files
            unpacked_files = list(temp_dir.glob('**/*.exe')) + list(temp_dir.glob('**/*.dll'))
            if unpacked_files:
                unpacked_file = output_dir / f"unpacked_{Path(input_file).name}"
                unpacked_files[0].replace(unpacked_file)
                # Clean up temp dir
                for f in temp_dir.glob('**/*'):
                    f.unlink()
                temp_dir.rmdir()
                return True, "Successfully unpacked with unipacker", str(unpacked_file)
            else:
                for f in temp_dir.glob('**/*'):
                    f.unlink()
                temp_dir.rmdir()
                return False, "No unpacked files found", None
        else:
            for f in temp_dir.glob('**/*'):
                f.unlink()
            temp_dir.rmdir()
            return False, f"Unipacker failed: {result.stderr}", None
    except subprocess.TimeoutExpired:
        if temp_dir.exists():
            for f in temp_dir.glob('**/*'):
                f.unlink()
            temp_dir.rmdir()
        return False, "TimeoutExpired: Unipacker took too long", None
    except Exception as e:
        if temp_dir.exists():
            for f in temp_dir.glob('**/*'):
                f.unlink()
            temp_dir.rmdir()
        return False, f"Unipacker error: {e}", None

def run_unipacker_worker(args):
    file_path, output_dir, worker_num = args
    try:
        success, message, unpacked_file = run_unipacker(file_path, output_dir)
    except Exception as e:
        success, message, unpacked_file = False, f"Worker error: {e}", None
    result = {
        "original_file": file_path,
        "unpacking_success": success,
        "unpacked_file": unpacked_file,
        "message": message
    }
    return result

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run Unipacker on packed files from JSON list')
    parser.add_argument('json', help='Path to packed_files.json')
    parser.add_argument('-o', '--output', help='Output directory for unpacked files', default='unpacked')
    parser.add_argument('-j', '--jobs', type=int, default=None, help='Number of parallel workers (default: number of logical CPUs)')
    args = parser.parse_args()

    packed_json = Path(args.json)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    progress_json = output_dir / "unipacker_progress.json"

    with open(packed_json, 'r') as f:
        data = json.load(f)
        files = data.get('files', [])

    # Load progress if exists
    processed_files = set()
    if progress_json.exists():
        try:
            with open(progress_json, 'r') as pf:
                progress = json.load(pf)
                processed_files = set(progress.get('processed_files', []))
            print(f"Resuming from checkpoint: {progress_json}")
            print(f"  Already processed: {len(processed_files)} files")
        except Exception as e:
            print(f"Warning: Could not load progress file: {e}")

    # Print skipped files
    for f in files:
        if f in processed_files:
            print(f"Skipping already processed file: {f}")

    # Filter files to process
    files_to_process = [f for f in files if f not in processed_files]
    if not files_to_process:
        print("All files have already been processed. Exiting.")
        return
    print(f"Files left to process: {len(files_to_process)}")

    results = []
    num_workers = args.jobs if args.jobs else multiprocessing.cpu_count()
    print(f"Using {num_workers} parallel workers.")
    args_list = [(file_path, output_dir, (i % num_workers) + 1) for i, file_path in enumerate(files_to_process)]

    def save_progress():
        with open(progress_json, 'w') as pf:
            json.dump({"processed_files": list(processed_files)}, pf, indent=2)

    with multiprocessing.Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered(run_unipacker_worker, args_list):
            results.append(result)
            processed_files.add(result['original_file'])
            save_progress()

    # Print summary
    total = len(results)
    successes = sum(1 for r in results if r['unpacking_success'])
    timeouts = sum(1 for r in results if 'TimeoutExpired' in (r['message'] or ''))
    errors = total - successes - timeouts
    print(f"\nSummary: {total} files processed")
    print(f"  Successes: {successes}")
    print(f"  Timeouts: {timeouts}")
    print(f"  Other errors/failures: {errors}")
    if total < len(files_to_process):
        print(f"Warning: Only {total} out of {len(files_to_process)} files were processed!")

    # Save results JSON
    results_json = output_dir / "unipacker_results.json"
    with open(results_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_json}")
    print(f"Progress saved to: {progress_json}")

    # Save timeouts JSON for rerun
    timeouts_list = [r['original_file'] for r in results if 'TimeoutExpired' in (r['message'] or '')]
    if timeouts_list:
        timeouts_json = output_dir / "unipacker_timeouts.json"
        with open(timeouts_json, 'w') as tf:
            json.dump({"files": timeouts_list}, tf, indent=2)
        print(f"Timeouts saved to: {timeouts_json}")

if __name__ == "__main__":
    main()