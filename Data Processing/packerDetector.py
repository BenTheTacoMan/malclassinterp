"""
Simple Malware Packing Detector using DiE
----------------------------------------
Detects packed/unpacked status using DiE and outputs two JSON lists.

Dependencies:
- Python 3.6+
- Detect-It-Easy (DiE) console version (diec.exe) must be available

Usage:
    python packerDetector.py input_dir_or_json --die-path path/to/diec.exe -o output_dir -j 4
"""
import os
import sys
import json
import subprocess
from pathlib import Path
import multiprocessing

def is_json_file(path):
    """Return True if the path is a JSON file."""
    return str(path).lower().endswith('.json')

class DiEAnalyzer:
    def __init__(self, die_path=None):
        """Initialize DiEAnalyzer with the path to diec.exe."""
        if die_path is None:
            die_path = Path("tools/die/diec.exe")  # Default relative path
        self.die_path = Path(die_path)
        self.die_available = self._test_die()
        if not self.die_available:
            print("Warning: DiE not found or not working at {}".format(self.die_path))

    def _test_die(self):
        """Test if DiE is available."""
        try:
            result = subprocess.run([str(self.die_path), '--version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0 or "Detect It Easy" in result.stdout
        except Exception:
            return False

    def is_packed(self, file_path):
        """Return True if DiE detects the file as packed, else False."""
        if not self.die_available:
            return False
        try:
            result = subprocess.run([
                str(self.die_path), str(file_path), '--json'
            ], capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                try:
                    die_output = json.loads(result.stdout)
                    detects = die_output.get('detects', [])
                    return len(detects) > 0
                except Exception:
                    return 'pack' in result.stdout.lower() or 'crypt' in result.stdout.lower()
            else:
                return False
        except Exception:
            return False

def analyze_file(args):
    file_path, die_path, tool_num = args
    analyzer = DiEAnalyzer(die_path=die_path)
    packed = analyzer.is_packed(file_path)
    return (str(file_path), packed, tool_num)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Simple DiE Packing Detector')
    parser.add_argument('input', help='Input file, directory, or JSON list')
    parser.add_argument('--die-path', help='Path to diec executable', default='tools/die/diec.exe')
    parser.add_argument('-o', '--output', help='Output directory for results', default='INSPECT')
    parser.add_argument('-j', '--jobs', type=int, default=None, help='Number of parallel workers (default: all CPU cores)')
    args = parser.parse_args()

    analyzer = DiEAnalyzer(die_path=args.die_path)
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_json = output_dir / "progress.json"
    packed_json = output_dir / "packed_files.json"
    not_packed_json = output_dir / "not_packed_files.json"

    # Load progress if exists
    packed_files = []
    not_packed_files = []
    processed_files = set()
    if progress_json.exists():
        try:
            with open(progress_json, 'r') as f:
                progress = json.load(f)
                packed_files = progress.get('packed_files', [])
                not_packed_files = progress.get('not_packed_files', [])
                processed_files = set(progress.get('processed_files', []))
            print(f"Resuming from checkpoint: {progress_json}")
            print(f"  Already processed: {len(processed_files)} files")
        except Exception as e:
            print(f"Warning: Could not load progress file: {e}")

    print(f"Processing files from: {input_path}")

    def save_progress():
        with open(progress_json, 'w') as f:
            json.dump({
                "packed_files": packed_files,
                "not_packed_files": not_packed_files,
                "processed_files": list(processed_files)
            }, f, indent=2)

    # Gather files to process
    files_to_process = []
    if is_json_file(input_path):
        with open(input_path, 'r') as f:
            data = json.load(f)
            files = data.get('files', [])
            for file_path in files:
                if str(file_path) not in processed_files:
                    files_to_process.append(file_path)
    elif input_path.is_file():
        if str(input_path) not in processed_files:
            files_to_process.append(str(input_path))
    elif input_path.is_dir():
        for file_path in input_path.glob("**/*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.exe', '.dll', '.bin']:
                if str(file_path) not in processed_files:
                    files_to_process.append(str(file_path))

    print(f"Total files to process: {len(files_to_process)}")

    num_workers = args.jobs if args.jobs else multiprocessing.cpu_count()
    print(f"Using {num_workers} parallel workers.")
    with multiprocessing.Pool(processes=num_workers) as pool:
        args_list = [(fp, str(args.die_path), (i % num_workers) + 1) for i, fp in enumerate(files_to_process)]
        for file_path, packed, tool_num in pool.imap_unordered(analyze_file, args_list):
            if packed:
                packed_files.append(file_path)
            else:
                not_packed_files.append(file_path)
            processed_files.add(file_path)
            save_progress()

    print("Summary:")
    print(f"  Files processed: {len(packed_files) + len(not_packed_files)}")
    print(f"  Packed files found: {len(packed_files)}")
    print(f"  Not packed files found: {len(not_packed_files)}")

    with open(packed_json, 'w') as f:
        json.dump({"packed": True, "files": packed_files}, f, indent=2)
    with open(not_packed_json, 'w') as f:
        json.dump({"packed": False, "files": not_packed_files}, f, indent=2)

    print(f"  Packed files list saved to: {packed_json}")
    print(f"  Not packed files list saved to: {not_packed_json}")
    print(f"  Progress file saved to: {progress_json}")

if __name__ == "__main__":
    main() 