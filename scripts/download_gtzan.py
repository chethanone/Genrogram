import os
import sys
import tarfile
import urllib.request
from pathlib import Path

DATASET_URL = "https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/genres.tar.gz"
TARGET_DIR = Path("data/gtzan")
TAR_PATH = TARGET_DIR / "genres.tar.gz"

def download_progress_hook(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    downloaded_mb = (count * block_size) / (1024 * 1024)
    total_mb = total_size / (1024 * 1024)
    sys.stdout.write(f"\rDownloading GTZAN: {percent}% [{downloaded_mb:.1f}/{total_mb:.1f} MB]")
    sys.stdout.flush()

def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if already extracted
    extracted_genres = list(TARGET_DIR.glob("genres*/*"))
    if len(extracted_genres) >= 10:
        print("GTZAN dataset already downloaded and extracted.")
        return

    if not TAR_PATH.exists():
        print(f"Downloading GTZAN from {DATASET_URL}...")
        req = urllib.request.Request(DATASET_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(TAR_PATH, 'wb') as out_file:
            total_size = int(response.headers.get('Content-Length', 0))
            block_size = 1024 * 1024 # 1MB blocks
            downloaded = 0
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)
                percent = int(downloaded * 100 / total_size) if total_size else 0
                sys.stdout.write(f"\rDownloading GTZAN: {percent}% [{downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB]")
                sys.stdout.flush()
        print("\nDownload complete.")

    print("Extracting archive...")
    with tarfile.open(TAR_PATH, "r:gz") as tar:
        tar.extractall(path=TARGET_DIR)
    print("Extraction complete.")

if __name__ == "__main__":
    main()
