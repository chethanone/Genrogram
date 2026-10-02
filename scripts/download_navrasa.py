from pathlib import Path
from urllib.request import Request, urlopen
import shutil
import time
import zipfile

BASE_URL = (
    "https://huggingface.co/datasets/"
    "beastLucifer/navrasa-5000-dataset/resolve/main/"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = PROJECT_ROOT / "data" / "expanded_raw" / "archives"
EXTRACT_DIR = PROJECT_ROOT / "data" / "expanded_raw"

ARCHIVES = {
    "Amapiano": "Global/Amapiano.zip",
    "Hyperpop": "Global/Hyperpop.zip",
    "K-Pop": "Global/K-Pop.zip",
    "Phonk": "Global/Phonk.zip",
    "Techno": "Global/Techno.zip",
    "Bollywood": "India/Bollywood.zip",
    "Desi_Hip-Hop": "India/Desi_Hip-Hop.zip",
    "Haryanvi": "India/Haryanvi.zip",
    "I-Pop": "India/I-Pop.zip",
    "Punjabi_Pop": "India/Punjabi_Pop.zip",
}

MAX_RETRIES = 4


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):
        temp_file = destination.with_suffix(".zip.part")

        try:
            print()
            print("=" * 72)
            print(
                f"Downloading: {destination.name} "
                f"(attempt {attempt}/{MAX_RETRIES})"
            )
            print("=" * 72)

            request = Request(
                url,
                headers={
                    "User-Agent": "GENROGRAM-Dataset-Downloader/1.1",
                },
            )

            with urlopen(request, timeout=120) as response:
                total_header = response.headers.get("Content-Length")
                total = int(total_header) if total_header else None

                downloaded = 0

                with open(temp_file, "wb") as output:
                    while True:
                        chunk = response.read(4 * 1024 * 1024)

                        if not chunk:
                            break

                        output.write(chunk)
                        downloaded += len(chunk)

                        if total:
                            percent = downloaded * 100 / total

                            print(
                                f"\rProgress: {percent:6.2f}% "
                                f"({downloaded / 1024 / 1024:.1f} MB / "
                                f"{total / 1024 / 1024:.1f} MB)",
                                end="",
                                flush=True,
                            )
                        else:
                            print(
                                f"\rDownloaded: "
                                f"{downloaded / 1024 / 1024:.1f} MB",
                                end="",
                                flush=True,
                            )

            print()

            actual_size = temp_file.stat().st_size

            if total is not None and actual_size != total:
                raise RuntimeError(
                    f"Incomplete download: "
                    f"{actual_size / 1024 / 1024:.2f} MB received, "
                    f"{total / 1024 / 1024:.2f} MB expected."
                )

            if actual_size == 0:
                raise RuntimeError("Downloaded file is empty.")

            # Validate the ZIP BEFORE accepting it.
            print("Validating ZIP archive...")

            if not zipfile.is_zipfile(temp_file):
                raise RuntimeError(
                    "Downloaded file is not a valid ZIP archive."
                )

            shutil.move(str(temp_file), str(destination))

            print(
                f"Saved successfully: {destination.name} "
                f"({actual_size / 1024 / 1024:.2f} MB)"
            )

            return

        except Exception as exc:
            temp_file.unlink(missing_ok=True)

            print()
            print(f"Download attempt failed: {exc}")

            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Failed to download {destination.name} "
                    f"after {MAX_RETRIES} attempts."
                ) from exc

            print("Retrying in 5 seconds...")
            time.sleep(5)


def archive_contains_audio(archive: Path) -> bool:
    with zipfile.ZipFile(archive, "r") as zf:
        for name in zf.namelist():
            lower = name.lower()

            if lower.endswith(
                (".wav", ".mp3", ".flac", ".ogg", ".m4a")
            ):
                return True

    return False


def extract_archive(genre: str, archive: Path) -> None:
    target = EXTRACT_DIR / genre
    target.mkdir(parents=True, exist_ok=True)

    print()
    print(f"Extracting {genre}...")

    with zipfile.ZipFile(archive, "r") as zf:
        members = zf.infolist()

        print(f"Archive members: {len(members)}")

        for member in members:
            member_path = Path(member.filename)

            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(
                    f"Unsafe archive path detected: {member.filename}"
                )

        zf.extractall(target)

    print(f"Extracted to: {target}")


def has_audio_files(directory: Path) -> bool:
    if not directory.exists():
        return False

    extensions = (
        "*.wav",
        "*.mp3",
        "*.flac",
        "*.ogg",
        "*.m4a",
    )

    for extension in extensions:
        if next(directory.rglob(extension), None) is not None:
            return True

    return False


def main() -> None:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 72)
    print("GENROGRAM - NAVRASA EXPANSION DATASET")
    print("=" * 72)
    print("Direct ZIP download mode")
    print("No datasets package")
    print("No torchcodec")
    print("=" * 72)

    for genre, relative_url in ARCHIVES.items():

        archive_path = ARCHIVE_DIR / f"{genre}.zip"
        target = EXTRACT_DIR / genre

        # Existing archive: validate it before using it.
        if archive_path.exists() and archive_path.stat().st_size > 0:

            size_mb = archive_path.stat().st_size / 1024 / 1024

            print()
            print(
                f"[FOUND] {genre}.zip "
                f"({size_mb:.2f} MB)"
            )

            if not zipfile.is_zipfile(archive_path):
                print(
                    f"[INVALID] {genre}.zip is incomplete/corrupt."
                )

                archive_path.unlink()

            else:
                print("[VALID] Existing ZIP archive.")

        # Download if no valid archive exists.
        if not archive_path.exists():

            url = BASE_URL + relative_url

            download_file(
                url,
                archive_path,
            )

        # Validate again.
        if not zipfile.is_zipfile(archive_path):
            raise RuntimeError(
                f"Archive validation failed: {archive_path}"
            )

        # Don't extract twice.
        if has_audio_files(target):
            print(
                f"[SKIP] {genre}: audio already extracted."
            )
        else:
            extract_archive(
                genre,
                archive_path,
            )

    print()
    print("=" * 72)
    print("NAVRASA DOWNLOAD + EXTRACTION COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()