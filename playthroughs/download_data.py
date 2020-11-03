import os
import zipfile
import argparse
import urllib.request
from os.path import join as pjoin

import tqdm


ZIP_FILENAME = "TextWorld_CoG2019.zip"
GAMES_URL = "https://aka.ms/ftwp/dataset.zip"


def download(url, filename=None, force=False):
    filename = filename or url.split('/')[-1]

    if os.path.isfile(filename) and not force:
        return filename

    def _report_download_status(chunk_id, max_chunk_size, total_size):
        size = chunk_id * max_chunk_size / 1024**2
        size_total = total_size / 1024**2
        unit = "Mb"
        if size <= size_total:
            print("{:.1f}{unit} / {:.1f}{unit}".format(size, size_total, unit=unit), end="\r")

    filename, _ = urllib.request.urlretrieve(url, filename, _report_download_status)
    return filename


def extract_games(zip_filename, dst, force=False):
    zipped_file = zipfile.ZipFile(zip_filename)
    filenames_to_extract = [f for f in zipped_file.namelist() if f.endswith(".z8") or f.endswith(".json")]

    subdirs = {
        "train": pjoin(dst, "train"),
        "valid": pjoin(dst, "valid"),
        "test": pjoin(dst, "test"),
    }
    for d in subdirs.values():
        if not os.path.isdir(d):
            os.makedirs(d)

    print("Extracting...")
    extracted_files = []
    for filename in tqdm.tqdm(filenames_to_extract):
        subdir = subdirs[os.path.basename(os.path.dirname(filename))]
        out_file = pjoin(subdir, os.path.basename(filename))
        extracted_files.append(out_file)
        if os.path.isfile(out_file) and not force:
            continue

        data = zipped_file.read(filename)
        with open(out_file, "wb") as f:
            f.write(data)

    return extracted_files


def build_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--games-dir", default="./games/",
                        help="Folder where to extract the downloaded games.")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Overwrite existing files.")
    parser.add_argument("-ff", "--force-download", action="store_true",
                        help="Download data again.")

    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    if not os.path.exists(args.games_dir) or args.force or args.force_download:
        filename = download(GAMES_URL, filename=ZIP_FILENAME, force=args.force_download)
        _ = extract_games(filename, dst=args.games_dir, force=args.force)
    else:
        parser.error("Destination folder already exists: {}. (Use -f to overwrite)".format(args.games_dir))


if __name__ == "__main__":
    main()
