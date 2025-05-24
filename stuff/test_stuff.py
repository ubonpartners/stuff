import os
from pathlib import Path
from shutil import copy2
import stuff.image as image
from PIL import ImageFile
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True  # In case of partial files

def get_all_jpeg_paths(root_dir):
    return [p for p in Path(root_dir).rglob("*") if p.suffix.lower() in ['.jpg', '.jpeg']]

def process_folder_safely_same_dir(directory, max_width, max_height, max_bpp=2.0):
    directory = Path(directory).resolve()
    input_images = list(get_all_jpeg_paths(directory))  # Pre-scan to avoid rescanning output

    total_images = len(input_images)
    transcoded_count = 0
    total_size_before = 0
    total_size_after = 0

    for img_path in tqdm(input_images):
        changed, size_before, size_after = image.image_copy_resize(
            str(img_path), max_width, max_height, max_bpp
        )
        if changed:
            transcoded_count += 1
        total_size_before += size_before
        total_size_after += size_after

    percent_transcoded = (transcoded_count / total_images * 100) if total_images > 0 else 0.0

    print(f"\nProcessed {total_images} images in {directory}")
    print(f"Transcoded/resized: {transcoded_count} ({percent_transcoded:.1f}%)")
    print(f"Total size before: {total_size_before / 1024:.1f} KB")
    print(f"Total size after:  {total_size_after / 1024:.1f} KB")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Safe JPEG processor in-place")
    parser.add_argument("directory", help="Directory containing JPEGs to process")
    parser.add_argument("--max_width", type=int, default=1920)
    parser.add_argument("--max_height", type=int, default=1920)
    parser.add_argument("--max_bpp", type=float, default=3.0)
    args = parser.parse_args()

    process_folder_safely_same_dir(args.directory, args.max_width, args.max_height, args.max_bpp)
