"""Image I/O, EXIF comments, SSIM, resize/transcode, and size helpers (JPEG-friendly)."""
import os
import struct
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np
import piexif
import piexif.helper
from PIL import Image, ImageOps, UnidentifiedImageError


def image_append_exif_comment(image_file, comment):
    """
    Append comment to the end of any existing exif 'usercomment'
    if already exists separate with ';'
    """
    user_comment=""
    try:
        exif_dict = piexif.load(image_file)
    except piexif._exceptions.InvalidImageDataError:
        print(f"image_append_exif_comment: Invalid image {image_file}")
        return False
    try:
        user_comment = piexif.helper.UserComment.load(exif_dict["Exif"][piexif.ExifIFD.UserComment])
    except KeyError:
        pass
    except ValueError:
        pass
    user_comment=user_comment+";"+comment
    exif_dict["Exif"][piexif.ExifIFD.UserComment] = piexif.helper.UserComment.dump(
        user_comment,
        encoding="unicode"
    )
    try:
        piexif.insert(
            piexif.dump(exif_dict),
            image_file
        )
    except Exception as e:
        print(f"image_append_exif_comment: piexif.insert failed file {image_file} exception {e}")
        return False
    return True

def image_get_exif_comment(image_file):
    """
    Get 'usercomment' string exif field from an image.
    Deal with exceptions
    """
    user_comment=""
    try:
        exif_dict = piexif.load(image_file)
    except piexif._exceptions.InvalidImageDataError:
        print(f"image_get_exif_comment: Invalid image {image_file}")
        return ""

    try:
        user_comment = piexif.helper.UserComment.load(exif_dict["Exif"][piexif.ExifIFD.UserComment])
    except KeyError:
        pass
    return user_comment

def image_ssim(img1, img2):
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        raise ImportError("Please install scikit-image to use image_ssim function: pip install scikit-image")
    # Resize if necessary
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Split into color channels
    channels1 = cv2.split(img1)
    channels2 = cv2.split(img2)

    # Compute SSIM for each channel
    ssim_scores = [
        ssim(c1, c2, data_range=255)
        for c1, c2 in zip(channels1, channels2)
    ]

    # Return average SSIM
    return np.mean(ssim_scores)

def get_image_size(filepath):
    """Attempts fast JPEG header parse, falls back to PIL."""

    # in case a loaded image was passed instead of a file path
    if isinstance(filepath, np.ndarray) and filepath.ndim in (2, 3):
        h, w, _ = filepath.shape
        return w, h

    try:
        with open(filepath, 'rb') as f:
            if f.read(2) != b'\xff\xd8':
                raise ValueError("Not a JPEG")
            while True:
                byte = f.read(1)
                while byte and byte != b'\xff':
                    byte = f.read(1)
                while byte == b'\xff':
                    byte = f.read(1)
                if byte in [b'\xc0', b'\xc1', b'\xc2', b'\xc3', b'\xc5', b'\xc6', b'\xc7',
                            b'\xc9', b'\xca', b'\xcb', b'\xcd', b'\xce', b'\xcf']:
                    f.read(3)
                    h, w = struct.unpack(">HH", f.read(4))
                    return w, h
                else:
                    length = struct.unpack(">H", f.read(2))[0]
                    f.read(length - 2)
    except Exception:
        try:
            with Image.open(filepath) as img:
                return img.width, img.height
        except UnidentifiedImageError:
            raise ValueError("Unsupported or corrupt image format")

def get_file_size(path):
    return os.path.getsize(path) if os.path.exists(path) else 0


def image_copy_resize(filepath, max_width, max_height, max_bpp=2.5, yuv420=False, fix_orientation=False):
    """Resize/transcode safely by writing to a temp file, then replacing the original."""

    try:
        width, height = get_image_size(filepath)
    except Exception as e:
        print(f"Warning: {e}, skipping.")
        return False, get_file_size(filepath), get_file_size(filepath)

    file_size_before = os.path.getsize(filepath)
    bpp = file_size_before / (width * height)
    needs_resize = width > max_width or height > max_height
    needs_transcode = bpp > max_bpp
    if not (needs_resize or needs_transcode or yuv420 or fix_orientation):
        return False, file_size_before, file_size_before

    tmp_path = None
    try:
        with Image.open(filepath) as img:
            # Preserve and update EXIF
            exif_bytes = None
            try:
                exif_dict = piexif.load(img.info.get("exif", b""))
                # Remove orientation tag if fixing orientation
                if fix_orientation and piexif.ImageIFD.Orientation in exif_dict.get("0th", {}):
                    del exif_dict["0th"][piexif.ImageIFD.Orientation]
                user_comment = ''
                if piexif.ExifIFD.UserComment in exif_dict.get("Exif", {}):
                    try:
                        user_comment = piexif.helper.UserComment.load(
                            exif_dict["Exif"][piexif.ExifIFD.UserComment]
                        )
                    except Exception:
                        user_comment = ''
                user_comment += f";transcode={max_width},{max_height},yuv={yuv420}"
                exif_dict["Exif"][piexif.ExifIFD.UserComment] = \
                    piexif.helper.UserComment.dump(user_comment, encoding="unicode")
                exif_bytes = piexif.dump(exif_dict)
            except Exception:
                pass

            # Fix orientation if needed
            if fix_orientation:
                img = ImageOps.exif_transpose(img)
            # Resize if needed
            if needs_resize:
                img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

            # Prepare temp file
            tmp_file = NamedTemporaryFile(suffix=".jpg", delete=False, dir=Path(filepath).parent)
            tmp_path = tmp_file.name
            tmp_file.close()

            # Encode loop to meet max_bpp
            quality = 90
            min_quality = 20
            save_args = {
                "format": "JPEG",
                "optimize": False,
                "progressive": False,
            }
            if exif_bytes:
                save_args["exif"] = exif_bytes
            if yuv420:
                save_args["subsampling"] = 'medium'

            while True:
                save_args["quality"] = quality
                img.save(tmp_path, **save_args)
                size_after = os.path.getsize(tmp_path)
                bpp_after = size_after / (width * height)
                # Retry if still too large and quality can be reduced
                if bpp_after <= max_bpp or quality <= min_quality:
                    break
                quality = max(min_quality, quality - 5)

            # Replace original with temp
            os.replace(tmp_path, filepath)
            file_size_after = os.path.getsize(filepath)
            return True, file_size_before, file_size_after

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        # Cleanup temp if something went wrong
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False, file_size_before, file_size_before
    finally:
        # Ensure no stray temp file remains
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def determine_scale_size(w, h, max_w, max_h, percent_stretch_allowed=10, round_w=2, round_h=2, allow_upscale=False):
    assert allow_upscale==False, "fix me!"
    rw=w
    rh=h
    if rw>max_w:
        rh=(rh*max_w)//rw
        rw=(rw*max_w)//rw
    if rh>max_h:
        rw=(rw*max_h)//rh
        rh=(rh*max_h)//rh
    if round_w!=0:
        rw+=round_w//2
        rw&=(~(round_w-1))
        if rw>max_w:
            rw-=round_w
    if round_h!=0:
        rh+=round_h//2
        rh&=(~(round_h-1))
        if rh>max_h:
            rh-=round_h
    if percent_stretch_allowed!=0:
        thr_w=(max_w*(100-percent_stretch_allowed))//100;
        thr_h=(max_h*(100-percent_stretch_allowed))//100;
        if rw>thr_w and w>=max_w:
            rw=max_w
        if rh>thr_h and h>=max_h:
            rh=max_h
    return rw, rh
