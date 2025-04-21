import piexif
import piexif.helper
import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2

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
