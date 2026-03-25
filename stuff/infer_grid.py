
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
import albumentations as A
from albumentations.augmentations.geometric.transforms import Affine
import stuff.coord as coord
import stuff.detections as detections
import stuff.display as disp
from io import BytesIO
import numpy as np
import cv2

def resize_to_box(img: Image.Image, max_width: int, max_height: int, resample=Image.Resampling.LANCZOS) -> Image.Image:
    """
    Resize image to fit within (max_width, max_height) while preserving aspect ratio.
    Unlike thumbnail(), this can upscale smaller images as well.
    Returns a new PIL.Image.
    """
    orig_width, orig_height = img.size

    # Calculate scaling factor
    scale = min(max_width / orig_width, max_height / orig_height)

    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)

    return img.resize((new_width, new_height), resample)


def create_image_grid(image_paths, grid_rows, grid_cols, width, height,
                      padding_ratio=0.05, bg_color=(0, 0, 0), max_random_shrink=0.20,
                      aug_rotate=0, aug_effects=0, aug_basic=True, allow_upscale=True,
                      texts=None, font_path=None, font_size=16, text_color=(255,255,255)):
    """
    Create a grid of images with optional per-cell text overlays at the bottom of each cell.

    Args:
        image_paths (list): List of image paths, raw bytes, or numpy arrays.
        grid_rows (int): Number of rows in grid.
        grid_cols (int): Number of columns in grid.
        width (int): Total width of output image.
        height (int): Total height of output image.
        padding_ratio (float): Fractional padding within each cell.
        bg_color (tuple): Background color for canvas.
        max_random_shrink (float): Max shrink fraction for augmentations.
        aug_rotate (float): Probability of rotation augmentation.
        aug_effects (float): Probability of final effects augmentation.
        aug_basic (bool): Whether to apply basic random flip/shrink.
        texts (list of str): Optional list of text per cell. None or shorter lists skip text.
        font_path (str): Path to a .ttf font file. If None, uses default font.
        font_size (int): Font size for overlay texts.
        text_color (tuple): Text color RGB.

    Returns:
        out_img_aug (PIL.Image): Final augmented image.
        norm_boxes (list): Normalized bounding boxes for each image cell.
    """
    num_cells = grid_rows * grid_cols
    cell_width = width // grid_cols
    cell_height = height // grid_rows
    pad_x = int(cell_width * padding_ratio)
    pad_y = int(cell_height * padding_ratio)
    out_img = Image.new('RGB', (width, height), color=bg_color)
    norm_boxes = []

    # Prepare rotation transformer if needed
    if aug_rotate != 0:
        subimage_transform = A.Compose([
            A.Affine(rotate=(-8, 8), shear=(-5, 5), fit_output=True, p=1.0)
        ])

    # Load font
    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    for idx, item in enumerate(image_paths):
        # Skip extra cells
        if idx >= num_cells:
            norm_boxes.append(None)
            continue

        # Load image
        try:
            if isinstance(item, (bytes, bytearray)):
                img = Image.open(BytesIO(item)).convert("RGB")
            elif isinstance(item, np.ndarray):
                rgb = cv2.cvtColor(item, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
            else:
                img = Image.open(item).convert("RGB")
        except Exception as e:
            print(f"Error loading image {type(item)} at index {idx}: {e}")
            norm_boxes.append(None)
            continue

        # Basic augmentations
        if aug_basic and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        shrink_w = random.uniform(1.0 - max_random_shrink, 1.0) if aug_basic else 1.0
        shrink_h = random.uniform(1.0 - max_random_shrink, 1.0) if aug_basic else 1.0

        max_img_width = int((cell_width - 2 * pad_x) * shrink_w)
        max_img_height = int((cell_height - 2 * pad_y) * shrink_h)

        if allow_upscale:
            img=resize_to_box(img, max_img_width, max_img_height, Image.Resampling.LANCZOS)
        else:
            img.thumbnail((max_img_width, max_img_height), Image.Resampling.LANCZOS)

        # Rotation augmentation
        if aug_rotate and random.random() < aug_rotate:
            transformed = subimage_transform(image=np.array(img))
            img = Image.fromarray(transformed['image'])

        # Compute cell position
        row = idx // grid_cols
        col = idx % grid_cols
        x0 = col * cell_width + pad_x + (cell_width - 2 * pad_x - img.width) // 2
        y0 = row * cell_height + pad_y + (cell_height - 2 * pad_y - img.height) // 2
        x1 = x0 + img.width
        y1 = y0 + img.height
        out_img.paste(img, (x0, y0))

        # Draw text if provided
        if texts and idx < len(texts) and texts[idx]:
            draw = ImageDraw.Draw(out_img)
            text = texts[idx]
            # Use textbbox for accurate size
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            tx = col * cell_width + (cell_width - text_w) // 2
            ty = (row + 1) * cell_height - pad_y - text_h
            draw.text((tx, ty), text, font=font, fill=text_color)

        # Full-canvas effects augmentation
        if aug_effects and random.random() < aug_effects:
            effects_transform = A.Compose([
                A.Blur(p=0.05), A.MedianBlur(p=0.05), A.MotionBlur(p=0.05),
                A.ToGray(p=0.02), A.CLAHE(p=0.03), A.RandomBrightnessContrast(p=0.05),
                A.RandomGamma(p=0.05), A.ISONoise(p=0.1), A.ChromaticAberration(p=0.02),
                A.RandomRain(p=0.02), A.RandomFog(p=0.02), A.RandomSunFlare(p=0.01),
                A.ImageCompression(quality_range=(30, 100), p=0.1)
            ])
            out_img = Image.fromarray(effects_transform(image=np.array(out_img))['image'])

        # Record normalized box
        norm_boxes.append([x0 / width, y0 / height, x1 / width, y1 / height])

    return out_img, norm_boxes

def infer_grid(inf_wrapper, images, grid_rows=4, grid_cols=4, width=1280, height=1280):
    batch_size=inf_wrapper.infer_batch_size
    num_per_image=grid_rows*grid_cols
    grid_images=[]
    grids=[]
    for i in range(0, len(images), num_per_image):
        batch_images = images[i:i + num_per_image]
        img,grid_boxes=create_image_grid(batch_images, grid_rows, grid_cols, width, height, max_random_shrink=0)
        #disp.faf_display([img])
        grid_images.append(img)
        grids.append(grid_boxes)
    out_dets=[[] for i in range(len(images))]
    for idx in range(0,len(grid_images), batch_size):
        res=inf_wrapper.infer(grid_images[idx:idx+batch_size])
        for j,r in enumerate(res):
            imgn=idx+j
            grid_boxes=grids[imgn]
            #disp.faf_display([grid_images[imgn], r],title=f"Image {imgn}")
            base_det_index=imgn*grid_rows*grid_cols
            for det in r:
                best_grid_box=None
                best_score=0
                for gn,g in enumerate(grid_boxes):
                    score=coord.box_ioma(g, det["box"])
                    if score>best_score:
                        best_score=score
                        best_grid_box=gn
                if best_grid_box is not None:
                    detections.unmap_detection_inplace(det, grid_boxes[best_grid_box])
                    out_dets[base_det_index+best_grid_box].append(det)
                else:
                    print("missed box!")
        #error
        #FIX ME missing detections; looks like attributes not merged and dets get clipped?
    return out_dets
