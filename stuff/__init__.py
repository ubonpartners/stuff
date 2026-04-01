"""Shared Python utilities for ML pipelines, drawing, I/O, inference, and tooling.

This package provides utilities for:
- Drawing and display (ARGBdraw, display, draw)
- Coordinate and box operations (coord, detections)
- Image and video processing (image, video, img_dedup)
- Inference wrappers (inference_wrapper, infer_grid)
- LLM integration (llm)
- Embeddings and face recognition (embedding, facerec, reid)
- Cloud storage (azure, gdrive)
- Multiprocessing (mp_workqueue)
- And more...
"""

# Video utilities
from .video import (
    RandomAccessVideoReader,
    download_youtube_to_mp4,
    get_video_framerate,
    mp4_to_h264,
    mp4_to_h26x,
    mp4_to_wav,
    video_to_jpegs,
)

# Coordinate and box operations
from .coord import (
    box_a,
    box_expand,
    box_h,
    box_i,
    box_ioma,
    box_iou,
    box_union,
    box_w,
    clip01,
    interpolate,
    interpolate2,
)

# Display and drawing
from .display import Display, faf_display

# Matching algorithms
from .match import match_lsa, match_lsa2, uniform_grid_partition

# File and system utilities
from .misc import (
    configure_root_logger,
    format_seconds_ago,
    get_dict_param,
    load_dictionary,
    makedir,
    name_from_file,
    rename,
    rm,
    rmdir,
    run_cmd,
    save_atomic_pickle,
    timestr,
)

# Ultralytics/YOLO utilities
from .ultralytics import (
    attributes_from_class_names,
    better_annotation,
    check_pose_points,
    draw_boxes,
    draw_pose,
    find_gt_from_point,
    fold_detections_to_attributes,
    get_face_triangle_points,
    has_face_points,
    has_pose_points,
    is_large,
    make_class_remap_table,
    map_keypoints,
    map_one_gt_keypoints,
    yolo_results_to_dets,
)

# Average precision calculation
from .ultralytics_ap import ap_calc

# Image utilities
from .image import (
    determine_scale_size,
    get_image_size,
    image_append_exif_comment,
    image_copy_resize,
    image_get_exif_comment,
    image_ssim,
)

# Multiprocessing work queue
from .mp_workqueue import mp_workqueue_run, test_work

# Re-identification
from .reid import cosine_similarity

# Image augmentation
from .augment import bt709_yuv420_augment, bt709_yuv420_augment_single

# Data table display
from .datatable import show_data

# Inference wrappers
from .inference_wrapper import InferenceWrapper, infer_model_name, inference_wrapper

# Grid-based inference
from .infer_grid import create_image_grid, infer_grid

# LLM integration
from .llm import simple_llm

# Azure storage
from .azure import fetch_file_from_azure

# Platform detection
from .platform_stuff import (
    is_jetson,
    platform_clear_caches,
    platform_num_gpus,
    platform_tss_key_stats,
)

# PCAP utilities
from .pcap_stuff import annexb_to_pcap, pcap_packet_streamer, parse_pcap

# Embeddings
from .embedding import (
    clip_encode_text,
    cosine_similarities_to_probabilities,
    get_jpeg_embeddings,
)

# Engine metadata helpers
from .engine import build_yolo_engine_metadata_from_pt, read_engine_metadata, write_engine_with_metadata

# Face recognition
from .facerec import get_sface_embedding

# Google Drive sync
from .gdrive import gdrive_delete_folders_with_prefix, gdrive_sync_mldata

# Image deduplication
from .img_dedup import ImgDedup

# Result caching
from .result_cache import ResultCache

__all__ = [
    # Video
    "RandomAccessVideoReader",
    "download_youtube_to_mp4",
    "get_video_framerate",
    "mp4_to_h264",
    "mp4_to_h26x",
    "mp4_to_wav",
    "video_to_jpegs",
    # Coordinates
    "box_a",
    "box_expand",
    "box_h",
    "box_i",
    "box_ioma",
    "box_iou",
    "box_union",
    "box_w",
    "clip01",
    "interpolate",
    "interpolate2",
    # Display
    "Display",
    "faf_display",
    # Matching
    "match_lsa",
    "match_lsa2",
    "uniform_grid_partition",
    # Misc utilities
    "configure_root_logger",
    "format_seconds_ago",
    "get_dict_param",
    "load_dictionary",
    "makedir",
    "name_from_file",
    "rename",
    "rm",
    "rmdir",
    "run_cmd",
    "save_atomic_pickle",
    "timestr",
    # Ultralytics
    "attributes_from_class_names",
    "better_annotation",
    "check_pose_points",
    "draw_boxes",
    "draw_pose",
    "find_gt_from_point",
    "fold_detections_to_attributes",
    "get_face_triangle_points",
    "has_face_points",
    "has_pose_points",
    "is_large",
    "make_class_remap_table",
    "map_keypoints",
    "map_one_gt_keypoints",
    "yolo_results_to_dets",
    # AP calculation
    "ap_calc",
    # Image
    "determine_scale_size",
    "get_image_size",
    "image_append_exif_comment",
    "image_copy_resize",
    "image_get_exif_comment",
    "image_ssim",
    # Multiprocessing
    "mp_workqueue_run",
    "test_work",
    # ReID
    "cosine_similarity",
    # Augmentation
    "bt709_yuv420_augment",
    "bt709_yuv420_augment_single",
    # Data table
    "show_data",
    # Inference
    "infer_model_name",
    "InferenceWrapper",
    "inference_wrapper",
    # Grid inference
    "create_image_grid",
    "infer_grid",
    # LLM
    "simple_llm",
    # Azure
    "fetch_file_from_azure",
    # Platform
    "is_jetson",
    "platform_clear_caches",
    "platform_num_gpus",
    "platform_tss_key_stats",
    # PCAP
    "annexb_to_pcap",
    "pcap_packet_streamer",
    "parse_pcap",
    # Embeddings
    "clip_encode_text",
    "cosine_similarities_to_probabilities",
    "get_jpeg_embeddings",
    # Engine metadata
    "build_yolo_engine_metadata_from_pt",
    "read_engine_metadata",
    "write_engine_with_metadata",
    # Face recognition
    "get_sface_embedding",
    # Google Drive
    "gdrive_delete_folders_with_prefix",
    "gdrive_sync_mldata",
    # Image deduplication
    "ImgDedup",
    # Result cache
    "ResultCache",
]