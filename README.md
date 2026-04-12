# Stuff

Shared Python utilities for ML pipelines, drawing, I/O, inference, and tooling. Boxes and coordinates use normalized `[0, 1]` unless noted.

## 📜 License

This project is dual licensed under:

1. **GNU Affero General Public License v3.0 (AGPL-3.0)** - See [LICENSE](LICENSE) file
2. **Ubon Cooperative License** - See [Ubon Cooperative License](https://github.com/ubonpartners/license/blob/main/LICENSE)

You may choose to use this code under either license. The AGPL-3.0 license applies to all users, while the Ubon Cooperative License provides additional permissions for Ubon Companies as defined in that license.

---

## 📦 Package Overview

| Module | Intent |
|--------|--------|
| **[ARGBdraw](#stuffargbdraw)** | Cairo ARGB drawing (boxes, lines, circles, text, images) with NumPy/OpenCV interop |
| **[augment](#stuffaugment)** | BT.709 YUV420 round-trip augmentation for BGR image batches |
| **[azure](#stuffazure)** | Fetch files from Azure blob storage (e.g. under `/mldata`) via azcopy/blobfuse config |
| **[coord](#stuffcoord)** | Axis-aligned boxes, points, ROI mapping, IoU, interpolation in normalized coords |
| **[datatable](#stuffdatatable)** | Pretty-print tabular results with section grouping and value colorization |
| **[detections](#stuffdetections)** | Unmap detection boxes/keypoints from a normalized ROI to full-image coords |
| **[display](#stuffdisplay)** | OpenCV window + overlay (ARGB), mouse events, optional video writer |
| **[draw](#stuffdraw)** | Draw primitives on OpenCV BGR images (lines, boxes, circles, text) in normalized coords |
| **[embedding](#stuffembedding)** | CLIP/MobileCLIP text encode; JPEG embedding pipeline; cosine similarity / softmax |
| **[facerec](#stufffacerec)** | SFace face embedding from a single JPEG (OpenCV FaceDetectorYN + FaceRecognizerSF) |
| **[gdrive](#stuffgdrive)** | Google Drive sync for `/mldata`: tar-based folder sync, versioned single-file sync |
| **[image](#stuffimage)** | Image I/O, EXIF user comment, SSIM, resize/transcode, JPEG size/header helpers |
| **[img_dedup](#stuffimg_dedup)** | Perceptual hash (pHash) + FAISS binary index for duplicate/near-duplicate detection |
| **[infer_grid](#stuffinfer_grid)** | Build image grids for batched inference, then unmap detections back to per-image |
| **[inference_wrapper](#stuffinference_wrapper)** | Unified inference over YOLO/Ultralytics, RF-DETR, Detectron2, MMDet, TRT/upyc, ensembles |
| **[llm](#stuffllm)** | OpenAI GPT and Google Gemini client with caching, cost stats, image inputs |
| **[llm_guff](#stuffllm_guff)** | Legacy/experimental Anthropic and Together LLM code (commented); to be ported into llm |
| **[match](#stuffmatch)** | Match detections to ground-truth: greedy or Hungarian (LSA); optional partition by grid |
| **[misc](#stuffmisc)** | Paths, logging, pickle, time strings, run_cmd, dict params, name_from_file |
| **[mp_workqueue](#stuffmp_workqueue)** | Multiprocessing work queue with progress bars and stdout capture |
| **[pcap_stuff](#stuffpcap_stuff)** | Annex B NAL parsing, RTP packet build, pcap streamer; SDP generation for H264/H265 |
| **[platform_stuff](#stuffplatform_stuff)** | Jetson detection, GPU count, CUDA cache clear, TSS key stats (pthread) |
| **[reid](#stuffreid)** | Cosine similarity for PyTorch 1D tensors (e.g. ReID embeddings) |
| **[result_cache](#stuffresult_cache)** | Pickle-backed list of results with add/get/delete by key match |
| **[ubtrk2](#stuffubtrk2)** | UBTRK2 BMFF-like 4CC tracker-run container reader/writer + payload codecs |
| **[test_stuff](#stufftest_stuff)** | Ad-hoc tests / scratch (not a formal test suite) |
| **[ultralytics](#stuffultralytics)** | YOLO result → normalized dets, draw boxes/pose, keypoint/face helpers, class remap |
| **[ultralytics_ap](#stuffultralytics_ap)** | Average precision (AP) from confidence/TP/class arrays (adapted from Ultralytics) |
| **[vectordb](#stuffvectordb)** | ChromaDB-backed simple vector DB with time/stream filtering |
| **[video](#stuffvideo)** | Random-access video reader (LRU cache), mp4→h264/jpegs, YouTube download, framerate |

---

## 📚 Per-Module Details

<a id="stuffargbdraw"></a>
### stuff.ARGBdraw

Cairo-based ARGB drawing with normalized `[0,1]` coordinates and NumPy/OpenCV interop.

- **ARGBdraw(width, height)**
  Cairo ImageSurface (ARGB32) with normalized `[0,1]` coords. Methods: `clear`, `box`, `line`, `circle`, `img` (paste resized BGR image), `text` (with optional background), `get_numpy_view`, `get_scaled_numpy_view`.

- **blend_argb_over_rgb(argb, rgb)**
  Composite premultiplied ARGB over RGB; returns uint8 RGB.

---

<a id="stuffaugment"></a>
### stuff.augment

BT.709 YUV420 round-trip augmentation for image batches.

- **bt709_yuv420_augment(batch_bgr, randomize, coeff_range, ...)**
  Simulate BT.709 YUV420 round-trip for (N,H,W,3) BGR; optional coefficient jitter and chroma subsample error.

- **bt709_yuv420_augment_single(img)**
  Single-image wrapper (adds batch dim, then returns first image).

---

<a id="stuffazure"></a>
### stuff.azure

Azure blob storage file fetching.

- **fetch_file_from_azure(file_path, blobfuse_yml_path)**
  Ensure file under `/mldata` exists locally; if not, copy from Azure using azcopy and blobfuse config.

---

<a id="stuffcoord"></a>
### stuff.coord

Axis-aligned boxes, points, ROI mapping, IoU, and interpolation in normalized coordinates.

- **clip01(x)**
  Clamp scalar to [0, 1].

- **box_w, box_h, box_a**
  Width, height, area of box `[x1,y1,x2,y2]`.

- **box_i(b1, b2), box_union(b1, b2), box_iou(b1, b2), box_ioma(b1, b2), box_expand(b, f)**
  Intersection area, union box, IoU, IoU over min area, expand box by factor (clip to [0,1]).

- **unmap_roi_point(roi, pt)** / **unmap_roi_box(roi, box)**
  Map normalized point/box from ROI to full coords (non-destructive).

- **unmap_roi_*_inplace(roi, pt\|box\|keypoints)**
  Same in-place.

- **point_in_box(pt, box)**
  Returns distance-like value if inside, else None.

- **interpolate(x, y, f)** / **interpolate2(x, y, f)**
  Linear blend; interpolate mutates list, interpolate2 returns new value(s).

---

<a id="stuffdatatable"></a>
### stuff.datatable

Pretty-print tabular results with section grouping and value colorization.

- **show_data(results_in, columns, column_text, sort_fn, section_key, add_section_dividers)**
  Build pandas DataFrame, round values, colorize by min/max per column, group by section_key, print via tabulate.

---

<a id="stuffdetections"></a>
### stuff.detections

Unmap detection boxes and keypoints from a normalized ROI to full-image coordinates.

- **unmap_detection_inplace(det, roi)**
  Unmap `det["box"]`, optional `subbox`, `pose_points`, `face_points` from ROI to full image.

---

<a id="stuffdisplay"></a>
### stuff.display

OpenCV display window with ARGB overlay, mouse events, and optional video writer.

- **Display(width, height, image, name, output, no_mouse)**
  OpenCV window + ARGB overlay; optional VideoWriter; mouse callback pushes events (normalized coords, selected boxes).

- **faf_display(list_of_stuff, title, width, height)**
  Show one image plus overlay boxes from dicts/lists; wait for key.

- **display_image_wait_key(image, scale, title)**
  Show single image, return first key event.

---

<a id="stuffdraw"></a>
### stuff.draw

Draw primitives on OpenCV BGR images using normalized coordinates.

- **set_colour(clr, chan, default, default_alpha)**
  Resolve colour name or tuple to BGR(A); supports `_half`, `_flashing`, etc.

- **draw_line(img, start, stop, clr, thickness)**
  Draw line on BGR image (normalized coords).

- **draw_box(img, box, clr, thickness)**
  Draw rectangle.

- **draw_circle(img, centre, radius, clr, thickness)**
  Filled circle.

- **draw_text(img, text, xc, yc, img_bg, font, fontScale, fontColor, bgColor, ...)**
  Multi-line text with background rect.

---

<a id="stuffembedding"></a>
### stuff.embedding

CLIP/MobileCLIP text encoding, JPEG embedding pipeline, and cosine similarity utilities.

- **clip_encode_text(text_strings, model_choice, device)**
  Encode with MobileCLIP v1 or v2 (open_clip); returns L2-normalized list of lists.

- **get_jpeg_embeddings(upyc_cli, jpegs, track_shared)**
  Run tracking pipeline on JPEGs; return face/CLIP embeddings and crops per frame (largest person).

- **cosine_similarity(vec1, vec2)**
  Cosine similarity for Python lists (not PyTorch).

- **cosine_similarities_to_probabilities(input, refs, scale)**
  Softmax over cosine similarities to refs.

---

<a id="stufffacerec"></a>
### stuff.facerec

SFace face recognition from JPEG images.

- **get_sface_embedding(jpeg_bytes)**
  Decode JPEG, detect face (YuNet), align/crop (SFace), return embedding as list.

---


<a id="stuffgdrive"></a>
### stuff.gdrive

Google Drive synchronization for `/mldata` paths.

- **gdrive_sync_mldata(local_path, direction, destructive, checksum, drive_id, ...)**
  Sync path under `/mldata`: directory → tar build/upload or download/extract; file → versioned upload/download. Uses OAuth and optional checksums.

- **RateTracker**, **DriveSync**, **compute_tree_hash**, **compute_file_hash**, **build_tar_with_progress**, and internal helpers support the above.

---

<a id="stuffimage"></a>
### stuff.image

Image I/O, EXIF comments, SSIM, resize/transcode, and JPEG size helpers.

- **image_append_exif_comment(image_file, comment)**
  Append to EXIF UserComment (separated by `;`).

- **image_get_exif_comment(image_file)**
  Read EXIF UserComment.

- **image_ssim(img1, img2)**
  Mean structural similarity over channels (skimage).

- **get_image_size(filepath)**
  Fast JPEG size or PIL fallback.

- **image_copy_resize(filepath, max_width, max_height, max_bpp, yuv420, fix_orientation)**
  Resize/transcode to temp then replace; optional EXIF orientation fix.

- **determine_scale_size(w, h, max_w, max_h, ...)**
  Scale size to fit with optional rounding and stretch limits.

---

<a id="stuffimg_dedup"></a>
### stuff.img_dedup

Perceptual hash-based image deduplication with FAISS.

- **phash_to_bytes(img)**
  64-bit perceptual hash as packed uint8[8] for FAISS binary index.

- **ImgDedup(threshold)**
  Streaming dedup: `test(path)` returns MatchResult and adds to index; `save(folder)` / `load(folder)` for persistence.

---

<a id="stuffinfer_grid"></a>
### stuff.infer_grid

Grid-based batched inference with detection unmapping.

- **resize_to_box(img, max_width, max_height, resample)**
  Resize PIL image to fit in box (can upscale).

- **create_image_grid(image_paths, grid_rows, grid_cols, width, height, ...)**
  Build single PIL image grid with optional augmentation and per-cell text; returns image and list of normalized boxes per cell.

- **infer_grid(inf_wrapper, images, grid_rows, grid_cols, width, height)**
  Run inference on grids, then unmap detections back to per-image lists.

---

<a id="stuffinference_wrapper"></a>
### stuff.inference_wrapper

Unified inference wrapper supporting multiple backends and ensemble methods.

- **infer_model_name(x)**
  Human-readable model name from path/spec (e.g. +flip, :640, .engine).

- **inference_wrapper(model_name, class_names, thr, nms_thr, ...)**
  Unified wrapper: Ultralytics YOLO (including world/pose/attributes), RF-DETR, Detectron2, MMDetection, .trt (ubon_pycstuff), upyc_track; supports ensembles and flip/scale params in name.

- **download_mmdet_config(config_path, repo)**
  Download MMDetection config from URL if missing.

- **merge_detections_wbf(detectors_outputs, iou_thr, skip_box_thr)**
  Weighted boxes fusion (ensemble_boxes).

- **ensemble_merge_results(results)**
  Per-image WBF over list of result lists.

- **infer**, **infer_cached**
  Dispatch to backend; cached version uses get_image_fn and batch window.

---

<a id="stuffllm"></a>
### stuff.llm

OpenAI GPT and Google Gemini client with caching and cost tracking.

- **simple_llm(model, cost_opt)**
  OpenAI (gpt-*) or Google Gemini client; tracks tokens and cost; optional system prompt and Gemini cache.

- **infer(prompt, images, attempts)**
  Chat completion with optional image list (base64 or paths).

- **infer_cached(j, attempts, should_hit)**
  File-based cache keyed by prompt/images/system prompt hash.

- **infer_cached_batch(jobs, attempts, num_parallel)**
  Prepare jobs, run missing in parallel, then return all from cache.

- **get_base64_jpeg(path, quality, optimize)**
  Load, thumbnail to model size, return base64 JPEG.

- **get_stats()**
  Cost and token stats.

---

<a id="stuffllm_guff"></a>
### stuff.llm_guff

Legacy/experimental LLM code (commented out).

- Commented-out legacy code: Anthropic and Together LLM clients for attribute generation; to be ported into llm when needed.

---

<a id="stuffmatch"></a>
### stuff.match

Matching algorithms for detections to ground-truth assignments.

- **match_lsa(dets, gts, mfn, mfn_context)**
  Hungarian (linear sum assignment) matching; cost matrix from mfn(det, gt, context).

- **match_lsa2(dets, gts, mfn, ..., partition_fn, match_method)**
  Hungarian or greedy (optionally with partition_fn for sparse edges); returns det_indices, gt_indices, costs.

- **match_greedy(dets, gts, mfn, mfn_context)**
  Greedy max-weight matching.

- **uniform_grid_partition(box, context)**
  Map box to bitmask of grid cells (default 4×16) for partition_fn.

---

<a id="stuffmisc"></a>
### stuff.misc

General file system, logging, and utility functions.

- **makedir, rename, rm, rmdir**
  Path operations.

- **load_dictionary(file_name)**
  Load JSON or YAML.

- **save_atomic_pickle(obj, path)**
  Write pickle to temp then replace.

- **timestr**, **get_dict_param**, **run_cmd**
  Time string, dict param with default, subprocess run.

- **configure_root_logger**, **format_seconds_ago**, **name_from_file**
  Logging and naming helpers.

---

<a id="stuffmp_workqueue"></a>
### stuff.mp_workqueue

Multiprocessing work queue with progress bars and stdout capture.

- **mp_workqueue_run(work_to_run, worker_fn, num_workers, desc, ...)**
  Run work items in processes (or in-process if num_workers=0); progress and stdout forwarded; optional process_setup_fn.

- **mpwq_progress(mpwq_context, desc, total, update)**
  Worker-side progress update.

- **test_work()**
  Example run with test_work_fn.

---

<a id="stuffpcap_stuff"></a>
### stuff.pcap_stuff

Annex B NAL parsing, RTP packet building, and pcap streaming utilities.

- **annexb_to_pcap(file_bytes, output_path, ...)**
  Parse Annex B NALs, wrap in RTP, write pcap.

- **parse_pcap(pcap_path)**
  Read pcap and yield packet/parsed info.

- **pcap_packet_streamer(...)**
  Stream packets with optional filtering.

- **generate_sdp(codec, port, fps, vps, sps, pps)**
  SDP string for H264/H265.

- **parse_annexb(file_bytes)**
  Split Annex B into NAL byte slices.

- **build_rtp_packet(...)**
  Build single RTP packet with Scapy.

---

<a id="stuffplatform_stuff"></a>
### stuff.platform_stuff

Platform detection and GPU/system utilities.

- **is_jetson()**
  True if machine is aarch64.

- **platform_num_gpus()**
  GPU count (1 on Jetson, else torch.cuda.device_count()).

- **platform_clear_caches()**
  gc.collect() and torch.cuda.empty_cache().

- **platform_tss_key_stats()**
  Report pthread TSS key usage (ceiling and allocated).

---

<a id="stuffreid"></a>
### stuff.reid

Re-identification utilities using cosine similarity.

- **cosine_similarity(vec1, vec2)**
  Cosine similarity for PyTorch 1D tensors (e.g. ReID); normalizes and returns scalar.

---

<a id="stuffresult_cache"></a>
### stuff.result_cache

Pickle-backed result caching with key-based matching.

- **ResultCache(resultfile)**
  Pickle-backed list; **add(result)**, **get(match)** (by key equality), **delete(match)**, **save()**.

---

<a id="stuffubtrk2"></a>
### stuff.ubtrk2

UBTRK2 binary tracker-run container and payload codecs.

- **Container**: BMFF-like length-prefixed 4CC boxes (`ubtf`, `meta`, repeated `fram`)
- **Metadata**: `meta` payload is UTF-8 YAML bytes
- **Frame schema**: each `fram` contains typed child boxes (`fhdr`, `trks`, `dets`, `dbug`, `imgp`, `xtra`)
- **`UBTRK2Writer(path, metadata)`** / **`UBTRK2Reader(path)`**
  Streaming writer + indexed reader for efficient persistence and replay
- **`encode_nested_arrays` / `decode_nested_payloads` / `decode_payload`**
  Payload wrappers for ndarray/bytes fields without sidecar files
- **`is_ubtrk2_file(path)`**
  Fast format probe for UBTRK2 file detection

---

<a id="stufftest_stuff"></a>
### stuff.test_stuff

Ad-hoc tests and scratch code (not a formal test suite).

---

<a id="stuffultralytics"></a>
### stuff.ultralytics

YOLO/Ultralytics result parsing, drawing, and keypoint utilities.

- YOLO result parsing: **yolo_results_to_dets**, **draw_boxes**, **fold_detections_to_attributes**.
- Keypoints/pose: **map_one_gt_keypoints**, **map_keypoints**, **has_pose_points**, **has_face_points**, **check_pose_points**, **get_face_triangle_points**, **draw_pose**.
- Box/class: **find_gt_from_point**, **make_class_remap_table**, **attributes_from_class_names**, **better_annotation**.
- Helpers: **is_large**, **box_expand** (re-export), etc.

---

<a id="stuffultralytics_ap"></a>
### stuff.ultralytics_ap

Average precision calculation utilities (adapted from Ultralytics).

- **compute_ap(recall, precision)**
  Average precision and envelope curves (from Ultralytics metrics).

- **smooth(y, f)**
  Box filter smoothing.

- **ap_calc(conf, tp, pred_cls, target_cls, nc, min_gt, pr_curves)**
  Per-class AP from confidence and TP/class arrays.

---

<a id="stuffvectordb"></a>
### stuff.vectordb

Simple vector database using ChromaDB.

- **SimpleVectorDB(collection_name)**
  ChromaDB collection (cosine); **add(uid, embedding, start_time, end_time, stream_id, payload)**, **query(embedding, top_k, start_time, end_time, stream_ids)**.

---

<a id="stuffvideo"></a>
### stuff.video

Random-access video reading and video processing utilities.

- **RandomAccessVideoReader(video_path, max_size, sw_decode)**
  LRU-cached random access by frame index; supports mp4 and .pcap (ubon_pycstuff).

- **mp4_to_h264**, **video_to_jpegs**, **mp4_to_h26x**, **get_video_framerate**
  Transcode and frame extraction.

- **download_youtube_to_mp4**, **mp4_to_wav**
  YouTube download and audio extract.
