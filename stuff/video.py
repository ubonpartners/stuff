from collections import OrderedDict
import stuff.misc as misc
import tempfile
import os
import shutil
import cv2
from PIL import Image
import io
import math
import re
import os
import hashlib
import subprocess
#from __future__ import annotations
from pathlib import Path
from typing import Union, Optional
import stuff.platform_stuff as stuff_platform

class RandomAccessVideoReader:
    """
    Lets you access frames in the file by time in any order.
    Works with a small LRU cache for video frames.
    Will reopen and re-read the entire
    video from the start if you request an older frame that isn't in the cache.
    """
    def __init__(self, video_path, max_size=10, sw_decode=False):
        if stuff_platform.is_jetson():
            sw_decode=True
        self.video_path = video_path
        self.max_size = max_size
        self.cache = OrderedDict()  # frame_index -> (frame, time)
        self.next_frame_index = 0   # next frame we haven't read yet
        self.is_pcap = False
        self.use_pyav = sw_decode

        # PCAP handling remains unchanged
        if video_path.endswith('.pcap'):
            self.is_pcap = True
            try:
                import ubon_pycstuff.ubon_pycstuff as upyc
                import cv2
            except ImportError:
                raise ImportError("pcap decode required ubon_pycstuff")
            self.upyc = upyc
            self.cv2 = cv2
            self.pcap_decoder = self.upyc.c_pcap_decoder(video_path)
        else:
            if self.use_pyav:
                try:
                    import av
                except ImportError:
                    raise ImportError("Please install PyAV: pip install av")
                self.av = av
                self._open_pyav()
            else:
                try:
                    import cv2
                except ImportError:
                    raise ImportError("Please install OpenCV: pip install opencv-python")
                self.cv2 = cv2
                self.cap = self.cv2.VideoCapture(video_path)
                self.fps = self.cap.get(self.cv2.CAP_PROP_FPS)

    def _open_pyav(self):
        """Open container and prepare PyAV decoder"""
        self.container = self.av.open(self.video_path, options={"hwaccel": "none"})
        self.video_stream = next(s for s in self.container.streams if s.type == 'video')
        # compute fps from stream average_rate
        rate = self.video_stream.average_rate
        self.fps = float(rate) if rate else None
        self.frame_iter = self.container.decode(video=self.video_stream.index)
        self.first_pts = None

    def _read_reset(self):
        """Reset decoder and cache"""
        self.next_frame_index = 0
        self.cache.clear()
        if self.is_pcap:
            self.pcap_decoder = self.upyc.c_pcap_decoder(self.video_path)
        elif self.use_pyav:
            # close old and reopen
            try:
                self.container.close()
            except Exception:
                pass
            self._open_pyav()
        else:
            self.cap.release()
            self.cap = self.cv2.VideoCapture(self.video_path)

    def _read_forward_until(self, frame_index):
        """
        Read frames up to 'frame_index' (inclusive), storing in cache.
        """
        n_read = 0
        while self.next_frame_index <= frame_index:
            if self.is_pcap:
                # existing pcap logic
                img = self.pcap_decoder.get_frame()
                success = img is not None
                if success:
                    if self.next_frame_index == 0:
                        self.first_frame_time = img.time
                    frame_time = (img.time - self.first_frame_time) / 90000.0
                    frame = img.to_numpy()
                    frame = self.cv2.cvtColor(frame, self.cv2.COLOR_RGB2BGR)
            elif self.use_pyav:
                try:
                    frame = next(self.frame_iter)
                    # record first PTS for time zero
                    if self.next_frame_index == 0:
                        self.first_pts = frame.pts * frame.time_base
                    frame_time = frame.pts * frame.time_base - self.first_pts
                    # convert to BGR NumPy array
                    img = frame.to_ndarray(format='bgr24')
                    success = True
                    frame = img
                except StopIteration:
                    success = False
            else:
                success, frame = self.cap.read()
                frame_time = self.next_frame_index / self.fps

            if not success:
                return n_read

            # cache and evict LRU
            self.cache[self.next_frame_index] = (frame, frame_time)
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
            self.next_frame_index += 1
            n_read += 1
        return n_read

    def get_frame_at_time(self, frame_time):
        # unchanged
        assert frame_time >= 0, "negative frame time"
        if frame_time == 0:
            return self.get_frame_at_index(0)
        while True:
            if self.cache:
                times = [t for (_, t) in self.cache.values()]
                if frame_time >= min(times) and frame_time <= max(times):
                    index = min(self.cache, key=lambda i: abs(self.cache[i][1] - frame_time))
                    return self.get_frame_at_index(index)
            if not self.cache or frame_time < min(times):
                self._read_reset()
            if self._read_forward_until(self.next_frame_index + 1) == 0:
                print(f"get_frame_at_time: time {frame_time} beyond end of file?")
                break
        return self.get_frame_at_index(self.next_frame_index)

    def get_frame_at_index(self, frame_index):
        # unchanged
        if frame_index in self.cache:
            frame, frame_time = self.cache.pop(frame_index)
            self.cache[frame_index] = (frame, frame_time)
            return frame, frame_time
        if frame_index < self.next_frame_index:
            self._read_reset()
        self._read_forward_until(frame_index)
        if frame_index in self.cache:
            frame, frame_time = self.cache.pop(frame_index)
            self.cache[frame_index] = (frame, frame_time)
            return frame, frame_time
        return None, None

def mp4_to_h264(src, dest, debug=False):
    fd, tmp_path = tempfile.mkstemp(suffix=".h264")
    os.close(fd)  # Close the open file descriptor so ffmpeg can overwrite it
    misc.rm(tmp_path)

    misc.run_cmd(f"ffmpeg -i {src} -c:v copy -bsf:v h264_mp4toannexb -an -f h264 {tmp_path}", debug=debug)
    shutil.move(tmp_path, dest)

def mp4_to_h26x(input_path, max_width, max_height, target_fps, codec='h264', output_folder=None):
    """
    Convert an MP4 file to an H.264 or H.265 elementary stream (.264/.265).

    Args:
        input_path (str): Path to the input MP4 file.
        max_width (int): Maximum allowed width for output.
        max_height (int): Maximum allowed height for output.
        target_fps (float): Maximum allowed framerate for output.
        codec (str): 'h264' or 'h265'. Defaults to 'h264'.
        output_folder (str, optional): Directory to place the output file. Defaults to None (same dir as input).

    Returns:
        str: Path to the generated elementary stream file.
    """
    # Probe the input file for width, height, and frame rate
    probe_cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        input_path
    ]
    output = subprocess.check_output(probe_cmd).decode().splitlines()
    orig_w = int(output[0])
    orig_h = int(output[1])
    # r_frame_rate is like '30000/1001' or '30/1'
    num, denom = output[2].split('/')
    orig_fps = float(num) / float(denom)

    # Compute scaling factor to fit within max dimensions while preserving aspect ratio
    scale = min(max_width / orig_w, max_height / orig_h)
    new_w = (int(orig_w * scale) // 4) * 4
    new_h = (int(orig_h * scale) // 4) * 4

    # Determine frame decimation factor K
    K = 1
    while orig_fps / K > target_fps:
        K += 1
    new_fps = orig_fps / K

    # Build output filename
    base, _ = os.path.splitext(os.path.basename(input_path))
    ext = '264' if codec == 'h264' else 'hevc'
    fopt = 'h264' if codec == 'h264' else 'hevc'
    filename = f"{base}_{new_w}x{new_h}_{new_fps:.2f}fps.{ext}"

    # Determine output folder and full path
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, filename)
    else:
        output_path = os.path.join(os.path.dirname(input_path), filename)

    if os.path.exists(output_path):
        return output_path

    # Set gop size (max 2 seconds)
    gop_size = int(new_fps * 2)

    # Build ffmpeg command
    vf_filters = [
        f"select='not(mod(n\\,{K}))'",
        f"scale={new_w}:{new_h}:force_original_aspect_ratio=decrease"
    ]
    ff_cmd = [
        'ffmpeg', '-i', input_path,
        '-vf', ','.join(vf_filters),
        '-vsync', 'vfr',
    #    '-r', str(new_fps),
        '-c:v', 'libx264' if codec == 'h264' else 'libx265',
        '-preset', 'medium',
        ('-x264-params' if codec == 'h264' else '-x265-params'),
        'annexb=1',
        '-bf', '0',  # no B-frames
        '-g', str(gop_size),
        '-an',
        '-f', fopt,
        output_path
    ]

    # Execute ffmpeg
    subprocess.run(ff_cmd, check=True)
    return output_path

def get_video_framerate(input_path):
    """
    Determine the framerate of a video file.

    Attempts to probe via ffprobe; if that fails, falls back to parsing the filename for a pattern like WxH_kfps.ext.

    Args:
        input_path (str): Path to the video file.

    Returns:
        float or None: Detected framerate in fps, or None if undetermined.
    """
    # First, try ffprobe
    try:
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            input_path
        ]
        output = subprocess.check_output(probe_cmd, stderr=subprocess.DEVNULL).decode().strip()
        if output:
            num, denom = output.split('/')
            fps = float(num) / float(denom)
            return fps
    except Exception:
        pass

    # Fallback: parse filename
    filename = os.path.basename(input_path)
    # look for pattern like _<number>fps
    m = re.search(r'_([0-9]+(?:\.[0-9]+)?)fps', filename)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None

def video_to_jpegs(mp4_path, width, height, fps):
    """
    Decodes an MP4 video and returns a list of JPEG byte arrays.
    Frames are scaled to fit within (width x height), preserving aspect ratio.
    Frames are sampled to ensure output framerate is ≤ fps.

    Args:
        mp4_path (str): Path to the MP4 file.
        width (int): Maximum width of output JPEGs.
        height (int): Maximum height of output JPEGs.
        fps (float): Maximum output framerate.

    Returns:
        List[bytes]: List of JPEG-encoded frames as byte arrays.
    """
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {mp4_path}")

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if input_fps <= 0:
        input_fps = 30.0  # fallback
    frame_interval = max(1, int(math.floor(input_fps / fps)))

    jpeg_bytes_list = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            # Resize while maintaining aspect ratio
            img.thumbnail((width, height), Image.LANCZOS)

            # Encode as JPEG in memory
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=85)
            b=buf.getvalue()
            jpeg_bytes_list.append(b)

        frame_idx += 1

    cap.release()
    return jpeg_bytes_list, input_fps/frame_interval

def download_youtube_to_mp4(friendlyname, url: str, cache_dir="/mldata/video/youtube") -> str:
    """
    Download a YouTube video as an MP4 up to 720p, using a cache directory.

    Args:
        url (str): The URL of the YouTube video.
        cache_dir (str): Path to a directory where downloads are cached.

    Returns:
        str: The path to the downloaded (or cached) MP4 file.
    """

    from yt_dlp import YoutubeDL

    if url.endswith(".mp4"):
        return url

    friendlyname=friendlyname.replace(" ", "_")

    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Create a deterministic filename: sha256 of URL, take first 12 chars
    url_hash = hashlib.sha256(url.encode('utf-8')).hexdigest()[:12]
    filename = f"{friendlyname}_{url_hash}.mp4"
    filepath = os.path.join(cache_dir, filename)

    # If already downloaded, skip download
    if os.path.isfile(filepath):
        return filepath

    # yt-dlp options: best mp4 <=720p + best audio, merge to mp4
    ydl_opts = {
        'format': 'bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]',
        'merge_output_format': 'mp4',
        'outtmpl': filepath,
        'quiet': True,               # suppress console output
        'no_warnings': True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=True)

    return filepath

class FFmpegNotFoundError(RuntimeError):
    """Raised when ffmpeg (or ffprobe) is not found on PATH."""

def mp4_to_wav(
    input_mp4: Union[str, Path],
    sample_rate: int,
    dest_dir: Union[str, Path],
    *,
    channels: int = 1,
    overwrite: bool = False,
    force_reencode: bool = False,
    min_filesize_bytes: int = 1024,
    verify: bool = False,
    ffmpeg_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Convert an MP4 file to a WAV file at a given sample rate, caching results.

    If an output file named ``<stem>_<sample_rate>.wav`` already exists inside
    ``dest_dir`` and appears valid (size >= ``min_filesize_bytes``), that path
    is returned immediately (cache hit). Otherwise the audio is (re)extracted
    with ffmpeg, resampled, and the resulting path returned.

    Parameters
    ----------
    input_mp4 : str | Path
        Path to the source .mp4 file.
    sample_rate : int
        Desired output sample rate in Hz (e.g. 16000, 44100).
    dest_dir : str | Path
        Directory to store the cached wav output.
    channels : int, default 1
        Number of audio channels for output (1 = mono, 2 = stereo, etc).
    overwrite : bool, default False
        If True and output exists, remove and recreate it (ignores cache).
    force_reencode : bool, default False
        If True, re-run ffmpeg even if the existing file seems valid (same as
        overwrite=True except it keeps the existing file until the new one
        succeeds).
    min_filesize_bytes : int, default 1024
        Minimum size to consider an existing wav valid (prevents returning
        truncated/empty files).
    verify : bool, default False
        If True, runs a quick ffprobe pass to ensure the file has expected
        sample rate & channel count; if mismatch, re-encodes.
    ffmpeg_path : str | Path | None
        Explicit path to ffmpeg executable. If None, uses 'ffmpeg' on PATH.
        (ffprobe is assumed to be alongside ffmpeg if verification is used.)

    Returns
    -------
    Path
        Path to the (existing or newly created) wav file.

    Raises
    ------
    FileNotFoundError
        If the input mp4 does not exist.
    FFmpegNotFoundError
        If ffmpeg (or ffprobe when verify=True) is not found.
    RuntimeError
        If ffmpeg fails to produce the expected output.

    Notes
    -----
    - Resampling & channel remixing uses ffmpeg filters: `aresample` and `pan`
      (when downmixing to mono).
    - For pure extraction without re-encode you *could* try `-acodec copy`,
      but since we want a specific PCM wav rate/channels, we decode to PCM S16LE.
    """
    input_path = Path(input_mp4)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input MP4 not found: {input_path}")

    if ffmpeg_path is None:
        ffmpeg_exec = shutil.which("ffmpeg")
    else:
        ffmpeg_exec = str(ffmpeg_path)

    if not ffmpeg_exec:
        raise FFmpegNotFoundError(
            "ffmpeg executable not found on PATH. Install ffmpeg or provide ffmpeg_path."
        )

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    out_name = f"{input_path.stem}_{sample_rate}.wav"
    out_path = dest / out_name

    def _valid_existing(p: Path) -> bool:
        if not p.is_file():
            return False
        if p.stat().st_size < min_filesize_bytes:
            return False
        if verify:
            # Use ffprobe to confirm properties
            ffprobe_exec = shutil.which("ffprobe") if ffmpeg_path is None else (
                str(Path(ffmpeg_exec).with_name("ffprobe"))
            )
            if not ffprobe_exec:
                raise FFmpegNotFoundError(
                    "ffprobe not found (needed because verify=True). Install ffprobe or disable verify."
                )
            probe_cmd = [
                ffprobe_exec,
                "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=sample_rate,channels",
                "-of", "default=noprint_wrappers=1:nokey=0",
                str(p),
            ]
            try:
                res = subprocess.run(
                    probe_cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                return False
            sr_ok, ch_ok = False, False
            for line in res.stdout.splitlines():
                if line.startswith("sample_rate="):
                    sr_ok = int(line.split("=")[1]) == sample_rate
                elif line.startswith("channels="):
                    ch_ok = int(line.split("=")[1]) == channels
            return sr_ok and ch_ok
        return True
    if out_path.exists():
        if overwrite:
            out_path.unlink()
        elif not force_reencode and _valid_existing(out_path):
            return out_path  # Cache hit
        # else we'll re-encode (force_reencode) but keep file until success.

    # Prepare ffmpeg command
    # -vn : ignore video
    # -ar : set audio sample rate
    # -ac : set audio channels
    # -f wav : ensure output container
    # -y only if overwrite True
    cmd = [
        ffmpeg_exec,
        "-i", str(input_path),
        "-vn",
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-sample_fmt", "s16",  # 16-bit PCM
        "-f", "wav",
    ]
    if overwrite:
        cmd.append("-y")
    else:
        cmd.extend(["-n"])  # fail if exists (we deleted earlier if overwrite)
    cmd.append(str(out_path))

    # If force_reencode and file exists, write to temp then atomically replace
    if force_reencode and out_path.exists() and not overwrite:
        tmp_path = out_path.with_suffix(".tmp.wav")
        if tmp_path.exists():
            tmp_path.unlink()
        cmd[-1] = str(tmp_path)
        target_path = out_path
    else:
        tmp_path = None
        target_path = out_path

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Clean up partial file if present
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(
            f"ffmpeg failed (exit {result.returncode}).\n"
            f"Command: {' '.join(cmd)}\n"
            f"Stdout:\n{result.stdout}\nStderr:\n{result.stderr}"
        )

    if tmp_path:
        # Replace existing only after successful encode
        if target_path.exists():
            target_path.unlink()
        tmp_path.rename(target_path)

    if not _valid_existing(target_path):
        # Remove invalid artifact
        try:
            target_path.unlink()
        except FileNotFoundError:
            pass
        raise RuntimeError("Output wav failed validation.")

    return target_path