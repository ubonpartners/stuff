from collections import OrderedDict
import cv2
import stuff.misc as misc
import tempfile
import os
import shutil

class RandomAccessVideoReader:
    """
    Lets you access frames in the file by time in any order.
    Works with a small LRU cache for video frames.
    Will reopen and re-read the entire
    video from the start if you request an older frame that isn't in the cache.
    
    The typical flow is forward-only, but if you jump back to an older frame
    that's no longer in the cache, we:
      - close the video
      - reopen from the start
      - read forward up to the requested frame
      - store those frames in the cache
    This can be slow if you do it a lot, but it guarantees you never get an error.
    """
    def __init__(self, video_path, max_size=10):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.max_size = max_size
        
        self.cache = OrderedDict()  # frame_index -> frame
        self.next_frame_index = 0   # the next frame we haven't read yet

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.num_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

    def _read_forward_until(self, frame_index):
        """
        Read from self.next_frame_index up to 'frame_index' (inclusive),
        storing frames in the cache, and update self.next_frame_index.
        If we can't read that many frames, we'll stop at the last frame we can read.
        """

        while self.next_frame_index <= frame_index:
            success, frame = self.cap.read()
            if not success:
                # Reached the end or can't read further
                return

            # Store in cache
            self.cache[self.next_frame_index] = frame
            # Evict LRU if over capacity
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)

            self.next_frame_index += 1

    def get_frame_at_time(self, frame_time):
        """
        Return the frame at time offset 'frame_time' seconds.
        """
        index=int(frame_time*self.fps)
        return self.get_frame_at_index(index)

    def get_frame_at_index(self, frame_index):
        """
        Return the frame at 'frame_index'.
        
        Cases:
        1) If 'frame_index' is in the cache, just pop+reinsert to refresh LRU usage.
        2) If 'frame_index' < self.next_frame_index but not in the cache:
            - Close, reopen from the start, clear cache,
            - Read forward up to 'frame_index'.
        3) If 'frame_index' >= self.next_frame_index:
            - Read forward from self.next_frame_index up to 'frame_index'.
        
        Return None if the requested frame cannot be read (EOF).
        """

        frame_index=max(0, min(frame_index, self.num_frames-1))

        # Case 1: Already in cache?
        if frame_index in self.cache:
            frame = self.cache.pop(frame_index)
            self.cache[frame_index] = frame
            return frame

        # Case 2: If the requested index is behind our read pointer and not cached
        if frame_index < self.next_frame_index:
            # Reopen from start
            self.cap.release()
            self.cap = cv2.VideoCapture(self.video_path)

            # Clear cache
            self.cache.clear()
            self.next_frame_index = 0

        # Case 3: Now we read forward until we reach frame_index
        self._read_forward_until(frame_index)

        # Attempt to retrieve the frame from the cache now
        if frame_index in self.cache:
            frame = self.cache.pop(frame_index)
            self.cache[frame_index] = frame
            return frame
        else:
            # We might have hit EOF
            return None

def mp4_to_h264(src, dest, debug=False):
    fd, tmp_path = tempfile.mkstemp(suffix=".h264")
    os.close(fd)  # Close the open file descriptor so ffmpeg can overwrite it
    misc.rm(tmp_path)

    misc.run_cmd(f"ffmpeg -i {src} -c:v copy -bsf:v h264_mp4toannexb -an -f h264 {tmp_path}", debug=debug)
    shutil.move(tmp_path, dest)