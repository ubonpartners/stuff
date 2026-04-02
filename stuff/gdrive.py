import os
import io
import re
import time
import tarfile
import mimetypes
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm
from collections import deque
from datetime import datetime

# ================================
# CONFIG: your Shared Drive ID
# ================================
DRIVE_ID = "0AEhZOUisyqrNUk9PVA"  # <-- your MLData Shared Drive
EXCLUDED_FILE_SUFFIXES = (".cache",)


# Transfer tuning knobs (override via env if needed).
def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "on")


DEFAULT_TRANSFER_CHUNK_SIZE = max(8, int(os.getenv("GDRIVE_TRANSFER_CHUNK_MB", "64"))) * 1024 * 1024
DEFAULT_ENABLE_PARALLEL_DOWNLOAD = _env_bool("GDRIVE_ENABLE_PARALLEL_DOWNLOAD", False)
DEFAULT_PARALLEL_DOWNLOAD_WORKERS = max(1, int(os.getenv("GDRIVE_DOWNLOAD_WORKERS", "1")))
DEFAULT_PARALLEL_MIN_BYTES = max(128, int(os.getenv("GDRIVE_PARALLEL_MIN_MB", "512"))) * 1024 * 1024


class RateTracker:
    """Tracks bytes over a sliding window to compute Mbps (single-threaded)."""
    def __init__(self, window_seconds: int = 30):
        self.window = window_seconds
        self.q = deque()  # (timestamp, bytes)
        self.bytes_sum = 0

    def add(self, nbytes: int):
        now = time.time()
        cutoff = now - self.window
        self.q.append((now, nbytes))
        self.bytes_sum += nbytes
        while self.q and self.q[0][0] < cutoff:
            _, b = self.q.popleft()
            self.bytes_sum -= b

    def mbps(self) -> float:
        total = self.bytes_sum
        return (total * 8.0) / (1e6 * max(self.window, 1))


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def guess_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def _is_sync_excluded(path: Path) -> bool:
    """Return True if this file should be excluded from hash/upload workflows."""
    return path.name.endswith(EXCLUDED_FILE_SUFFIXES)


# ---------- Tree/file hashing ----------
def compute_tree_hash(local_base: Path, checksum: bool = True) -> Tuple[str, List[Tuple[str, int, int, Optional[str]]]]:
    """
    Deterministic tree hash over the folder without building a tar.
    Returns (hex_hash, records) where each record=(relpath, size, mtime, md5|None).
    """
    local_base = local_base.resolve()
    records = []
    for root, _, filenames in os.walk(local_base):
        for fn in filenames:
            p = Path(root) / fn
            if not p.is_file():
                continue
            if _is_sync_excluded(p):
                continue
            rel = str(p.relative_to(local_base)).replace("\\", "/")
            st = p.stat()
            fmd5 = md5_file(p) if checksum else None
            records.append((rel, st.st_size, int(st.st_mtime), fmd5))
    records.sort(key=lambda r: r[0])
    h = hashlib.md5()
    for rel, size, mtime, fmd5 in records:
        h.update(rel.encode("utf-8")); h.update(b"\0")
        h.update(str(size).encode("ascii")); h.update(b"\0")
        h.update(str(mtime).encode("ascii")); h.update(b"\0")
        if fmd5 is not None:
            h.update(fmd5.encode("ascii"))
        h.update(b"\n")
    return h.hexdigest(), records


def compute_file_hash(path: Path, checksum: bool = True) -> str:
    """
    Hash for a single file.
    - checksum=True  -> MD5(contents)
    - checksum=False -> MD5(size||mtime)  (fast, but weaker)
    """
    st = path.stat()
    if checksum:
        return md5_file(path)
    h = hashlib.md5()
    h.update(str(st.st_size).encode("ascii")); h.update(b"\0")
    h.update(str(int(st.st_mtime)).encode("ascii"))
    return h.hexdigest()


# ---------- Tar build with byte-progress ----------
import io as _io
class _ReadProgress(_io.BufferedReader):
    """Wraps a file object and updates a tqdm progress bar on reads."""
    def __init__(self, raw, bar):
        super().__init__(raw)
        self._bar = bar
    def read(self, size=-1):
        chunk = super().read(size)
        if chunk:
            self._bar.update(len(chunk))
        return chunk

def build_tar_with_progress(local_base: Path, tar_path: Path, records, chunk_size: int = 1024 * 1024):
    total_bytes = sum(r[1] for r in records)
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, mode="w") as tf, tqdm(
        total=total_bytes, desc="Tarring", unit="B", unit_scale=True, unit_divisor=1024
    ) as bar:
        for rel, _size, _mtime, _fmd5 in records:
            src = local_base / rel
            if not src.is_file():
                continue
            ti = tf.gettarinfo(name=str(src), arcname=rel)
            with src.open("rb", buffering=0) as raw:
                wrapped = _ReadProgress(raw, bar)
                tf.addfile(ti, fileobj=wrapped)


# ================================
# Drive ops
# ================================
class DriveSync:
    def __init__(self, service, drive_id: str):
        self.drive_id = drive_id
        self.root_id = drive_id
        self.flags = dict(supportsAllDrives=True, includeItemsFromAllDrives=True)
        self.svc = service

    def ensure_folder_path(self, path_parts: Tuple[str, ...]) -> str:
        parent = self.root_id
        for part in path_parts:
            if not part:
                continue
            child = self._find_child_folder(parent, part)
            if not child:
                child = self.svc.files().create(
                    body={"name": part, "mimeType": "application/vnd.google-apps.folder", "parents": [parent]},
                    fields="id", supportsAllDrives=True
                ).execute()["id"]
            parent = child
        return parent

    def _find_child_folder(self, parent_id: str, name: str) -> Optional[str]:
        escaped = name.replace('"', '\\"')
        q = (
            f"'{parent_id}' in parents and "
            f"mimeType='application/vnd.google-apps.folder' and "
            f'name="{escaped}" and trashed=false'
        )
        resp = self.svc.files().list(
            q=q, corpora="drive", driveId=self.drive_id,
            fields="files(id,name,createdTime)", orderBy="createdTime asc", pageSize=2, **self.flags
        ).execute()
        files = resp.get("files", [])
        if len(files) > 1:
            ids = ", ".join(f.get("id", "?") for f in files)
            raise RuntimeError(
                f"Ambiguous Drive folder path: multiple folders named '{name}' under parent '{parent_id}' (ids: {ids})"
            )
        return files[0]["id"] if files else None

    def list_files_in(self, parent_id: str) -> List[dict]:
        request = self.svc.files().list(
            q=f"'{parent_id}' in parents and trashed=false and mimeType!='application/vnd.google-apps.folder'",
            corpora="drive", driveId=self.drive_id,
            fields="nextPageToken, files(id,name,mimeType,md5Checksum,size,modifiedTime)",
            pageSize=1000, **self.flags,
        )
        out = []
        while request is not None:
            resp = request.execute()
            out.extend(resp.get("files", []))
            request = self.svc.files().list_next(request, resp)
        return out

    def list_folders_in(self, parent_id: str) -> List[dict]:
        request = self.svc.files().list(
            q=f"'{parent_id}' in parents and trashed=false and mimeType='application/vnd.google-apps.folder'",
            corpora="drive",
            driveId=self.drive_id,
            fields="nextPageToken, files(id,name,createdTime)",
            orderBy="name asc",
            pageSize=1000,
            **self.flags,
        )
        out = []
        while request is not None:
            resp = _retry(lambda: request.execute())
            out.extend(resp.get("files", []))
            request = self.svc.files().list_next(request, resp)
        return out


# ================================
# Public API (OAuth only)
# ================================
def _build_drive_service(
    client_secret: Optional[str],
    token_path: Optional[str],
    headless: bool,
    scopes,
):
    from googleapiclient.discovery import build
    try:
        from google.oauth2.credentials import Credentials
        if token_path and os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, scopes)
        else:
            raise FileNotFoundError
        if not creds.valid and getattr(creds, "expired", False) and getattr(creds, "refresh_token", None):
            from google.auth.transport.requests import Request

            creds.refresh(Request())
            if token_path:
                with open(token_path, "w") as f:
                    f.write(creds.to_json())
    except Exception:
        from google_auth_oauthlib.flow import InstalledAppFlow

        flow = InstalledAppFlow.from_client_secrets_file(client_secret, scopes)
        creds = flow.run_console() if headless else flow.run_local_server(port=0)
        if token_path:
            with open(token_path, "w") as f:
                f.write(creds.to_json())
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def gdrive_sync_mldata(
    local_path: str,
    direction: str = "from_drive",            # "to_drive" or "from_drive"
    destructive: bool = False, # ignored in tar mode; keep for parity
    checksum: bool = False,     # must be consistent across up/down
    drive_id: Optional[str] = DRIVE_ID,
    client_secret: Optional[str] = "/mldata/gdrive/client_secret.json",
    token_path: Optional[str] = "/mldata/gdrive/token.json",
    keep_local_tar: bool = False,  # NEW: delete local tar by default (False)
    headless: bool = False,
    scopes=("https://www.googleapis.com/auth/drive",),
):
    """
    If local_path is a directory: tar-based sync of the folder.
    If local_path is a file: versioned single-file sync (no tar).
    """
    # --- Build Drive service via OAuth ---
    service = _build_drive_service(client_secret=client_secret, token_path=token_path, headless=headless, scopes=scopes)

    # --- Path checks and remote path resolution ---
    lp = Path(local_path).resolve()
    mldata_root = Path("/mldata").resolve()
    if not str(lp).startswith(str(mldata_root)):
        raise ValueError(f"local_path must be under /mldata; got {lp}")

    rel_from_mldata = str(lp.relative_to(mldata_root)).replace("\\", "/")
    remote_parts = tuple([p for p in rel_from_mldata.split("/") if p])

    drive = DriveSync(service, drive_id)

    # Helper: resolve the remote parent and base names
    remote_parent_id = drive.ensure_folder_path(remote_parts[:-1]) if len(remote_parts) > 0 else drive.root_id
    base_name = remote_parts[-1] if remote_parts else lp.name
    stem = lp.stem
    suffix = lp.suffix  # includes leading dot or ""

    if direction == "to_drive":
        # original behavior (requires local to exist)
        if lp.is_dir():
            _sync_folder_to_drive_as_tar(drive, lp, remote_parent_id, base_name, checksum, keep_local_tar=keep_local_tar)
        elif lp.is_file():
            _sync_file_to_drive_versioned(drive, lp, remote_parent_id, stem, suffix, checksum)
        else:
            raise ValueError(f"local_path must exist as a file or directory; got {lp}")

    elif direction == "from_drive":
        # NEW: if local doesn't exist, infer from Drive contents
        if lp.exists():
            if lp.is_dir():
                _sync_drive_tar_to_folder(drive, lp, remote_parent_id, base_name, checksum)
            elif lp.is_file():
                _sync_drive_versioned_to_file(drive, lp, remote_parent_id, stem, suffix, checksum)
            else:
                # Very rare (e.g., special file). Treat as error.
                raise ValueError(f"local_path exists but is neither file nor directory; got {lp}")
        else:
            files = drive.list_files_in(remote_parent_id)
            tar_candidate = _pick_latest_tar(files, base_name)
            versioned_candidate = _pick_latest_versioned(files, stem, suffix)

            if tar_candidate:
                # Treat as folder tar sync
                _sync_drive_tar_to_folder(drive, lp, remote_parent_id, base_name, checksum)
            elif versioned_candidate:
                # Treat as single-file sync
                _sync_drive_versioned_to_file(drive, lp, remote_parent_id, stem, suffix, checksum)
            else:
                print(f"Gdrive_sync: Nothing on Drive matching {base_name}_*.tar or {stem}_*{suffix}. Nothing to do.")
                return
    else:
        raise ValueError("direction must be 'to_drive' or 'from_drive'")


def gdrive_delete_folders_with_prefix(
    prefix: str,
    *,
    parent_folder_id: Optional[str] = None,
    drive_id: Optional[str] = DRIVE_ID,
    client_secret: Optional[str] = "/mldata/gdrive/client_secret.json",
    token_path: Optional[str] = "/mldata/gdrive/token.json",
    headless: bool = False,
    scopes=("https://www.googleapis.com/auth/drive",),
    dry_run: bool = False,
) -> Dict[str, object]:
    """
    Delete all folders whose names start with `prefix` under a given Drive parent (default: shared-drive root).
    Returns a summary dictionary.
    """
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("prefix must be a non-empty string")

    service = _build_drive_service(client_secret=client_secret, token_path=token_path, headless=headless, scopes=scopes)
    drive = DriveSync(service, drive_id)
    parent_id = parent_folder_id or drive.root_id

    folders = drive.list_folders_in(parent_id)
    matches = [f for f in folders if str(f.get("name", "")).startswith(prefix)]
    if not matches:
        print(f"Gdrive_sync: No folders found with prefix '{prefix}' under parent '{parent_id}'.")
        return {"parent_id": parent_id, "prefix": prefix, "matched": 0, "deleted": 0, "folders": []}

    deleted = []
    for f in matches:
        name = f.get("name", "")
        file_id = f.get("id", "")
        if dry_run:
            print(f"[dry-run] Would delete folder: {name} ({file_id})")
        else:
            print(f"Deleting folder: {name} ({file_id})")
            _retry(lambda fid=file_id: drive.svc.files().delete(fileId=fid, supportsAllDrives=True).execute())
        deleted.append({"id": file_id, "name": name})

    return {
        "parent_id": parent_id,
        "prefix": prefix,
        "matched": len(matches),
        "deleted": 0 if dry_run else len(matches),
        "folders": deleted,
        "dry_run": bool(dry_run),
    }


# ================================
# Folder -> Drive (tar)
# ================================
TAR_NAME_RE_TMPL = r"^{base}_(\d{{8}}-\d{{6}})_{hash}\.tar$"
TAR_ANY_RE_TMPL = r"^{base}_(\d{{8}}-\d{{6}})_[0-9a-fA-F]{{32}}\.tar$"

def _sync_folder_to_drive_as_tar(
    drive: DriveSync,
    local_base: Path,
    remote_parent_id: str,
    base_name: str,
    checksum: bool,
    keep_local_tar: bool = False,
    upload_chunk_size: int = DEFAULT_TRANSFER_CHUNK_SIZE,
):
    from googleapiclient.http import MediaFileUpload

    # 1) Compute tree hash and get records
    tree_hash, records = compute_tree_hash(local_base, checksum=checksum)

    # 2) If remote already has exact hash -> skip
    want_pat = re.compile(TAR_NAME_RE_TMPL.format(base=re.escape(base_name), hash=re.escape(tree_hash)))
    for f in drive.list_files_in(remote_parent_id):
        if want_pat.match(f["name"]):
            print(f"Up-to-date tar exists on Drive: {f['name']}. Skipping upload.")
            return

    # 3) Build tar path (UTC timestamp)
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    tar_name = f"{base_name}_{ts}_{tree_hash}.tar"
    tar_path = local_base.parent / tar_name

    # 4) Build tar with progress if not already present
    if not tar_path.exists():
        print(f"Building tar: {tar_path}")
        build_tar_with_progress(local_base, tar_path, records)

    # 5) Upload with byte-accurate progress
    total_bytes = os.path.getsize(tar_path)
    rate = RateTracker(window_seconds=30)
    media = MediaFileUpload(str(tar_path), mimetype=guess_mime(tar_path), resumable=True, chunksize=upload_chunk_size)
    request = drive.svc.files().create(
        body={"name": tar_name, "parents": [remote_parent_id]},
        media_body=media, fields="id", supportsAllDrives=True
    )
    response = None
    last = 0
    with tqdm(total=total_bytes, desc="Uploading", unit="B", unit_scale=True, unit_divisor=1024) as bar:
        while response is None:
            status, response = _retry(lambda: request.next_chunk())
            if status:
                sent = int(getattr(status, "resumable_progress", 0)) - last
                if sent > 0:
                    rate.add(sent); last += sent
                    bar.update(sent); bar.set_postfix_str(f"{rate.mbps():.2f} Mbps")
        if last < total_bytes:
            bar.update(total_bytes - last)

    print("Upload complete.")

    # 6) Delete local tar unless user wants to keep it
    if not keep_local_tar:
        try:
            tar_path.unlink()
            print(f"Deleted local tar {tar_path}")
        except Exception as e:
            print(f"Warning: could not delete tar {tar_path}: {e}")


# ================================
# Drive -> Folder (untar)
# ================================
def _parse_tar_name(base_name: str, fname: str) -> Optional[Tuple[datetime, str]]:
    m = re.match(TAR_ANY_RE_TMPL.format(base=re.escape(base_name)), fname)
    if not m:
        return None
    dt_str = m.group(1)
    hash_hex = fname.rsplit("_", 1)[-1].split(".tar")[0]
    try:
        dt = datetime.strptime(dt_str, "%Y%m%d-%H%M%S")
    except ValueError:
        return None
    return dt, hash_hex

def _pick_latest_tar(files: List[dict], base_name: str) -> Optional[dict]:
    candidates = []
    for f in files:
        parsed = _parse_tar_name(base_name, f["name"])
        if parsed:
            dt, _ = parsed
            candidates.append((dt, f))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]

def _sync_drive_tar_to_folder(
    drive: DriveSync,
    local_base: Path,
    remote_parent_id: str,
    base_name: str,
    checksum: bool,
    download_chunk_size: int = DEFAULT_TRANSFER_CHUNK_SIZE,
):
    files = drive.list_files_in(remote_parent_id)
    latest = _pick_latest_tar(files, base_name)
    if not latest:
        print(f"Gdrive_sync: No tar named like {base_name}_*.tar found on Drive. Nothing to do.")
        return

    _dt, hash_in_name = _parse_tar_name(base_name, latest["name"])

    # NEW: only compute local hash if the folder already exists
    local_hash = None
    if local_base.exists():
        local_hash, _ = compute_tree_hash(local_base, checksum=checksum)
        if local_hash == hash_in_name:
            print(f"Gdrive_sync: local folder {local_base} already matches tar hash {hash_in_name}. Skipping download.")
            return

    tar_local_path = local_base.parent / latest["name"]
    print(f"Gdrive_sync: Downloading {latest['name']} from Drive ...")
    _download_file(drive, latest["id"], tar_local_path, RateTracker(), chunk_size=download_chunk_size)

    print(f"Extracting into {local_base} ...")
    local_base.mkdir(parents=True, exist_ok=True)  # ensure it exists before extract
    with tarfile.open(tar_local_path, "r") as tf:
        members = tf.getmembers()
        for m in members:
            _validate_tar_member(m, latest["name"])
        with tqdm(total=len(members), desc="Extracting", unit="file") as bar:
            for m in members:
                tf.extract(m, path=local_base)
                bar.update(1)
    print("Extraction complete.")

# ================================
# Single-file: naming & sync
# ================================
# Example: my.bin -> my_20250101-120304_<HASH>.bin
def _format_versioned_filename(stem: str, suffix: str, dt: datetime, hash_hex: str) -> str:
    ts = dt.strftime("%Y%m%d-%H%M%S")
    return f"{stem}_{ts}_{hash_hex}{suffix}"

FILE_ANY_RE_TMPL = r"^{stem}_(\d{{8}}-\d{{6}})_[0-9a-fA-F]{{32}}{suffix}$"

def _parse_versioned_filename(stem: str, suffix: str, fname: str) -> Optional[Tuple[datetime, str]]:
    m = re.match(FILE_ANY_RE_TMPL.format(stem=re.escape(stem), suffix=re.escape(suffix)), fname)
    if not m:
        return None
    dt_str = m.group(1)
    hash_hex = fname[len(stem)+1: -len(suffix)]  # "<ts>_<hash>"
    hash_hex = hash_hex.split("_")[-1]
    try:
        dt = datetime.strptime(dt_str, "%Y%m%d-%H%M%S")
    except ValueError:
        return None
    return dt, hash_hex

def _pick_latest_versioned(files: List[dict], stem: str, suffix: str) -> Optional[dict]:
    candidates = []
    for f in files:
        parsed = _parse_versioned_filename(stem, suffix, f["name"])
        if parsed:
            dt, _ = parsed
            candidates.append((dt, f))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]

def _sync_file_to_drive_versioned(
    drive: DriveSync,
    local_file: Path,
    remote_parent_id: str,
    stem: str,
    suffix: str,
    checksum: bool,
    upload_chunk_size: int = DEFAULT_TRANSFER_CHUNK_SIZE,
):
    from googleapiclient.http import MediaFileUpload

    if _is_sync_excluded(local_file):
        print(f"Gdrive_sync: Skipping excluded file {local_file}")
        return

    file_hash = compute_file_hash(local_file, checksum=checksum)
    want_name = _format_versioned_filename(stem, suffix, datetime.utcnow(), file_hash)

    # If any remote file already has suffix _<hash>, skip (regardless of timestamp)
    pat = re.compile(FILE_ANY_RE_TMPL.format(stem=re.escape(stem), suffix=re.escape(suffix)))
    for f in drive.list_files_in(remote_parent_id):
        if pat.match(f["name"]) and f["name"].endswith(f"_{file_hash}{suffix}"):
            print(f"Up-to-date version exists on Drive: {f['name']}. Skipping upload.")
            return

    rate = RateTracker(window_seconds=30)
    media = MediaFileUpload(str(local_file), mimetype=guess_mime(local_file), resumable=True, chunksize=upload_chunk_size)
    request = drive.svc.files().create(
        body={"name": want_name, "parents": [remote_parent_id]},
        media_body=media, fields="id", supportsAllDrives=True
    )

    total_bytes = os.path.getsize(local_file)
    response = None
    last = 0
    with tqdm(total=total_bytes, desc="Uploading", unit="B", unit_scale=True, unit_divisor=1024) as bar:
        while response is None:
            status, response = _retry(lambda: request.next_chunk())
            if status:
                sent = int(getattr(status, "resumable_progress", 0)) - last
                if sent > 0:
                    rate.add(sent); last += sent
                    bar.update(sent); bar.set_postfix_str(f"{rate.mbps():.2f} Mbps")
        if last < total_bytes:
            bar.update(total_bytes - last)
    print("Upload complete.")

def _sync_drive_versioned_to_file(
    drive: DriveSync,
    local_file: Path,
    remote_parent_id: str,
    stem: str,
    suffix: str,
    checksum: bool,
    download_chunk_size: int = DEFAULT_TRANSFER_CHUNK_SIZE,
):
    files = drive.list_files_in(remote_parent_id)
    latest = _pick_latest_versioned(files, stem, suffix)
    if not latest:
        print(f"No versioned file like {stem}_*.{suffix or ''} found on Drive. Nothing to do.")
        return

    _dt, hash_in_name = _parse_versioned_filename(stem, suffix, latest["name"])
    if local_file.exists() and compute_file_hash(local_file, checksum=checksum) == hash_in_name:
        print(f"Local file already matches hash {hash_in_name}. Skipping download.")
        return

    # NEW: ensure parent dir exists
    local_file.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = local_file.parent / latest["name"]  # download to versioned filename
    print(f"Downloading {latest['name']} from Drive ...")
    _download_file(drive, latest["id"], tmp_path, RateTracker(), chunk_size=download_chunk_size)

    tmp_path.replace(local_file)
    print(f"Wrote {local_file}")


# ================================
# Transfer helpers
# ================================
def _download_file(
    drive: DriveSync,
    file_id: str,
    dest_path: Path,
    rate: RateTracker,
    chunk_size: int = DEFAULT_TRANSFER_CHUNK_SIZE,
):
    from googleapiclient.http import MediaIoBaseDownload
    meta = drive.svc.files().get(fileId=file_id, fields="size", supportsAllDrives=True).execute()
    total_bytes = int(meta.get("size", 0)) if meta.get("size") is not None else None

    if (
        DEFAULT_ENABLE_PARALLEL_DOWNLOAD
        and
        total_bytes
        and total_bytes >= DEFAULT_PARALLEL_MIN_BYTES
        and DEFAULT_PARALLEL_DOWNLOAD_WORKERS > 1
    ):
        _download_file_parallel(
            drive=drive,
            file_id=file_id,
            dest_path=dest_path,
            total_bytes=total_bytes,
            rate=rate,
            chunk_size=chunk_size,
            workers=DEFAULT_PARALLEL_DOWNLOAD_WORKERS,
        )
        return

    request = drive.svc.files().get_media(fileId=file_id, supportsAllDrives=True)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with io.FileIO(str(dest_path), "wb") as fh, tqdm(
        total=total_bytes, desc="Downloading", unit="B", unit_scale=True, unit_divisor=1024
    ) as bar:
        downloader = MediaIoBaseDownload(fh, request, chunksize=chunk_size)
        done = False
        while not done:
            before = fh.tell()
            status, done = _retry(lambda: downloader.next_chunk())
            after = fh.tell()
            delta = max(0, after - before)
            if delta:
                rate.add(delta)
                bar.update(delta)

def _download_file_parallel(
    drive: DriveSync,
    file_id: str,
    dest_path: Path,
    total_bytes: int,
    rate: RateTracker,
    chunk_size: int,
    workers: int,
):
    req = drive.svc.files().get_media(fileId=file_id, supportsAllDrives=True)
    url = req.uri
    ranges = []
    start = 0
    while start < total_bytes:
        end = min(total_bytes - 1, start + chunk_size - 1)
        ranges.append((start, end))
        start = end + 1

    def fetch_range(start_end: Tuple[int, int]) -> int:
        s, e = start_end
        headers = {"Range": f"bytes={s}-{e}", "Accept-Encoding": "identity"}
        resp, content = _retry(
            lambda: drive.svc._http.request(url, method="GET", headers=headers)
        )
        status = int(getattr(resp, "status", 0))
        if status not in (200, 206):
            raise RuntimeError(f"Range request failed: HTTP {status} for bytes={s}-{e}")
        expected_len = e - s + 1
        got_len = len(content)
        if got_len != expected_len and not (status == 200 and s == 0 and got_len == total_bytes):
            raise RuntimeError(
                f"Range length mismatch for bytes={s}-{e}: expected {expected_len}, got {got_len}"
            )
        return s, content

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(dest_path), os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o644)
    try:
        os.ftruncate(fd, total_bytes)
        lock = threading.Lock()
        with tqdm(total=total_bytes, desc="Downloading", unit="B", unit_scale=True, unit_divisor=1024) as bar:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(fetch_range, r) for r in ranges]
                for fut in as_completed(futures):
                    s, content = fut.result()
                    os.pwrite(fd, content, s)
                    n = len(content)
                    with lock:
                        rate.add(n)
                        bar.update(n)
                        bar.set_postfix_str(f"{rate.mbps():.2f} Mbps")
    finally:
        os.close(fd)


def _retry(fn, retries: int = 6, base_delay: float = 0.8):
    from googleapiclient.errors import HttpError  # lazy
    for attempt in range(retries):
        try:
            return fn()
        except HttpError as e:
            if getattr(e, "resp", None) and e.resp.status in (403, 429, 500, 502, 503, 504):
                time.sleep(base_delay * (2 ** attempt) + 0.1 * (attempt + 1))
            else:
                raise
        except Exception:
            time.sleep(base_delay * (2 ** attempt) + 0.1 * (attempt + 1))
    return fn()


def _validate_tar_member(member: tarfile.TarInfo, tar_name: str) -> None:
    """
    Fast safety checks for tar extraction to prevent path traversal or special-file creation.
    Keeps overhead low for very large archives.
    """
    p = PurePosixPath(member.name)
    if p.is_absolute() or ".." in p.parts:
        raise RuntimeError(f"Unsafe tar member path '{member.name}' in '{tar_name}'")
    if member.issym() or member.islnk() or member.ischr() or member.isblk() or member.isfifo():
        raise RuntimeError(f"Unsupported/special tar member '{member.name}' in '{tar_name}'")
    if not (member.isdir() or member.isfile()):
        raise RuntimeError(f"Unsupported tar member type '{member.name}' in '{tar_name}'")
