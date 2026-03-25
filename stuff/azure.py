import os
import subprocess
import yaml
from pathlib import Path
from typing import Optional

def fetch_file_from_azure(file_path: str, blobfuse_yml_path: Optional[str] = None):
    """
    Ensure a file exists locally; if not, fetch it from Azure using azcopy copy (single file).

    Args:
        file_path (str): Absolute local file path, must start with /mldata/.
        blobfuse_yml_path (Optional[str]): Path to the blobfuse2.yml config file.
                                           If None, uses BLOBFUSE_YML_PATH environment variable.
    """
    if not file_path.startswith("/mldata/"):
        raise ValueError("file_path must start with /mldata/")

    local_file = Path(file_path)
    if local_file.exists():
        print(f"File already exists locally: {file_path}")
        return
    else:
        print(f"File {file_path} does not exist locally, fetching ONNX")

    # Determine blobfuse config path
    if blobfuse_yml_path is None:
        blobfuse_yml_path = os.environ.get("BLOBFUSE_YML_PATH")
        if blobfuse_yml_path is None:
            raise ValueError("blobfuse_yml_path not provided and BLOBFUSE_YML_PATH environment variable is not set.")

    # Load Azure config
    with open(blobfuse_yml_path) as f:
        cfg = yaml.safe_load(f)
        endpoint = cfg['azstorage']['endpoint']
        container = cfg['azstorage']['container']
        sas = cfg['azstorage']['sas']

    # Construct blob URL and destination
    relative_path = Path(file_path).relative_to("/mldata")
    blob_url = f"{endpoint}/{container}/{relative_path}{sas}"

    # Ensure parent directory exists
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    # Execute azcopy copy
    cmd = f'azcopy copy "{blob_url}" "{file_path}"'
    print("Executing:", cmd)
    return_code = subprocess.call(cmd, shell=True, env=os.environ.copy())

    if return_code != 0:
        raise RuntimeError(f"azcopy failed with code {return_code}")
    elif local_file.exists():
        print(f"Successfully copied file: {file_path}")
    else:
        raise FileNotFoundError(f"File not found after azcopy: {file_path}")