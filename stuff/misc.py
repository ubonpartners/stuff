import os
import sys
import json
import shutil
from pathlib import Path
import yaml
import pickle
import subprocess
import shlex

def makedir(path: str) -> None:
    """
    Create a directory, including any necessary parent directories.

    Args:
        path (str): The directory path to create.
    """
    Path(path).mkdir(parents=True, exist_ok=True)

def rename(source: str, destination: str) -> None:
    """
    Rename a file or directory.

    Args:
        source (str): The current name/path of the file or directory.
        destination (str): The new name/path for the file or directory.
    """
    os.rename(source, destination)

def rm(file_path: str) -> None:
    """
    Remove a file from the filesystem.

    Args:
        file_path (str): The path to the file to be removed.
    """
    os.remove(file_path)

def rmdir(directory_path: str) -> None:
    """
    Recursively remove a directory and all of its contents.

    Args:
        directory_path (str): The directory path to remove.
    """
    if os.path.isdir(directory_path):
        try:
            shutil.rmtree(directory_path)
        except OSError as error:
            print(f"Error removing directory '{error.filename}': {error.strerror}")

def load_dictionary(file_name: str):
    """
    Load a dictionary from a JSON or YAML file.

    This function automatically determines the file format based on the file extension.
    
    Args:
        file_name (str): The path to the file (JSON, YML, or YAML) to load.
    
    Returns:
        dict: The loaded dictionary if successful, otherwise None.
    """
    if file_name.endswith(".json"):
        with open(file_name, 'r') as json_file:
            return json.load(json_file)
    elif file_name.endswith((".yml", ".yaml")):
        with open(file_name, 'r') as yaml_file:
            try:
                return yaml.load(yaml_file, Loader=yaml.CSafeLoader)
            except yaml.YAMLError as exc:
                print(f"Error loading YAML file '{file_name}': {exc}")
                exit(1)
    else:
        print(f"Unsupported file extension for file: {file_name}")
        return None
    
def save_atomic_pickle(data, filename):
    """
    Pickles 'data' to filename atomically
    """
    temp_filename=filename+".tmp"
    if os.path.isfile(temp_filename):
        rm(temp_filename)
    with open(temp_filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        rename(temp_filename, filename) # atomic, replaces any existing file

def timestr(delta):
    delta=int(delta)
    s=delta%60
    m=(delta//60)%60
    h=delta//3600
    r=f"{m:02d}:{s:02d}"
    if h!=0:
        r=f"{h:02d}:"+r
    return r

def get_dict_param(dict, name, default):
    if dict is not None and name in dict:
        return dict[name]
    return default

def run_cmd(cmd, debug=False, timeout=240, num_attempts=3):
    if isinstance(cmd, str):
        cmd=shlex.split(cmd)

    if debug:
        subprocess.run(cmd)
        return
    for attempt in range(num_attempts):
        try:
            result=subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            print(f"⚠️ Command {cmd} attempt {attempt} timed out after {timeout} seconds")
            print(f"\n STDOUT: {e.stdout}")
            print(f" STDERR: {e.stderr}")
            print("======================\n")
            if attempt+1==num_attempts:
                sys.exit(1)
            print("Retrying....")
            continue
        stdout_str = result.stdout
        stderr_str = result.stderr
        if result.returncode!=0:
            print(f"⚠️ Command {cmd} attempt {attempt} failed returncode {result.returncode}")
            print(f"\n STDOUT: {stdout_str}")
            print(f" STDERR: {stderr_str}")
            print("======================\n")
            if attempt+1==num_attempts:
                sys.exit(1)
            print("Retrying....")
            continue
        break