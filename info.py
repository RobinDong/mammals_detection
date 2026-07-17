import subprocess
import platform
import sys
import os

def get_git_info():
    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        return branch, commit
    except Exception as e:
        return "Unknown", "Unknown"

def get_nvidia_info():
    try:
        driver_info = subprocess.check_output(["nvidia-smi"]).decode()
        cuda_version = None
        driver_version = None

        for line in driver_info.splitlines():
            if "Driver Version" in line and "CUDA Version" in line:
                parts = line.split()
                driver_index = parts.index("Version:") + 1
                cuda_index = parts.index("CUDA") + 2
                driver_version = parts[driver_index]
                cuda_version = parts[cuda_index]
                break

        return driver_version or "Unknown", cuda_version or "Unknown"
    except Exception:
        return "Unknown", "Unknown"

def get_kernel_version():
    return platform.platform()

def get_os_version():
    return subprocess.check_output(["cat", "/etc/issue"]).decode().strip()

def get_python_version():
    return sys.version

def get_pytorch_version():
    try:
        import torch
        return torch.__version__
    except ImportError:
        return "PyTorch not installed"

def get_dataset_dir_stats(path):
    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a valid directory.")

    # Get first-level subdirectories
    subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    num_subdirs = len(subdirs)

    # Count total files recursively
    total_files = 0
    for root, _, files in os.walk(path):
        total_files += len(files)

    # Get directory size using du -sh
    try:
        du_output = subprocess.check_output(['du', '-sh', path]).decode().split()[0]
    except Exception as e:
        du_output = "Unknown"

    print("Dataset path:", path)
    print("Dataset num_subdirectories:", num_subdirs)
    print("Dataset total_files:", total_files)
    print("Dataset directory_size:", du_output)

def main():
    print("=== System Information ===")
    branch, commit = get_git_info()
    print(f"Git Branch: {branch}")
    print(f"Git Commit: {commit}")

    driver_version, cuda_version = get_nvidia_info()
    print(f"NVIDIA Driver Version: {driver_version}")
    print(f"CUDA Version: {cuda_version}")

    print(f"OS Version: {get_os_version()}")
    print(f"Linux Kernel Version: {get_kernel_version()}")
    print(f"Python Version: {get_python_version()}")
    print(f"PyTorch Version: {get_pytorch_version()}")

    get_dataset_dir_stats(sys.argv[1])

if __name__ == "__main__":
    main()
