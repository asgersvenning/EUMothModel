from subprocess import run, PIPE, Popen
from utils.config import get_config, deparse_args
import atexit, os

config = get_config()
remote, remote_subdir, local = config['remote'], config['remote_subdir'], config['local']

def mount(remote_directory = remote, remote_subdirectory = remote_subdir, local_directory = local):
    # Create local directory for remote mount
    mkdir_result = run(f'mkdir -p {local_directory}', shell=True, stdout=PIPE, stderr=PIPE)
    # Check if directory was created
    if mkdir_result.returncode != 0:
        print("Failed to create local directory")
        print(mkdir_result.stderr.decode("utf-8"))
        return False
    
    # Create mount command
    mount_cmd = f'rclone mount {remote_directory}:{remote_subdirectory} {local_directory}{deparse_args(config, "rclone")}'
    print(f"Mounting remote directory with command: {mount_cmd}")
    # Mount remote directory
    mount_result = run(mount_cmd, shell=True, stdout=PIPE, stderr=PIPE)
    # Check if directory was mounted
    if mount_result.returncode != 0:
        print("Failed to mount remote directory")
        # Remove local directory if mount failed
        rm_result = run(f'rm -d {local_directory}', shell=True, stdout=PIPE, stderr=PIPE)
        if rm_result.returncode != 0:
            print("Failed to remove local directory")
            print(rm_result.stderr.decode("utf-8"))
        else:
            print("Removed local directory for remote mount")
        return False
    
    # If mount was successful, register unmount function
    print("Mounted remote directory successfully")
    atexit.register(unmount, local_directory)
    return True

def unmount(local_directory = local):
    is_mounted = run(f'mountpoint -q {local_directory}', shell=True, stdout=PIPE, stderr=PIPE).returncode == 0

    if is_mounted:
        # Unmount remote directory
        unmount_result = run(f'fusermount -u {local_directory}', shell=True, stdout=PIPE, stderr=PIPE)
        if unmount_result.returncode != 0:
            print("Failed to unmount remote directory")
            print(unmount_result.stderr.decode("utf-8"))
            return_value = False
        else:
            print("Unmounted remote directory")
    else:
        print("Remote directory is not mounted")
        return_value = False
    
    # Remove local directory if present
    if os.path.isdir(local_directory):
        rm_result = run(f'rm -d {local_directory}', shell=True, stdout=PIPE, stderr=PIPE)
        if rm_result.returncode != 0:
            print("Failed to remove local directory")
            print(rm_result.stderr.decode("utf-8"))
            return_value = False
        else:
            print("Removed local directory for remote mount")
    else:
        print("Local directory does not exist")
        return_value = False

    atexit.unregister(unmount)
    return return_value