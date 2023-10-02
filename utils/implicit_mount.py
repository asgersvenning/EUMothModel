import subprocess, os, warnings, tempfile, shutil, time, uuid, re, threading, queue
from typing import List, Union, Set, Tuple
from multiprocessing import Pool, cpu_count
from queue import Queue

from utils.config import get_implicit_mount_config

class ImplicitMount:
    time_stamp_pattern = re.compile(r"^\s*(\S+\s+){8}") # This is used to strip the timestamp from the output of the lftp shell
    END_OF_OUTPUT = '# LFTP_END_OF_OUTPUT_IDENTIFIER {uuid} #'  # This is used to signal the end of output when reading from stdout

    def __init__(self, user: str= None, remote: str=None, strict: bool=False, verbose: bool=False):
        # Default argument configuration and type checking
        self.default_config = get_implicit_mount_config()
        if user is None:
            user = self.default_config['user']
        if remote is None:
            remote = self.default_config['remote']
        if not isinstance(user, str):
            raise TypeError("Expected str, got {}".format(type(user)))
        if not isinstance(remote, str):
            raise TypeError("Expected str, got {}".format(type(remote)))
        if not isinstance(strict, bool):
            raise TypeError("Expected bool, got {}".format(type(strict)))
        if not isinstance(verbose, bool):
            raise TypeError("Expected bool, got {}".format(type(verbose)))
        
        # Set attributes
        self.user = user
        self.remote = remote
        self.strict = strict
        self.lftp_shell = None
        self.verbose = verbose

        self.stdout_queue = Queue()
        self.stderr_queue = Queue()
        self.lock = threading.Lock()

    @staticmethod
    def format_options(**kwargs) -> str:
        options = []
        for key, value in kwargs.items():
            # print(f'key: |{key}|, value: |{value}|')
            prefix = "-" if len(key) == 1 else "--"
            this_option = f"{prefix}{key}"
            
            if value is not None and value != "":
                this_option += f" {str(value)}"
                
            options.append(this_option)
            
        options = " ".join(options)
        return options
    
    def _readerthread(self, stream, queue: Queue):  # No longer static
        while True:
            output = stream.readline()
            if output:
                queue.put(output)
            else:
                break

    def _read_stdout(self, timeout: float = 0, strip_timestamp: bool = True, uuid_str: str = None) -> List[str]:
        EoU = self.END_OF_OUTPUT.format(uuid=uuid_str)
        lines = []
        start_time = time.time()
        while True:
            if timeout and (time.time() - start_time > timeout):
                raise TimeoutError("Timeout while reading stdout")
            if not self.stdout_queue.empty():
                line = self.stdout_queue.get()
                if EoU in line:
                    break
                if strip_timestamp:
                    line = re.sub(self.time_stamp_pattern, "", line)
                # if not line.startswith("wait "):
                if self.verbose:
                    print(line.strip())
                lines.append(line.strip())
            else:
                err = self._read_stderr()
                if err and not err.startswith("wait: no current job"):
                    raise Exception(f"Error while executing command: {err}")
                time.sleep(0.001)  # Moved to the else clause
        return lines

    def _read_stderr(self) -> str:
        errors = []
        while not self.stderr_queue.empty():
            errors.append(self.stderr_queue.get())
        return ''.join(errors)
    
    def execute_command(self, command: str, output: bool=True, blocking: bool=True, execute: bool=True, default_args: Union[dict, None]=None, **kwargs) -> Union[str, List[str], None]:
        # Merge default arguments and keyword arguments. Keyword arguments will override default arguments.
        if not isinstance(default_args, dict) and default_args is not None:
            raise TypeError("Expected dict or None, got {}".format(type(default_args)))
        
        # Combine default arguments and keyword arguments
        if default_args is None:
            default_args = {}
        # Remove optional "uuid_str" argument from kwargs
        if "uuid_str" in kwargs:
            uuid_str = kwargs.pop("uuid_str")
        else:
            uuid_str = None
        # Combine default arguments and kwargs
        args = {**default_args, **kwargs} 
        
        # Format arguments
        formatted_args = self.format_options(**args)

        # Combine command and arguments
        full_command = f"{command} {formatted_args}" if formatted_args else command
        if output:
            if uuid_str is None:
                uuid_str = str(uuid.uuid4())
        if not blocking:
            full_command += " &"

        if not execute:
            return full_command
        else:
            output = self._execute_command(full_command, output=output, blocking=blocking, uuid_str=uuid_str)
            if isinstance(output, list):
                if len(output) == 0:
                    return None
                elif len(output) == 1:
                    return output[0]
                else:
                    return output
            else:
                return None

    def _execute_command(self, command: str, output: bool=True, blocking: bool=True, uuid_str: Union[str, None]=None) -> Union[List[str], None]:
        if output and not blocking:
            raise ValueError("Non-blocking output is not supported.")
        if uuid_str is None and (blocking or output):
            uuid_str = str(uuid.uuid4())
            # raise ValueError("uuid_str must be specified if output is True.")
        if not command.endswith("\n"):
            command += "\n"
        if self.verbose:
            print(f"Executing command: {command}")
        with self.lock: 
            # Execute command
            self.lftp_shell.stdin.write(command)
            self.lftp_shell.stdin.flush()

            # Blocking and end of output logic
            if blocking or output:
                EoU_cmd = f"echo {self.END_OF_OUTPUT.format(uuid=uuid_str)}\n"
                if self.verbose:
                    print(f"Executing command: {EoU_cmd}")
                self.lftp_shell.stdin.write(EoU_cmd)
                self.lftp_shell.stdin.flush()
            if blocking:
                if self.verbose:
                    print("Executing command: wait")
                self.lftp_shell.stdin.write("wait\n")
                self.lftp_shell.stdin.flush()
            if output:
                return self._read_stdout(uuid_str=uuid_str)
            elif blocking:
                self._read_stdout(uuid_str=uuid_str)
                return None

    def mount(self, lftp_settings: Union[dict, None]=None) -> None:
        # set mirror:use-pget-n 5;set net:limit-rate 0;set xfer:parallel 5;set mirror:parallel-directories true;set ftp:sync-mode off;"
        # Merge default settings and user settings. User settings will override default settings.
        lftp_settings = {**self.default_config['lftp'], **lftp_settings} if lftp_settings is not None else self.default_config['lftp']
        # Format settings
        lftp_settings_str = ""
        for key, value in lftp_settings.items():
            lftp_settings_str += f" set {key} {value};"
        if self.verbose:
            print(f"Mounting {self.remote} as {self.user}")
        # "Mount" the remote directory using an lftp shell with the sftp protocol and the specified user and remote, and the specified lftp settings
        lftp_mount_cmd = f'open -u {self.user} -p 2222 sftp://{self.remote};{lftp_settings_str}'
        if self.verbose:
            print(f"Executing command: lftp")
        # Start the lftp shell
        try:
            self.lftp_shell = subprocess.Popen(
                executable="lftp",
                args=[],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0,
            )
        except Exception as e:
            raise Exception("Failed to start subprocess.") from e

        # Start the stdout and stderr reader threads
        self.stdout_thread = threading.Thread(
            target=self._readerthread,
            args=(self.lftp_shell.stdout, self.stdout_queue)
        )
        self.stderr_thread = threading.Thread(
            target=self._readerthread,
            args=(self.lftp_shell.stderr, self.stderr_queue)
        )

        # Set the threads as daemon threads so that they will be terminated when the main thread terminates
        self.stdout_thread.daemon = True
        self.stderr_thread.daemon = True
        
        # Start the threads
        self.stdout_thread.start()
        self.stderr_thread.start()

        # Execute the mount command on the lftp shell (connect to the remote directory)
        self.execute_command(lftp_mount_cmd, output=False, blocking=False)

        if self.verbose:
            print("Waiting for connection...")
        # Check if we are connected to the remote directory by executing the pwd command on the lftp shell
        # If we are connected, the output of the pwd command will be the path of the remote directory
        # If we are not connected, the output of the pwd command will be empty. Then we raise an error and terminate the lftp shell
        connected_path = self.pwd()
        print(f"Connected to {connected_path}")
        if not connected_path:
            self.unmount()
            raise RuntimeError(f"Failed to connect. Check internet connection or if {self.remote} is online.")

    def unmount(self, timeout: float = 1) -> None:
        # Close Popen streams explicitly
        self.execute_command("exit kill top", output=False, blocking=False)
        waited = 0
        while True:
            if self.lftp_shell.poll() is not None:
                break
            time.sleep(0.1)
            waited += 0.1
            if waited > timeout:
                self.lftp_shell.terminate()
                break
        self.lftp_shell.stdout.close()
        self.lftp_shell.stdin.close()
        self.lftp_shell.stderr.close()
        self.lftp_shell = None

    def pget(self, remote_path: str, local_destination: str, blocking: bool=True, execute: bool=True, output: Union[bool, None]=None, **kwargs):
        if output is None:
            output = blocking
        default_args = {'n': 5}
        args = {**default_args, **kwargs}
        formatted_args = self.format_options(**args)
        full_command = f"pget {formatted_args} {remote_path} -o {local_destination}"
        exec_output = self.execute_command(
            full_command, 
            output=output, 
            blocking=blocking,
            execute=execute
        )
        if not execute:
            return exec_output
        
        # Construct and return the absolute local path
        file_name = os.path.basename(remote_path)
        abs_local_path = os.path.abspath(os.path.join(local_destination, file_name))
        return abs_local_path

    def ls(self, path: str = ".", recursive: bool=False, use_cache: bool=True) -> List[str]:
        if path.startswith(".."):
            raise NotImplementedError("ls does not support relative backtracing paths yet.")
        elif path.startswith("./"):
            path = path[2:]
        elif path == ".":
            path = ""
        # Recursive ls is implemented by using the "cls" command, which returns a list of permissions and paths
        # and then recursively calling ls on each of the paths that are directories, 
        # which is determined by checking if the permission starts with "d"
        if recursive:
            recls = "" if use_cache else "re"
            this_level = self.execute_command(f"{recls}cls {path} -1 --perm")
            if isinstance(this_level, str):
                this_level = [this_level]
            output = []
            for perm_path in this_level:
                if not " " in perm_path:
                    continue
                perm, path = perm_path.split(" ", 1)
                if perm.startswith("d"):
                    output += self.ls(path, recursive=True)
                else:
                    if path.startswith("."):
                        path = path[2:]
                    output += [path]
            return output
        else:
            output = self.execute_command(f"cls {path} -1")
        if path.startswith("."):
            if isinstance(output, list):
                return [i[2:] for i in output]
            else:
                return output[2:]
        else:
            return output
    
    def lls(self, local_path: str, **kwargs):
        # TODO: 
        # currently the value of the "R" or "recursive" argument is ignored, 
        # the recursive version is always used if either of these arguments are specified
        # (and the non-recursive version is always used if neither of these arguments are specified)
        if "R" in kwargs or "recursive" in kwargs:
            if local_path == "":
                local_path = "."
            return self.execute_command(f"!find {local_path} -type f -exec realpath --relative-to={local_path} {{}} \;")
        return self.execute_command(f"!ls {local_path}", **kwargs)

    def cd(self, remote_path: str, **kwargs):
        self.execute_command(f"cd {remote_path}", output=False, **kwargs)

    def pwd(self) -> str:
        return self.execute_command("pwd")

    def lcd(self, local_path: str) -> str:
        self.execute_command(f"lcd {local_path}", output=False)

    def lpwd(self) -> str:
        return self.execute_command("lpwd")
    
    def _get_current_files(self, dir_path: str) -> Set[str]:
        return self.lls(dir_path, R="")

    def mirror(self, remote_path: str, local_destination: str, blocking: bool=True, execute: bool=True, **kwargs):
         # Capture the state of the directory before the operation
        pre_existing_files = self._get_current_files(local_destination)
        if isinstance(pre_existing_files, str):
            pre_existing_files = list(pre_existing_files)
        if pre_existing_files:
            pre_existing_files = set(pre_existing_files)
        else:
            pre_existing_files = set()

        # Execute the mirror command
        default_args = {'P': 5, 'use-cache': None}
        exec_output = self.execute_command(
            f"mirror {remote_path} {local_destination}", 
            output=blocking, 
            blocking=blocking, 
            execute=execute,
            default_args=default_args, 
            **kwargs
        )
        if not execute:
            return exec_output
        
        # Capture the state of the directory after the operation
        post_download_files = self._get_current_files(local_destination)
        if isinstance(post_download_files, str):
            post_download_files = list(post_download_files)
        if post_download_files:
            post_download_files = set(post_download_files)
        else:
            post_download_files = set()

        # Calculate the set difference to get the newly downloaded files
        new_files = post_download_files - pre_existing_files
        
        return list(new_files)
        

class RemotePathIterator:
    def __init__(self, io_handler: "IOHandler", batch_size: int=64, batch_parallel: int=10, max_queued_batches: int=3, n_local_files: int=3*64, **kwargs):
        self.io_handler = io_handler
        if "file_index" not in self.io_handler.cache:
            self.remote_paths = self.io_handler.get_file_index(**kwargs)
        else:
            if kwargs:
                warnings.warn(f'Using cached file index. [{", ".join(kwargs.keys())}] will be ignored.')
            self.remote_paths = self.io_handler.cache["file_index"]
        self.temp_dir = self.io_handler.lpwd()
        self.batch_size = batch_size
        self.batch_parallel = batch_parallel
        self.max_queued_batches = max_queued_batches
        self.n_local_files = n_local_files
        if self.n_local_files < self.batch_size:
            warnings.warn(f"n_local_files ({self.n_local_files}) is less than batch_size ({self.batch_size}). This may cause files to be deleted before they are consumed. Consider increasing n_local_files. Recommended value: {self.batch_size * self.max_queued_batches}")
        self.idx = 0
        self.download_queue = Queue()
        self.delete_queue = Queue()
        self.stop_requested = False
        self.not_cleaned = True

        # State variables
        self.download_thread = None
        self.last_item = None
        self.last_batch_consumed = 0
        self.consumed_files = 0

    def download_files(self):
        queued_batches = 0
        for i in range(0, len(self.remote_paths), self.batch_size):
            if self.stop_requested:
                break
                
            while queued_batches >= self.max_queued_batches and not self.stop_requested:
                # Wait until a batch has been consumed (or multiple batches, if the consumer is fast and the producer is slow) before downloading another batch
                if self.last_batch_consumed > 0:
                    self.last_batch_consumed -= 1
                    break
                time.sleep(0.01)  # Wait until a batch has been consumed

            batch = self.remote_paths[i:i + self.batch_size]
            local_paths = self.io_handler.download(batch, n = self.batch_parallel)
            for local_path, remote_path in zip(local_paths, batch):
                self.download_queue.put((local_path, remote_path))
                
            queued_batches += 1
            
            # Deletion logic moved to __next__ to maintain minimal queued files

    def start_download_queue(self) -> None:
        self.download_thread = threading.Thread(target=self.download_files)
        self.download_thread.start()

    def __iter__(self) -> "RemotePathIterator":
        self.start_download_queue()
        return self
    
    def __len__(self) -> int:
        return len(self.remote_paths)

    def __next__(self) -> Tuple[str, str]:
        # Delete files if the queue is too large
        while self.delete_queue.qsize() > self.n_local_files:
            try:
                os.remove(self.delete_queue.get())
            except Exception as e:
                warnings.warn(f"Failed to remove file: {e}")

        # Handle stop request and end of iteration
        if self.stop_requested or self.idx >= len(self.remote_paths):
            self.__del__()
            raise StopIteration

        # Get next item from queue or raise error if queue is empty
        try:
            next_item = self.download_queue.get() # Timeout not applicable, since there is no guarantees on the size of the files or the speed of the connection
            # Update state to ensure that the producer keeps the queue prefilled
            # It is a bit complicated because the logic must be able to handle the case where the consumer is faster than the producer,
            # in this case the producer may be multiple batches behind the consumer.
            self.consumed_files += 1
            if self.consumed_files >= self.batch_size:
                self.consumed_files -= self.batch_size
                self.last_batch_consumed += 1
        except queue.Empty: # TODO: Can this happen?
            if self.stop_requested:
                self.__del__()
                raise StopIteration
            else:
                raise RuntimeError("Download queue is empty but no stop was requested. Check the download thread.")

        # Update state
        self.idx += 1
        self.delete_queue.put(next_item)
        
        # Return next item
        return next_item

    def __del__(self) -> None:
        if self.not_cleaned:
            self.stop_requested = True
            self.download_thread.join(timeout=10)
            while not self.download_queue.empty():
                try:
                    os.remove(self.download_queue.get())
                except Exception as e:
                    warnings.warn(f"Failed to remove file: {e}")
                    

class IOHandler(ImplicitMount):
    def __init__(self, local_dir: Union[str, None]=None, user_confirmation: bool=False, clean: Union[bool, None]=None, n_threads: Union[int, None]=None, **kwargs):
        super().__init__(**kwargs)
        if local_dir is None:
            if self.default_config['local_dir'] is None:
                local_dir = tempfile.TemporaryDirectory().name
            else:
                local_dir = self.default_config['local_dir']
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        self.original_local_dir = os.path.abspath(local_dir)
        self.local_dir = local_dir
        self.user_confirmation = user_confirmation
        self.do_clean = clean if clean is not None else True
        self.last_download = None
        self.last_type = None
        self.cache = {}
        self.pool = None
        if n_threads and n_threads > 1: # There are some lines here and there that make it seem like multi-threaded operation is supported, but it is not. This is a TODO.
            raise NotImplementedError("Multi-threaded operation is not implemented yet.")
        self.n_threads = n_threads if n_threads else 1

    def iter(self, remote_path: Union[str, List[str]]) -> "RemotePathIterator":
        iter_temp_dir = tempfile.TemporaryDirectory()
        return RemotePathIterator(self, remote_path, iter_temp_dir)

    def __enter__(self) -> "IOHandler":
        self.mount()
        self.lcd(self.local_dir)
        # Check if the number of threads is valid
        if not isinstance(self.n_threads, int) or self.n_threads < 0:
            raise TypeError("Expected a positive int, got {} ({})".format(self.n_threads, type(self.n_threads)))

        if self.n_threads > cpu_count():
            warnings.warn(f"Number of threads {self.n_threads} is greater than number of CPUs {cpu_count()}. Using {cpu_count()} threads instead.")
            self.n_threads = cpu_count()

        # Create pool
        if self.n_threads > 1:
            self.pool = Pool(self.n_threads)

        # Messages
        print(f"Local directory: {self.lpwd()}")

        # Return self
        return self

    def __exit__(self, *args, **kwargs):
        # Positional and keyword arguments simply catch any arguments passed to the function to be ignored
        if self.do_clean:
            self.clean()
        self.unmount()
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
        self.pool = None

    # Methods for using the IOHandler without context management
    def start(self) -> None:
        self.__enter__()
        print("IOHandler.start() is unsafe. Use IOHandler.__enter__() instead if possible.")
        print("OBS: Remember to call IOHandler.stop() when you are done.")

    def stop(self) -> None:
        self.__exit__()

    def download(self, remote_path: Union[str, List[str]], local_destination: Union[str, List[str], None]=None, blocking: bool=True, **kwargs) -> Union[str, List[str]]:
        if not isinstance(remote_path, str) and len(remote_path) > 1:
            return self.multi_download(remote_path, local_destination, **kwargs)
        
        if local_destination is None:
            local_destination = self.lpwd()

        # Check if remote and local have file extensions:
        # The function assumes files have extensions and directories do not.
        remote_has_ext, local_has_ext = os.path.splitext(remote_path)[1] != "", os.path.splitext(local_destination)[1] != ""

        # If both remote and local have file extensions, the local destination should be a file path.
        if remote_has_ext and local_has_ext and os.path.isdir(local_destination):
            raise ValueError("Destination must be a file path if both remote and local have file extensions.")

        # If the remote does not have a file extension, the local destination should be a directory.
        if not remote_has_ext and not os.path.isdir(local_destination):
            raise ValueError("Destination must be a directory if remote does not have a file extension.")
        
        # If the remote is a single file, use pget.
        if remote_has_ext:
            local_result = self.pget(remote_path, local_destination, blocking, **kwargs)
            self.last_type = "file"
        # Otherwise use mirror.
        else:
            if not os.path.exists(local_destination):
                try:
                    os.makedirs(local_destination)
                except FileExistsError:
                    pass
            local_result = self.mirror(remote_path, local_destination, blocking, **kwargs)
            self.last_type = "directory"
        
        # TODO: Check local_result == local_destination

        self.last_download = local_result
        return local_result
    
    def multi_download(self, remote_paths: List[str], local_destination: Union[str, List[str]], blocking: bool=True, n: int=5, **kwargs) -> List[str]:
        if not isinstance(remote_paths, list):
            raise TypeError("Expected list, got {}".format(type(remote_paths)))
        if not (isinstance(local_destination, str) or isinstance(local_destination, list) or local_destination is None):
            raise TypeError("Expected str or list, got {}".format(type(local_destination)))
        if isinstance(local_destination, str):
            local_destination = [local_destination + os.sep + os.path.basename(r) for r in remote_paths]
        elif local_destination is None:
            local_destination = self.lpwd()
            local_destination = [local_destination + os.sep + os.path.basename(r) for r in remote_paths]
        if len(remote_paths) != len(local_destination):
            raise ValueError("remote_paths and local_destination must have the same length.")
        if any([os.path.splitext(l)[1] != os.path.splitext(r)[1] for l, r in zip(local_destination, remote_paths)]):
            raise ValueError("Local and remote file extensions must match.")
        
        ### Legacy code: 
        ### Previously, this was done by queueing pget commands and then executing them in a single thread.
        #
        # single_commands = []
        # for r, l in zip(remote_paths, local_destination):
        #     single_commands += [self.pget(r, l, blocking=True, output=False, execute=False, **kwargs)]
        # multi_command = "(" + " & ".join(single_commands) + ")"
        #
        # Download the files in parallel using a single mget command 
        # (this is much faster than using many pget commands)
        # 
        
        single_commands = []
        for r, l in zip(remote_paths, local_destination):
            single_commands += [l + " -o " + r]

        multi_command = f"mget -P {n} " + " ".join(remote_paths)
        self.execute_command(multi_command, output=blocking, blocking=blocking)
        for l in local_destination:
            if not os.path.exists(l):
                raise RuntimeError(f"Failed to download {l}")

        self.last_download = local_destination
        self.last_type = "multi"
        return local_destination
    
    def get_file_index(self, skip: int=0, nmax: Union[int, None]=None, override: bool=False) -> List[str]:
        # Check if file index exists
        files_in_dir = self.ls()
        file_index_exists = "folder_index.txt" in files_in_dir
        if not file_index_exists:
            raise RuntimeError(f"Folder index does not exist in {files_in_dir}")
        # If override is True, delete the file index if it exists
        if override and file_index_exists:
            self.execute_command("rm folder_index.txt")
            # Now the file index does not exist (duh)
            file_index_exists = False
        # If the file index does not exist, create it
        if not file_index_exists:
            raise NotImplementedError("Creating file index is not implemented yet.")
            # TODO: Fix the command below.
            # This works in a local shell, but not in the lftp shell:
            # self.execute_command("find . -type f -exec realpath --relative-to=. {} \; > folder_index.txt")
        
        # Download the file index
        file_index_path = self.download("folder_index.txt")
        # Read the file index
        file_index = []
        with open(file_index_path, "r") as f:
            for i, line in enumerate(f):
                if i < skip:
                    continue
                if nmax is not None and i >= (skip + nmax):
                    break
                file_index.append(line.strip())
        return file_index
    
    def cache_file_index(self, skip: int=0, nmax: Union[int, None]=None, override: bool=False) -> None:
        self.cache["file_index"] = self.get_file_index(skip, nmax, override)

    def store_last(self, dst: str):
        # If the last download was a single file, the destination should be a file.
        # Otherwise, it should be a directory.
        if self.last_download is None:
            warnings.warn("No last download to store")
            return
        if self.last_type == "unknown":
            # This should only happen if the last download was a multi download
            # In this case the last type must be inferred from the last_download path (by checking if it has a file extension)
            self.last_type = "file" if os.path.splitext(self.last_download)[1] != "" else "directory"
        # Check if destination has file extension (if last download was a file) or not (if last download was a directory)
        dst_has_ext = os.path.splitext(dst)[1] != ""
        if self.last_type == "file":
            if not dst_has_ext:
                raise ValueError("Destination must have file extension if it is NOT a directory.")
            # self.execute_command(f"local mv {self.last_download} {dst}")
            # Use shutil instead
            shutil.move(self.last_download, dst)
        elif self.last_type == "directory":
            if dst_has_ext:
                raise ValueError("Destination must NOT have file extension if it IS a directory.")
            # self.execute_command(f"local mv {self.last_download} {dst}")
            # Use shutil instead
            shutil.move(self.last_download, dst)
            
        # This part is too complicated, and is used to handle the case where the last download was a multi download only
        else:
            if self.last_type == "multi":
                self.last_type = "unknown"
                last_downloads = self.last_download.deepcopy()
                for path in last_downloads:
                    self.last_download = path
                    self.store_last(dst)
            else:
                self.clean_last()

    def clean(self):
        if self.user_confirmation:
            # Ask for confirmation
            confirmation = input(f"Are you sure you want to delete all files in the current directory {self.lpwd()}? (y/n)")
            if confirmation.lower() != "y":
                print("Aborted")
                return
        print("Cleaning up...")
        shutil.rmtree(self.lpwd())

    def clean_last(self):
        if self.last_download is None:
            warnings.warn("No last download to clean")

        if self.user_confirmation:
            # Ask for confirmation
            if self.last_download == "directory":
                confirmation = input(f"Are you sure you want to delete all files in {self.last_download}? (y/n)")
            elif self.last_download == "file":
                confirmation = input(f"Are you sure you want to delete {self.last_download}? (y/n)")
            elif self.last_download == "multi":
                confirmation = input(f"Are you sure you want to delete all files and or contents of {self.last_download}? (y/n)")
            else:
                raise RuntimeError(f"Unknown last type {self.last_type}")
            if confirmation.lower() != "y":
                print("Aborted")
                return
        
        if self.original_local_dir == self.lpwd():
            if self.last_type == "file":
                os.remove(self.last_download)
            elif self.last_type == "directory":
                shutil.remove(self.last_download)
            elif self.last_type == "multi":
                for path in self.last_download:
                    if os.path.exists(path):
                        if os.path.isfile(path):
                            os.remove(path)
                        else:
                            shutil.rmtree(path)
            else:
                raise RuntimeError(f"Unknown last type {self.last_type}")

            self.last_download = None
            self.last_type = None
        else:
            warnings.warn("Last download was not in original local directory. Not cleaning.")

