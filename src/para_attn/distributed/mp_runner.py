import contextlib
import os
import socket
import threading
import time
import traceback

import torch.distributed as dist
import torch.multiprocessing as mp


class ExceptionOutput(Exception):
    pass


class MPDistRunner:
    def __init__(self, *, rank=None, persist_attrs=None, temp_attrs=None):
        self.rank = rank
        self.persist_attrs = persist_attrs or {}
        self.temp_attrs = temp_attrs or {}

        self.array_size = None
        self.shared_array = None
        self.input_queue = None
        self.output_queue = None
        self.status = None
        self.lock = None

        self.processes = []

    def __del__(self):
        self.terminate()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.terminate()

    @property
    def world_size(self):
        raise NotImplementedError

    @property
    def start_method(self):
        return "spawn"

    def clone(self, *, rank=None):
        return self.__class__(rank=rank, persist_attrs={**self.persist_attrs})

    def init_env(self, *, master_addr=None, master_port=None):
        if master_addr is None:
            if "MASTER_ADDR" not in os.environ:
                master_addr = "127.0.0.1"
                os.environ["MASTER_ADDR"] = master_addr
        else:
            os.environ["MASTER_ADDR"] = master_addr
        if master_port is None:
            if "MASTER_PORT" not in os.environ:
                master_port = find_free_port()
                os.environ["MASTER_PORT"] = str(master_port)
        else:
            os.environ["MASTER_PORT"] = str(master_port)

    def start(self, args=(), kwargs=None, *, timeout=None):
        if self.processes:
            raise RuntimeError("Processes are already started")

        if kwargs is None:
            kwargs = {}

        self.init_env()

        world_size = self.world_size
        if world_size <= 0:
            raise ValueError(f"World size {world_size} is invalid")

        start_method = self.start_method
        mp_ = mp.get_context(start_method)

        barrier = mp_.Barrier(world_size)
        input_queues = [mp_.JoinableQueue(maxsize=1) for _ in range(world_size)]
        output_queue = mp_.Queue(maxsize=1)
        exception_queues = [mp_.Queue(maxsize=1) for _ in range(world_size)]
        status = mp_.Value("i", 0, lock=True)
        processes = []

        try:
            for rank in range(world_size):
                process = mp_.Process(
                    target=self.worker,
                    args=(
                        self.clone(rank=rank),
                        args,
                        kwargs,
                        barrier,
                        input_queues[rank],
                        output_queue if rank == 0 else None,
                        exception_queues[rank],
                        status if rank == 0 else None,
                    ),
                    daemon=True,
                )
                process.start()
                processes.append(process)

            exceptions = []
            begin_time = time.time()
            while True:
                if timeout is not None and time.time() - begin_time >= timeout:
                    raise RuntimeError("Timeout during initialization")
                for rank, (process, exception_queue) in enumerate(zip(processes, exception_queues)):
                    if process.is_alive():
                        if exception_queue.empty():
                            continue
                        else:
                            exception = exception_queue.get()
                            if exception is None:
                                exceptions.append(None)
                            else:
                                exceptions.append((rank, exception))
                    else:
                        if exception_queue.empty():
                            exceptions.append((rank, RuntimeError(f"Process {rank} is not alive")))
                        else:
                            exception = exception_queue.get()
                            if exception is None:
                                exceptions.append(
                                    (rank, RuntimeError(f"Process {rank} is not alive after initialization"))
                                )
                            else:
                                exceptions.append((rank, exception))
                if len(exceptions) == world_size or any(e is not None for e in exceptions):
                    break
                time.sleep(0.0)
            exceptions = [e for e in exceptions if e is not None]
            if exceptions:
                msg = "\n".join(f"Rank {rank}: {exception}" for rank, exception in exceptions)
                raise RuntimeError(f"Exceptions occurred:\n{msg}")
        except Exception:
            self.terminate()
            raise

        self.input_queues = input_queues
        self.output_queue = output_queue
        self.exception_queues = exception_queues
        self.status = status
        self.lock = threading.RLock()
        self.processes = processes

        return self

    def terminate(self):
        self.input_queues = None
        self.output_queue = None
        self.exception_queues = None
        self.status = None

        for process in self.processes:
            if process.is_alive():
                process.terminate()
        self.processes = []

    def restart(self, *args, **kwargs):
        self.terminate()
        self.start(*args, **kwargs)

    def is_alive(self):
        return self.processes and all(process.is_alive() for process in self.processes)

    def is_idle(self):
        if not self.is_alive():
            return False

        with self.status.get_lock():
            return self.status.value == 0

    def is_almost_idle(self):
        if not self.is_alive():
            return False

        with self.status.get_lock():
            return self.status.value in (0, 2)

    def __call__(self, args=(), kwargs=None, *, timeout=None):
        if kwargs is None:
            kwargs = {}

        with self.lock:
            if not self.processes:
                raise RuntimeError("Processes are not started")
            for rank, process in enumerate(self.processes):
                if not process.is_alive():
                    raise RuntimeError(f"Process {rank} is not alive")

            if timeout is not None:
                begin_time = time.time()
            for input_queue in self.input_queues:
                input_queue.put((args, kwargs))
            for input_queue in self.input_queues:
                input_queue.join()
            if timeout is not None:
                end_time = time.time()
                duration = end_time - begin_time
                timeout = max(0.0, timeout - duration)
                if timeout == 0:
                    raise RuntimeError("Timeout during processing")
            begin_time = time.time()
            while True:
                if timeout is not None and time.time() - begin_time >= timeout:
                    raise RuntimeError("Timeout during initialization")
                if self.output_queue.empty():
                    for rank, process in enumerate(self.processes):
                        if not process.is_alive():
                            raise RuntimeError(f"Process {rank} is not alive")
                    time.sleep(0.0)
                    continue
                output = self.output_queue.get()
                exceptions = []
                for rank, exception_queue in enumerate(self.exception_queues):
                    if (rank == 0 and isinstance(output, ExceptionOutput)) or not exception_queue.empty():
                        exceptions.append((rank, exception_queue.get()))
                if exceptions:
                    msg = "\n".join(f"Rank {rank}: {exception}" for rank, exception in exceptions)
                    raise RuntimeError(f"Exceptions occurred:\n{msg}")
                if isinstance(output, ExceptionOutput):
                    raise RuntimeError("Exception occurred")
                return output

    def init_process_group(self):
        if dist.is_initialized():
            return
        dist.init_process_group(world_size=self.world_size, rank=self.rank)

    def destroy_process_group(self):
        if not dist.is_initialized():
            return
        dist.destroy_process_group()

    def init_processor(self):
        pass

    def process_task(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def worker(
        cls,
        runner,
        args,
        kwargs,
        barrier,
        input_queue,
        output_queue,
        exception_queue,
        status,
    ):
        runner.init_process_group()

        try:
            try:
                runner.init_processor(*args, **kwargs)
            except Exception as e:
                exception = RuntimeError(f"Failed to initialize processor: {e}\n{traceback.format_exc()}")
                if exception_queue is not None:
                    exception_queue.put(exception)
                # raise
                while True:
                    time.sleep(0.0)

            if exception_queue is not None:
                exception_queue.put(None)

            while True:
                input_args, input_kwargs = input_queue.get()

                if status is not None:
                    with status.get_lock():
                        status.value = 1

                output = None
                exception = None

                if output_queue is not None:
                    while not output_queue.empty():
                        output_queue.get()
                if exception_queue is not None:
                    while not exception_queue.empty():
                        exception_queue.get()
                input_queue.task_done()

                if exception is None:
                    try:
                        output = runner.process_task(*input_args, **input_kwargs)
                    except Exception as e:
                        exception = RuntimeError(f"Failed to process task: {e}\n{traceback.format_exc()}")

                if status is not None:
                    with status.get_lock():
                        status.value = 2

                if exception_queue is not None:
                    if exception is not None:
                        exception_queue.put(exception)

                if output_queue is not None:
                    if exception is None:
                        output_queue.put(output)
                    else:
                        output_queue.put(ExceptionOutput())

                barrier.wait()

                if status is not None:
                    with status.get_lock():
                        status.value = 0
        finally:
            runner.destroy_process_group()


def find_free_port():
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
