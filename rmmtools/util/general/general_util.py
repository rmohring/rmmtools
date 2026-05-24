import os
import datetime
import functools
from typing import Union, Iterable, List, Any, Optional

import logging
import logging.handlers

import yaml

from .datetime_util import get_now_local, get_now_dttmstr_local

__all__ = [
    "get_logger",
    "plog",
    "logwrap",
    "get_filename",
    "safe_filename",
    "read_yaml",
    "strip_nones",
    "check_not_none",
    "get_val_or_alt_or_raise",
    "set_val_if_none",
    "listify",
    "check_mem",
]


## Logging functions
def get_logger(
    name: str, level: Optional[str] = None, log_file: Optional[str] = None
) -> logging.Logger:
    """Creates and returns a logger instance.

    Args:
        name (str): Name of the logger.
        level (str, optional): Logging level. Defaults to "debug".
        log_file (str, optional): Path to log file. Defaults to None.

    Returns:
        logging.Logger: Configured logger instance.
    """
    log = logging.getLogger(name)
    level = set_val_if_none(level, "debug")

    log.setLevel(getattr(logging, level.upper()))
    # formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # The default stream handler is to stderr
    fmt = "%(asctime)s - %(name)-15s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt)
    logging.basicConfig(format=fmt)

    # logging.StreamHandler()
    # # stdout stream
    # sh = logging.StreamHandler(sys.stdout)
    # sh.setFormatter(formatter)
    # log.addHandler(sh)

    # rotating log file
    if log_file is not None:
        fh = logging.handlers.RotatingFileHandler(log_file, backupCount=7)
        fh.setFormatter(formatter)
        log.addHandler(fh)
    return log


def plog(text: str = "now", flush: bool = True):
    """Prints a message with the current timestamp.

    Args:
        text (str, optional): Message text. Defaults to "now".
        flush (bool, optional): Flush the output. Defaults to True.
    """
    print(f"{get_now_local()}: {text}", flush=flush)
    return


def logwrap(func):
    """Decorator to log the start and end of a function, for use within a class
    (the reference to `self` is explicit).

    Args:
        func (function): Function to be wrapped.

    Returns:
        function: Wrapped function with logging.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        msg = func.__name__
        self.log.info(f"Start: {msg}")
        start_time = datetime.datetime.now()
        result = func(self, *args, **kwargs)
        end_time = datetime.datetime.now()
        self.log.info(f"End: {msg} ({end_time - start_time} elapsed)")
        return result

    return wrapper


## File-related functions
def get_filename(path, fname):
    return os.path.join(os.path.expanduser(path), fname)


def safe_filename(
    filename: str = None,
    clobber: bool = False,
    raise_on_error: bool = False,
) -> Union[str, None]:
    """Generates a safe filename by appending a timestamp if necessary.

    Args:
        filename (str): Original filename.
        clobber (bool, optional): Whether to overwrite existing files.
            Defaults to False.
        raise_on_error (bool, optional): Raise an error if the file exists and clobber
            is False. Defaults to False.

    Returns:
        Union[str, None]: Adjusted filename or None if error.

    Raises:
        ValueError: If filename exists and clobber is False.
    """
    check_not_none(filename, msg="Must specify filename")

    return_val = filename
    if os.path.exists(return_val) and not clobber:
        if raise_on_error:
            raise ValueError(f"{return_val} exists and not clobbering!")
        else:
            return_val = f"{return_val}_{get_now_dttmstr_local(fmt='%Y%m%d.%H%M%S')}"
            if os.path.exists(return_val):
                raise ValueError(f"{return_val} exists despite adding timestamp!")

    return return_val


def read_yaml(filename) -> dict:
    with open(filename) as fh:
        retval = yaml.safe_load(fh)
    return retval


## Value, Parameter, and List manipulation functions
def strip_nones(d: dict) -> dict:
    if d is None:
        return {}

    return {k: v for k, v in d.items() if v is not None}


def check_not_none(
    val: Union[Any, Iterable] = None,
    require_all: bool = True,
    raise_on_fail: bool = True,
    msg: str = None,
) -> bool:
    """Checks that a value (or collection of values) is not None/NaN.

    Behavior by type:
        - None: fails
        - list/tuple: checks each element for None
        - dict: checks each *value* for None
        - pd.Series: checks for None/NaN entries
        - pd.DataFrame: checks for None/NaN entries
        - anything else (str, int, etc.): passes if not None

    Args:
        val (Union[Any, Iterable], required): Single value or collection to check.
        require_all (bool, optional): If True, fails if *any* element is
            None/NaN. If False, fails only if *all* elements are None/NaN.
            Defaults to True.
        raise_on_fail (bool, optional): If True, will raise a ValueError on fail
            condition. If False, will return the condition as a bool.
        msg (str, optional): Message to raise as ValueError. Defaults to
            "Missing value!".

    Raises:
        ValueError: If it fails and `raise_on_fail`==True
    """
    import pandas as pd

    if val is None:
        ok = False
    elif isinstance(val, (list, tuple)):
        if require_all:
            ok = all(v is not None for v in val)
        else:
            ok = any(v is not None for v in val)
    elif isinstance(val, dict):
        if require_all:
            ok = all(v is not None for v in val.values())
        else:
            ok = any(v is not None for v in val.values())
    elif isinstance(val, pd.Series):
        if require_all:
            ok = not val.isna().any()
        else:
            ok = val.notna().any()
    elif isinstance(val, pd.DataFrame):
        if require_all:
            ok = not val.isna().any().any()
        else:
            ok = val.notna().any().any()
    else:
        ok = True

    if not ok and raise_on_fail:
        if msg is None:
            msg = "Missing value!"
        raise ValueError(msg)

    return ok


def get_val_or_alt_or_raise(val: Any, alt: Any = None, raise_error: bool = True):
    """Convenience method typically used to provide a fallback argument
    to a method in case the first choice is None.

    Args:
        val (Any): The item to check.
        alt (Any, optional): The fallback alternative. Defaults to None.
        raise_error(bool, optional): If both `val` and `alt` are None,
            it will raise a ValueError

    Raises:
        ValueError: If both `val` and `arg` are None.

    Returns:
        Any: The `val` is not None, otherwise the `alt`.
    """
    if val is not None:
        return val
    elif alt is not None:
        return alt
    elif raise_error:
        raise ValueError("Missing value, and no default specifed!")


def set_val_if_none(thing: Any, val: Any):
    """If thing is None, set it to val"""
    if thing is None:
        return val
    else:
        return thing


def listify(
    thing: Union[object, Iterable],
    stretch_length: int = 0,
    make_strings: bool = False,
) -> list:
    """Forces the input to be returned as a Python list. Accepts single
    values of strings or numbers, tuples, lists, numpy arrays, or
    anything that has a tolist() method.

    Args:
        thing (multiple types): Single item or list-like that will be
            converted into a Python list.
        stretch_length (int, optional): If the stretch_length is larger
            than the length of the provided `thing`, the resulting list
            will be extended to a new length equal to `stretch_length`
            by repeating the last element of the input `thing`.
        make_strings (bool, optional): If True, every item in the list
            will be converted to str. Defaults to False.
    """
    result: List[object] = []

    if isinstance(thing, Iterable) and not isinstance(thing, str):
        result = list(thing)
    elif thing is not None:
        result = [thing]

    if make_strings:
        result = [str(x) for x in result]

    if thing is not None and stretch_length > len(result):
        result.extend([result[-1]] * (stretch_length - len(result)))

    return result


## Performance monitoring functions
def check_mem(thresh: Optional[float] = None, frac: float = 0.9) -> int:
    """Cross-platform memory usage checker that can be used to terminate or
    modify the current program if a threshold is exceeded.

    Args:
        thresh (float, optional): Absolute used memory threshold in MB above
            which to raise an exception. If None, the method will default
            to the total memory times the `frac` parameter.
        frac (float, optional): If no absolute threshold is provided, this
            controls the fraction of total memory available above which this
            method will raise an exception. Defaults to 0.9.

    Returns:
        int: The current amount of used memory in MB.

    Raises:
        ValueError: If used memory exceeds the specified / calculated threshold
    """
    import platform

    system = platform.system()

    if system == "Linux":
        meminfo = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                meminfo[parts[0].rstrip(":")] = int(parts[1])
        total_memory = meminfo["MemTotal"] // 1024
        available = meminfo.get("MemAvailable", meminfo.get("MemFree", 0)) // 1024
        used_memory = total_memory - available

    elif system == "Windows":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        total_memory = stat.ullTotalPhys // (1024 * 1024)
        used_memory = total_memory - stat.ullAvailPhys // (1024 * 1024)

    elif system == "Darwin":
        import subprocess

        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True
        )
        total_memory = int(result.stdout.strip()) // (1024 * 1024)
        result = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
        # vm_stat reports in pages (typically 4096 bytes)
        pages_free = 0
        for line in result.stdout.splitlines():
            if "Pages free" in line:
                pages_free += int(line.split()[-1].rstrip("."))
            elif "Pages inactive" in line:
                pages_free += int(line.split()[-1].rstrip("."))
        used_memory = total_memory - (pages_free * 4096) // (1024 * 1024)

    else:
        raise NotImplementedError(f"check_mem not supported on {system}")

    if thresh is None:
        thresh = total_memory * frac

    if used_memory > thresh:
        raise ValueError(f"Mem usage too high! ({used_memory} of {total_memory})")

    return used_memory
