from collections import defaultdict
from enum import Enum
from threading import Lock


class WorkStatus(Enum):
    """Work status."""
    DOING = 0
    DONE = 1


class HandleManager:
    """A thread-safe manager for handles.

    This class is used to manage handles for the distributed training.
    """

    def __init__(self):
        # int -> WorkStatus
        self._last_handle = 0
        self._handles = defaultdict(lambda: WorkStatus.DOING)
        self._lock = Lock()

    def allocate_handle(self):
        """Allocate a handle.

        Returns:
            int: The handle.
        """
        with self._lock:
            self._last_handle += 1
            handle = self._last_handle
            self._handles[handle] = WorkStatus.DOING
            return handle

    def mark_done(self, handle):
        """Mark a handle as done.

        Args:
            handle (int): The handle.
        """
        with self._lock:
            self._handles[handle] = WorkStatus.DONE

    def poll(self, handle):
        """Poll a handle.

        Args:
            handle (int): The handle.

        Returns:
            bool: True if the handle is done, False otherwise.
        """
        with self._lock:
            return self._handles[handle] == WorkStatus.DONE
