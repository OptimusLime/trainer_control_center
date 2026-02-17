"""TaskRegistry — discovers Task subclasses from acc/tasks/.

Same pattern as RecipeRegistry. Scans .py files for Task subclasses.
Supports hot-reload via importlib.reload + file watcher.

Foundation: generalizes the RecipeRegistry pattern. M3 will clone this
again for generators, proving the pattern works for all registries.
"""

import importlib
import importlib.util
import os
import sys
import threading
from typing import Optional

from acc.tasks.base import Task


# Files to skip when scanning for task subclasses
_SKIP_FILES = frozenset({"base.py", "registry.py"})


class TaskRegistry:
    """Auto-discovers Task subclasses from a directory.

    Unlike RecipeRegistry (which returns instances), this returns classes —
    tasks need constructor args (name, dataset, weight, etc.) so the caller
    must instantiate them.

    Optionally watches the directory for changes and reloads automatically.
    """

    def __init__(self, tasks_dir: str = None):
        if tasks_dir is None:
            tasks_dir = os.path.join(os.path.dirname(__file__))
        self.tasks_dir = tasks_dir
        self._task_classes: dict[str, type[Task]] = {}
        self._file_mtimes: dict[str, float] = {}
        self._watcher_thread: Optional[threading.Thread] = None
        self._watcher_stop = threading.Event()
        self.scan()

    def scan(self) -> None:
        """Scan the tasks directory for Task subclasses."""
        for filename in os.listdir(self.tasks_dir):
            if not filename.endswith(".py"):
                continue
            if filename.startswith("_") or filename in _SKIP_FILES:
                continue
            filepath = os.path.join(self.tasks_dir, filename)
            self._file_mtimes[filename] = os.path.getmtime(filepath)
            self._load_module(filename)

    def _load_module(self, filename: str) -> None:
        """Load a single .py file and register any Task subclasses found."""
        module_name = f"acc.tasks.{filename[:-3]}"
        filepath = os.path.join(self.tasks_dir, filename)

        try:
            if module_name in sys.modules:
                module = importlib.reload(sys.modules[module_name])
            else:
                spec = importlib.util.spec_from_file_location(module_name, filepath)
                if spec is None or spec.loader is None:
                    return
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

            # Scan for Task subclasses
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, Task)
                    and attr is not Task
                ):
                    self._task_classes[attr.__name__] = attr

        except Exception as e:
            print(f"[TaskRegistry] Error loading {filename}: {e}")

    def reload(self) -> None:
        """Re-scan all task files."""
        self._task_classes.clear()
        self.scan()

    def list(self) -> list[dict]:
        """List available task classes as dicts with metadata."""
        result = []
        for class_name, cls in self._task_classes.items():
            info = {
                "class_name": class_name,
                "description": (cls.__doc__ or "").strip().split("\n")[0],
                "module": cls.__module__,
            }
            result.append(info)
        return result

    def get(self, class_name: str) -> Optional[type[Task]]:
        """Get a task class by name. Returns the class, not an instance."""
        return self._task_classes.get(class_name)

    # ─── File Watcher ───

    def start_watcher(self, poll_interval: float = 2.0) -> None:
        """Start a background thread that watches for task file changes.

        Polls every poll_interval seconds. When a .py file is added,
        removed, or modified, triggers a reload of that file.
        """
        if self._watcher_thread is not None and self._watcher_thread.is_alive():
            return

        self._watcher_stop.clear()

        def _watch():
            while not self._watcher_stop.is_set():
                self._watcher_stop.wait(poll_interval)
                if self._watcher_stop.is_set():
                    break
                self._check_for_changes()

        self._watcher_thread = threading.Thread(
            target=_watch, daemon=True, name="task-watcher"
        )
        self._watcher_thread.start()
        print("[TaskRegistry] File watcher started")

    def stop_watcher(self) -> None:
        """Stop the file watcher thread."""
        self._watcher_stop.set()
        if self._watcher_thread is not None:
            self._watcher_thread.join(timeout=5.0)
            self._watcher_thread = None
        print("[TaskRegistry] File watcher stopped")

    def _check_for_changes(self) -> None:
        """Check for new, modified, or deleted task files."""
        current_files = set()
        changed = False

        for filename in os.listdir(self.tasks_dir):
            if not filename.endswith(".py"):
                continue
            if filename.startswith("_") or filename in _SKIP_FILES:
                continue
            current_files.add(filename)
            filepath = os.path.join(self.tasks_dir, filename)
            mtime = os.path.getmtime(filepath)

            if filename not in self._file_mtimes:
                print(f"[TaskRegistry] New task file: {filename}")
                self._file_mtimes[filename] = mtime
                self._load_module(filename)
                changed = True
            elif mtime > self._file_mtimes[filename]:
                print(f"[TaskRegistry] Task file changed: {filename}")
                self._file_mtimes[filename] = mtime
                self._load_module(filename)
                changed = True

        # Check for deleted files
        old_files = set(self._file_mtimes.keys())
        for removed in old_files - current_files:
            print(f"[TaskRegistry] Task file removed: {removed}")
            del self._file_mtimes[removed]
            # Remove task classes that came from this module
            module_name = f"acc.tasks.{removed[:-3]}"
            self._task_classes = {
                name: cls
                for name, cls in self._task_classes.items()
                if cls.__module__ != module_name
            }
            changed = True
