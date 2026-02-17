"""GeneratorRegistry — discovers DatasetGenerator subclasses from acc/generators/.

Same pattern as TaskRegistry and RecipeRegistry. Scans .py files for
DatasetGenerator subclasses. Supports hot-reload via importlib.reload + file watcher.
"""

import importlib
import importlib.util
import os
import sys
import threading
from typing import Optional

from acc.generators.base import DatasetGenerator


# Files to skip when scanning for generator subclasses
_SKIP_FILES = frozenset({"base.py", "registry.py"})


class GeneratorRegistry:
    """Auto-discovers DatasetGenerator subclasses from a directory.

    Returns instances (unlike TaskRegistry which returns classes) because
    generators are stateless — they just need generate(**params).
    """

    def __init__(self, generators_dir: str = None):
        if generators_dir is None:
            generators_dir = os.path.join(os.path.dirname(__file__))
        self.generators_dir = generators_dir
        self._generators: dict[str, type[DatasetGenerator]] = {}
        self._file_mtimes: dict[str, float] = {}
        self._watcher_thread: Optional[threading.Thread] = None
        self._watcher_stop = threading.Event()
        self.scan()

    def scan(self) -> None:
        """Scan the generators directory for DatasetGenerator subclasses."""
        for filename in os.listdir(self.generators_dir):
            if not filename.endswith(".py"):
                continue
            if filename.startswith("_") or filename in _SKIP_FILES:
                continue
            filepath = os.path.join(self.generators_dir, filename)
            self._file_mtimes[filename] = os.path.getmtime(filepath)
            self._load_module(filename)

    def _load_module(self, filename: str) -> None:
        """Load a single .py file and register any DatasetGenerator subclasses found."""
        module_name = f"acc.generators.{filename[:-3]}"
        filepath = os.path.join(self.generators_dir, filename)

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

            # Scan for DatasetGenerator subclasses
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, DatasetGenerator)
                    and attr is not DatasetGenerator
                    and hasattr(attr, "name")
                    and attr.name != "unnamed"
                ):
                    self._generators[attr.name] = attr

        except Exception as e:
            print(f"[GeneratorRegistry] Error loading {filename}: {e}")

    def reload(self) -> None:
        """Re-scan all generator files."""
        self._generators.clear()
        self.scan()

    def list(self) -> list[dict]:
        """List available generators as dicts with metadata."""
        result = []
        for name, cls in self._generators.items():
            gen = cls()
            result.append(gen.describe())
        return result

    def get(self, name: str) -> Optional[DatasetGenerator]:
        """Get a generator instance by name."""
        cls = self._generators.get(name)
        if cls is None:
            return None
        return cls()

    # ─── File Watcher ───

    def start_watcher(self, poll_interval: float = 2.0) -> None:
        """Start a background thread that watches for generator file changes."""
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
            target=_watch, daemon=True, name="generator-watcher"
        )
        self._watcher_thread.start()
        print("[GeneratorRegistry] File watcher started")

    def stop_watcher(self) -> None:
        """Stop the file watcher thread."""
        self._watcher_stop.set()
        if self._watcher_thread is not None:
            self._watcher_thread.join(timeout=5.0)
            self._watcher_thread = None
        print("[GeneratorRegistry] File watcher stopped")

    def _check_for_changes(self) -> None:
        """Check for new, modified, or deleted generator files."""
        current_files = set()
        changed = False

        for filename in os.listdir(self.generators_dir):
            if not filename.endswith(".py"):
                continue
            if filename.startswith("_") or filename in _SKIP_FILES:
                continue
            current_files.add(filename)
            filepath = os.path.join(self.generators_dir, filename)
            mtime = os.path.getmtime(filepath)

            if filename not in self._file_mtimes:
                print(f"[GeneratorRegistry] New generator file: {filename}")
                self._file_mtimes[filename] = mtime
                self._load_module(filename)
                changed = True
            elif mtime > self._file_mtimes[filename]:
                print(f"[GeneratorRegistry] Generator file changed: {filename}")
                self._file_mtimes[filename] = mtime
                self._load_module(filename)
                changed = True

        # Check for deleted files
        old_files = set(self._file_mtimes.keys())
        for removed in old_files - current_files:
            print(f"[GeneratorRegistry] Generator file removed: {removed}")
            del self._file_mtimes[removed]
            module_name = f"acc.generators.{removed[:-3]}"
            self._generators = {
                name: cls
                for name, cls in self._generators.items()
                if cls.__module__ != module_name
            }
            changed = True
