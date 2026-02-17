"""RecipeRegistry — discovers Recipe subclasses from acc/recipes/.

Same pattern as task/generator discovery. Scans .py files for Recipe
subclasses. Supports hot-reload via importlib.reload + file watcher.
"""

import importlib
import importlib.util
import os
import sys
import threading
import time
from typing import Optional

from acc.recipes.base import Recipe


class RecipeRegistry:
    """Auto-discovers Recipe subclasses from a directory.

    Optionally watches the directory for changes and reloads automatically.
    """

    def __init__(self, recipes_dir: str = None):
        if recipes_dir is None:
            recipes_dir = os.path.join(os.path.dirname(__file__))
        self.recipes_dir = recipes_dir
        self._recipes: dict[str, type[Recipe]] = {}
        self._file_mtimes: dict[str, float] = {}
        self._watcher_thread: Optional[threading.Thread] = None
        self._watcher_stop = threading.Event()
        self.scan()

    def scan(self) -> None:
        """Scan the recipes directory for Recipe subclasses."""
        for filename in os.listdir(self.recipes_dir):
            if not filename.endswith(".py"):
                continue
            if filename.startswith("_") or filename in ("base.py", "runner.py", "registry.py"):
                continue
            filepath = os.path.join(self.recipes_dir, filename)
            self._file_mtimes[filename] = os.path.getmtime(filepath)
            self._load_module(filename)

    def _load_module(self, filename: str) -> None:
        """Load a single .py file and register any Recipe subclasses found."""
        module_name = f"acc.recipes.{filename[:-3]}"
        filepath = os.path.join(self.recipes_dir, filename)

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

            # Scan for Recipe subclasses
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, Recipe)
                    and attr is not Recipe
                    and hasattr(attr, "name")
                ):
                    self._recipes[attr.name] = attr

        except Exception as e:
            print(f"[RecipeRegistry] Error loading {filename}: {e}")

    def reload(self) -> None:
        """Re-scan all recipe files."""
        self._recipes.clear()
        self.scan()

    def list(self) -> list[dict]:
        """List available recipes as dicts."""
        return [
            {"name": cls.name, "description": cls.description}
            for cls in self._recipes.values()
        ]

    def get(self, name: str) -> Optional[Recipe]:
        """Get a recipe instance by name."""
        cls = self._recipes.get(name)
        if cls is None:
            return None
        return cls()

    # ─── File Watcher ───

    def start_watcher(self, poll_interval: float = 2.0) -> None:
        """Start a background thread that watches for recipe file changes.

        Polls every `poll_interval` seconds. When a .py file is added,
        removed, or modified, triggers a reload of that file.
        """
        if self._watcher_thread is not None and self._watcher_thread.is_alive():
            return  # Already running

        self._watcher_stop.clear()

        def _watch():
            while not self._watcher_stop.is_set():
                self._watcher_stop.wait(poll_interval)
                if self._watcher_stop.is_set():
                    break
                self._check_for_changes()

        self._watcher_thread = threading.Thread(
            target=_watch, daemon=True, name="recipe-watcher"
        )
        self._watcher_thread.start()
        print("[RecipeRegistry] File watcher started")

    def stop_watcher(self) -> None:
        """Stop the file watcher thread."""
        self._watcher_stop.set()
        if self._watcher_thread is not None:
            self._watcher_thread.join(timeout=5.0)
            self._watcher_thread = None
        print("[RecipeRegistry] File watcher stopped")

    def _check_for_changes(self) -> None:
        """Check for new, modified, or deleted recipe files."""
        current_files = set()
        changed = False

        for filename in os.listdir(self.recipes_dir):
            if not filename.endswith(".py"):
                continue
            if filename.startswith("_") or filename in ("base.py", "runner.py", "registry.py"):
                continue
            current_files.add(filename)
            filepath = os.path.join(self.recipes_dir, filename)
            mtime = os.path.getmtime(filepath)

            if filename not in self._file_mtimes:
                # New file
                print(f"[RecipeRegistry] New recipe file: {filename}")
                self._file_mtimes[filename] = mtime
                self._load_module(filename)
                changed = True
            elif mtime > self._file_mtimes[filename]:
                # Modified file
                print(f"[RecipeRegistry] Recipe file changed: {filename}")
                self._file_mtimes[filename] = mtime
                self._load_module(filename)
                changed = True

        # Check for deleted files
        old_files = set(self._file_mtimes.keys())
        for removed in old_files - current_files:
            print(f"[RecipeRegistry] Recipe file removed: {removed}")
            del self._file_mtimes[removed]
            # Remove recipes that came from this module
            module_name = f"acc.recipes.{removed[:-3]}"
            self._recipes = {
                name: cls
                for name, cls in self._recipes.items()
                if cls.__module__ != module_name
            }
            changed = True
