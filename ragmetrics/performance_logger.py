import atexit
import json
from datetime import datetime
from pathlib import Path
from threading import Lock


class PerformanceLogger:
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__new__(cls)
                cls._instance.metrics_log = []
                atexit.register(cls._instance.save_to_file)
        return cls._instance

    @staticmethod
    def find_project_root(marker_files=("pyproject.toml", "setup.py")):
        current_dir = Path(__file__).resolve().parent
        while current_dir != current_dir.root:
            if any((current_dir / marker).exists() for marker in marker_files):
                return current_dir
            current_dir = current_dir.parent
        return None

    def log(self, metric_name: str, function_name: str, value: str, unit: str):
        self.metrics_log.append(
            {
                "metric": metric_name,
                "function_name": function_name,
                "value": value,
                "unit": unit,
            }
        )

    def save_to_file(self):
        filename = (
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_performance_log.json"
        )
        project_root = self.find_project_root()

        if project_root:
            filepath = project_root / filename
            with open(filepath, "w") as f:
                json.dump(self.metrics_log, f, indent=4, ensure_ascii=False)
            print(f"Performance log saved to {filename}")
