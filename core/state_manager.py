import json
import os
import tempfile
import threading
import time
from datetime import datetime

try:
    import portalocker
except ImportError:
    portalocker = None

STATE_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "nexus_state.json")
LOCK_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "nexus_state.lock")

class StateManager:
    def __init__(self, state_file=None, lock_file=None):
        self.lock = threading.RLock()
        self.state_file_path = state_file or STATE_FILE_PATH
        self.lock_file_path = lock_file or LOCK_FILE_PATH
        self.state = {}
        self.load()

    def set_instance(self, instance_tag=None):
        """
        Dynamically scopes state to a specific terminal instance (e.g. 't1', 't2').
        """
        with self.lock:
            data_dir = os.path.dirname(STATE_FILE_PATH)
            if instance_tag:
                self.state_file_path = os.path.join(data_dir, f"nexus_state_{instance_tag}.json")
                self.lock_file_path = os.path.join(data_dir, f"nexus_state_{instance_tag}.lock")
            else:
                self.state_file_path = STATE_FILE_PATH
                self.lock_file_path = LOCK_FILE_PATH
            self.load()

    def _acquire_file_lock(self, timeout=3.0):
        if not portalocker:
            return None
        dir_name = os.path.dirname(self.lock_file_path)
        os.makedirs(dir_name, exist_ok=True)
        try:
            return portalocker.Lock(self.lock_file_path, mode='a', timeout=timeout, flags=portalocker.LOCK_EX)
        except Exception:
            return None

    def load(self):
        with self.lock:
            if os.path.exists(self.state_file_path):
                for attempt in range(10):
                    try:
                        lock = self._acquire_file_lock()
                        if lock:
                            with lock:
                                with open(self.state_file_path, 'r') as f:
                                    self.state = json.load(f)
                        else:
                            with open(self.state_file_path, 'r') as f:
                                self.state = json.load(f)
                        break
                    except Exception as e:
                        if attempt == 9:
                            print(f"[STATE] ⚠️ Error loading persistent memory: {e}")
                            self.state = {}
                        time.sleep(0.05)
            else:
                self.state = {}

    def save(self):
        with self.lock:
            try:
                dir_name = os.path.dirname(self.state_file_path)
                os.makedirs(dir_name, exist_ok=True)
                fd, temp_path = tempfile.mkstemp(dir=dir_name, prefix="state_", suffix=".tmp")
                try:
                    with os.fdopen(fd, 'w') as f:
                        json.dump(self.state, f, indent=4)
                        f.flush()
                        os.fsync(f.fileno())
                    
                    # Atomic replace with shared lockfile synchronization
                    lock = self._acquire_file_lock()
                    if lock:
                        with lock:
                            for attempt in range(10):
                                try:
                                    os.replace(temp_path, self.state_file_path)
                                    break
                                except OSError as replace_err:
                                    if attempt == 9:
                                        raise replace_err
                                    time.sleep(0.05)
                    else:
                        for attempt in range(10):
                            try:
                                os.replace(temp_path, self.state_file_path)
                                break
                            except OSError as replace_err:
                                if attempt == 9:
                                    raise replace_err
                                time.sleep(0.05)
                except Exception as e:
                    if os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except Exception:
                            pass
                    raise e
            except Exception as e:
                print(f"[STATE] ⚠️ Error saving persistent memory: {e}")

    def get(self, key, default=None):
        with self.lock:
            return self.state.get(key, default)

    def set(self, key, value):
        with self.lock:
            if self.state.get(key) == value:
                return
            self.state[key] = value
            self.save()

# Global instance for easy import across modules
nexus_state = StateManager()
