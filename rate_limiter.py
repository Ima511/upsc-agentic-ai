import time
import threading

class RateLimiter:
    def __init__(self, max_calls_per_minute=10):
        self.interval = 60.0 / max_calls_per_minute
        self.lock = threading.Lock()
        self.last_call = 0.0

    def wait(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last_call = time.time()
