import os
import queue


class ManagerWatchdog(object):
    def __init__(self):
        self.manager_pid = os.getppid()
        self.manager_dead = False

    def is_alive(self):
        if not self.manager_dead:
            self.manager_dead = os.getppid() != self.manager_pid
        return not self.manager_dead


def _worker_loop(dataset, index_queue, data_queue, done_event, collate_fn):
    try:
        watchdog = ManagerWatchdog()

        while watchdog.is_alive():
            try:
                r = index_queue.get(timeout=10.0)
            except queue.Empty:
                continue
            if r is None:
                assert done_event.is_set()
                return
            elif done_event.is_set():
                continue

            idx, batch_indices = r
            try:
                samples = collate_fn([dataset[i] for i in batch_indices])
            except Exception:
                data_queue.put((idx, None))
            else:
                data_queue.put((idx, samples))
                del samples

    except KeyboardInterrupt:
        pass
