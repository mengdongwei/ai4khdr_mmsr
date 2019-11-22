import numpy as np
import queue
from multiprocessing import Process, Queue, Event
import torch
from data.sampler import SequentialSampler, RandomSampler, BatchSampler
from data.worker import _worker_loop


class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

def mixup(lrs, hrs, lrx2, alpha=2):
    lam = np.random.beta(alpha, alpha)
    index = np.random.randint(0, lrs.shape[0])
    lrs = lrs * lam + (1 - lam) * lrs[index]
    hrs = hrs * lam + (1 - lam) * hrs[index]
    lrx2 = lrx2 * lam + (1 - lam) * lrx2[index]
    return lrs, hrs, lrx2

def mixup_enhance(lrs, hrs, lrx2, alpha=2):
    lam = np.random.beta(alpha, alpha)
    assert (lrs.shape[0] % 2) == 0
    lrs[0::2] = lrs[0::2] * lam + (1 - lam) * lrs[1::2]
    hrs[0::2] = hrs[0::2] * lam + (1 - lam) * hrs[1::2]
    lrx2[0::2] = lrx2[0::2] * lam + (1 - lam) * lrx2[1::2]
    return lrs, hrs, lrx2

def CollateFn(input_list):
    lrs, hrs, lrx2 = [], [], []
    for input in input_list:
        lrs.append(input[0])
        hrs.append(input[1])
        lrx2.append(input[2])

    lrs = np.array(lrs)
    hrs = np.array(hrs)
    lrx2 = np.array(lrx2)
    if lrs.shape[0] > 0 and (lrs.shape[0] % 2) == 0:
        lrs, hrs, lrx2 = mixup_enhance(lrs, hrs, lrx2)
    data = {}
    data['LQ'] = torch.tensor(lrs)
    data['GT'] = torch.tensor(hrs)
    data['LQX2'] = torch.tensor(lrx2)
    return data


class DataLoader(object):
    __initialized = False

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, collate_fn=CollateFn, drop_last=True, timeout=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.drop_last = drop_last
        self.timeout = timeout

        if timeout < 0:
            raise ValueError('timeout option should be non-negative')

        if self.num_workers < 0:
            raise ValueError('num_workers option cannot be negative; use num_workers=0 to disable multiprocessing.')

        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = RandomSampler(dataset)
                else:
                    sampler = SequentialSampler(dataset)
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.__initialized = True

    def __iter__(self):
        return _DataLoaderIter(self)

    def __len__(self):
        return len(self.batch_sampler)


class _DataLoaderIter(object):
    def __init__(self, loader):
        self.dataset = loader.dataset
        self.collate_fn = loader.collate_fn
        self.batch_sampler = loader.batch_sampler
        self.num_workers = loader.num_workers
        self.timeout = loader.timeout

        self.sample_iter = iter(self.batch_sampler)

        if self.num_workers > 0:
            self.worker_queue_idx = 0
            self.worker_result_queue = Queue()
            self.batches_outstanding = 0
            self.worker_pids_set = False
            self.shutdown = False
            self.send_idx = 0
            self.rcvd_idx = 0
            self.reorder_dict = {}
            self.done_event = Event()

            self.index_queues = []
            self.workers = []
            for i in range(self.num_workers):
                index_queue = Queue()
                index_queue.cancel_join_thread()
                w = Process(
                    target=_worker_loop,
                    args=(self.dataset, index_queue, self.worker_result_queue, self.done_event, self.collate_fn))
                w.daemon = True
                w.start()
                self.index_queues.append(index_queue)
                self.workers.append(w)

            self.data_queue = self.worker_result_queue

            # prime the prefetch loop
            for _ in range(2 * self.num_workers):
                self._put_indices()

    def __len__(self):
        return len(self.batch_sampler)

    def _try_get_batch(self, timeout=10):
        try:
            data = self.data_queue.get(timeout=timeout)
            return (True, data)
        except Exception as e:
            if not all(w.is_alive() for w in self.workers):
                pids_str = ', '.join(str(w.pid) for w in self.workers if not w.is_alive())
                raise RuntimeError('DataLoader worker (pid(s) {}) exited unexpectedly'.format(pids_str))
            if isinstance(e, queue.Empty):
                return (False, None)
            raise

    def _get_batch(self):
        if self.timeout > 0:
            sucess, data = self._try_get_batch(self.timeout)
            if sucess:
                return data
            else:
                raise RuntimeError('DataLoader timed out after {} seconds'.format(self.timeout))

        else:
            while True:
                success, data = self._try_get_batch()
                if success:
                    return data

    def __next__(self):
        if self.num_workers == 0:
            indices = next(self.sample_iter)
            batch = self.collate_fn([self.dataset[i] for i in indices])
            return batch

        if self.rcvd_idx in self.reorder_dict:
            batch = self.reorder_dict.pop(self.rcvd_idx)
            return self._process_next_batch(batch)

        if self.batches_outstanding == 0:
            self._shutdown_workers()
            raise StopIteration

        while True:
            assert (not self.shutdown and self.batches_outstanding > 0)
            idx, batch = self._get_batch()
            self.batches_outstanding -= 1
            if idx != self.rcvd_idx:
                self.reorder_dict[idx] = batch
                continue
            return self._process_next_batch(batch)

    def __iter__(self):
        return self

    def _put_indices(self):
        assert self.batches_outstanding < 2 * self.num_workers
        indices = next(self.sample_iter, None)
        if indices is None:
            return
        self.index_queues[self.worker_queue_idx].put((self.send_idx, indices))
        self.worker_queue_idx = (self.worker_queue_idx + 1) % self.num_workers
        self.batches_outstanding += 1
        self.send_idx += 1

    def _process_next_batch(self, batch):
        self.rcvd_idx += 1
        self._put_indices()
        return batch

    def _shutdown_workers(self):
        if not self.shutdown:
            self.shutdown = True
            try:
                self.done_event.set()

                self.worker_result_queue.put(None)
                self.worker_result_queue.close()

                for q in self.index_queues:
                    q.put(None)
                    q.close()

                for w in self.workers:
                    w.join()
            finally:
                pass

    def __del__(self):
        if self.num_workers > 0:
            self._shutdown_workers()

if __name__ == '__main__':
    lrs = np.zeros((32, 128, 128, 1))
    hrs = np.zeros((32, 256, 256, 1))
    mixup_enhance(lrs, hrs)
