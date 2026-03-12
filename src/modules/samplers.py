import os, random
from torch.utils.data import IterableDataset, get_worker_info

class netFoundLengthBucketedIterable(IterableDataset):
    def __init__(self, base, batch_size, buffer_size=2048,
                 seed=42, drop_last=True):
        self.base = base
        self.batch_size = int(batch_size)
        self.buffer_size = int(buffer_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)

    def __iter__(self):
        wi = get_worker_info()
        ds = self.base

        # --- shard per worker to avoid duplicates ---
        if wi is not None and hasattr(ds, "shard"):
            # each worker takes a disjoint subset of shards/examples
            ds = ds.shard(num_shards=wi.num_workers, index=wi.id)

        # RNG: include rank + worker to keep streams different
        rank = int(os.environ.get("RANK", "0"))
        worker_id = wi.id if wi is not None else 0
        rng = random.Random(self.seed + 1009 * rank + 9176 * worker_id)

        buf = []
        for ex in iter(ds):
            buf.append(ex)
            if len(buf) >= self.buffer_size:
                yield from self._drain(buf, rng)
                buf = []

        if buf and not self.drop_last:
            yield from self._drain(buf, rng, allow_partial=True)

    def _drain(self, buf, rng, allow_partial=False):
        # logic here is this:
        # 1. We sort by number of bursts in the flow to group together flows with similar number of bursts as this will multiply padding by number of bursts (significantly)
        # 2. We further sort by the maximum burst size to group within the same number of bursts, as this will increase padding by maximum burst size (less significantly)
        # both of these infos are contained in 'dataset_burst_sizes' as the length (total number of bursts) and the max value (maximum burst size)
        sort_func = lambda x: (len(x['dataset_burst_sizes']), max(x['dataset_burst_sizes']))

        buf.sort(key=sort_func)
        batches = [buf[i:i+self.batch_size] for i in range(0, len(buf), self.batch_size)]
        if self.drop_last and not allow_partial and batches and len(batches[-1]) < self.batch_size:
            batches.pop()
        rng.shuffle(batches)
        for b in batches:
            for ex in b:
                yield ex
