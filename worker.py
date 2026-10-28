"""Generic per-worker context helper."""

import dataclasses

from numpy.random import SeedSequence, default_rng


@dataclasses.dataclass
class WorkerContext:
    """Per-worker RNG seed and optional experiment-specific data."""
    seed: object
    data: object = None
    _rng: object = dataclasses.field(default=None, init=False, repr=False)

    def get_seed_sequence(self):
        return self.seed

    def get_rng(self):
        if self._rng is None:
            self._rng = default_rng(self.seed)
        return self._rng

    @classmethod
    def spawn(cls, n_workers, worker_data=None):
        """Create independent worker contexts with RNG seeds."""
        if worker_data is None:
            worker_data = [None] * n_workers
        if len(worker_data) != n_workers:
            raise ValueError(
                'The number of worker_data entries must equal n_workers')
        return list(map(cls, SeedSequence().spawn(n_workers), worker_data))
