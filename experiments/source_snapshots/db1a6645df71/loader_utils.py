"""Deterministic, spawn-safe CPU prefetch; no CUDA work in loader processes."""
import random
import numpy as np
import torch


def initialize_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def options(config, device):
    workers = config.get('loader_workers', 2) if device.type == 'cuda' else 0
    if type(workers) is not int or not 0 <= workers <= 16:
        raise ValueError('LOADER_WORKERS must be an integer between 0 and 16')
    if type(config.get('loader_prefetch', 2)) is not int or config.get('loader_prefetch', 2) < 1:
        raise ValueError('LOADER_PREFETCH must be a positive integer')
    result = dict(num_workers=workers, pin_memory=device.type == 'cuda')
    if workers:
        result.update(multiprocessing_context='spawn', persistent_workers=True,
                      prefetch_factor=config.get('loader_prefetch', 2), worker_init_fn=initialize_worker)
    return result
