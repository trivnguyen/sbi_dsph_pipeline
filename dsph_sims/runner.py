"""Batch simulation of one spec into HDF5 shards (ProcessPoolExecutor).

The runner the per-family scripts used to carry, made generic: draw from
`spec`, simulate in worker processes (each with isolated agama state, so
no locking), apply the spec's cuts, and flush accepted galaxies to
`data.<i>.h5` shards laid out for `jgnn.datasets.io.read_graph_dataset`.
Per-galaxy columns are whatever `spec.columns` returns, plus
num_stars/ptr/pos/vel; the run's arguments, the spec name and its box go
to `config.<i>.json`.
"""

import argparse
import datetime
import itertools
import json
import os
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from glob import glob
from typing import Iterator

import h5py
import numpy as np
from tqdm import tqdm

from .models import common
from .spec import ModelSpec


def parse_args(argv=None, description=None) -> argparse.Namespace:
    """The CLI every simulator script exposes (simulate.sbatch depends
    on --n-sims, --n-workers and --output-dir)."""
    parser = argparse.ArgumentParser(
        description=description or 'Simulate the 6D stellar kinematics '
        'of dwarf galaxies (ProcessPoolExecutor version).')
    parser.add_argument(
        '--n-sims', type=int, default=100_000,
        help='Number of galaxies to attempt to simulate.')
    parser.add_argument(
        '--n-workers', type=int, default=os.cpu_count(),
        help='Number of worker processes to use.')
    parser.add_argument(
        '--n-stars', type=int, default=None,
        help='Mean number of stars sampled per galaxy '
             '(default: the spec\'s).')
    parser.add_argument(
        '--n-stars-range', type=int, nargs=2, default=None,
        metavar=('LO', 'HI'),
        help='Draw the star count uniformly in [LO, HI] per galaxy '
             'instead of Poisson about --n-stars.')
    parser.add_argument(
        '--sample-threads', type=int, default=1,
        help='OpenMP threads each worker may use internally for agama. '
             'Keep n-workers x sample-threads within the allocation.')
    parser.add_argument(
        '--galaxies-per-file', type=int, default=100_000,
        help='Maximum number of successful galaxies per output file.')
    parser.add_argument(
        '--output-dir', type=str, default='./simdata',
        help='Directory to create (if needed) and save output files to.')
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for reproducibility.')
    parser.add_argument(
        '--max-pending', type=int, default=None,
        help='Maximum in-flight simulations (default: 4x n-workers). '
             'Lower it if simulations run out of memory.')
    parser.add_argument(
        '--append', action='store_true',
        help='Append to existing output files instead of overwriting. '
             'The output directory must already exist.')
    return parser.parse_args(argv)


def iter_params(
    spec: ModelSpec, n_sims: int, n_stars: int, rng: np.random.Generator,
) -> Iterator[tuple[np.ndarray, int, 'int | None']]:
    """Lazily draw (params, num_stars, seed) triples, memory flat in
    n_sims. `seed` is None unless the spec is seeded."""
    for _ in range(n_sims):
        params, num_stars = spec.draw(rng, n_stars)
        seed = int(rng.integers(0, 2 ** 32)) if spec.seeded else None
        yield params, num_stars, seed


def simulate_one(spec_name: str, params: np.ndarray, num_stars: int,
                 seed: 'int | None' = None):
    """Simulate + preprocess one galaxy in a worker process.

    Looks the spec up by name so nothing but a string crosses the
    process boundary.

    Returns:
        (params, posvel, extras) if accepted, else (None, None, reason)
        with reason a str: the simulator's own, 'simulate' for a bare
        None, or 'cuts' for the preprocess rejection.
    """
    from . import get_spec  # Reason: avoid the package/runner import cycle
    spec = get_spec(spec_name)
    if spec.seeded:
        out = spec.simulate(num_stars, params, np.random.default_rng(seed))
    else:
        out = spec.simulate(num_stars, params)
    if out is None or isinstance(out, str):
        return None, None, out or 'simulate'
    posvel, extras = out
    posvel = spec.preprocess(posvel, params)
    if posvel is None:
        return None, None, 'cuts'
    return params, posvel, extras


def save_simdata(
    spec: ModelSpec, theta_list, posvel_list, extras_list, file_path: str,
) -> None:
    """Write one shard in the layout read_graph_dataset expects."""
    rows = [spec.columns(theta, extras)
            for theta, extras in zip(theta_list, extras_list)]
    num_stars = [pv.shape[0] for pv in posvel_list]
    ptr = np.cumsum([0] + num_stars)
    pos = np.concatenate([pv[:, :3] for pv in posvel_list], axis=0)
    vel = np.concatenate([pv[:, 3:6] for pv in posvel_list], axis=0)
    graph_features = list(spec.column_names) + ['num_stars', 'ptr']
    with h5py.File(file_path, 'w') as f:
        for name in spec.column_names:
            f.create_dataset(name, data=np.array([r[name] for r in rows]))
        f.create_dataset('num_stars', data=num_stars)
        f.create_dataset('ptr', data=ptr)
        f.create_dataset('pos', data=pos)
        f.create_dataset('vel', data=vel)
        f.attrs['all_features'] = graph_features + ['pos', 'vel']
        f.attrs['node_features'] = ['pos', 'vel']
        f.attrs['graph_features'] = graph_features
        f.attrs['model_spec'] = spec.name


def save_config(args: argparse.Namespace, spec: ModelSpec, file_path: str,
                **counts: int) -> None:
    """Record the run: its arguments, the spec, the spec's box and, once
    the run is over, the outcome counts (n_success and one n_reject_<reason>
    per rejection reason) that define the effective prior."""
    config = vars(args).copy()
    config['model_spec'] = spec.name
    config['param_names'] = list(spec.param_names)
    config['prior_min'] = list(spec.prior_min)
    config['prior_max'] = list(spec.prior_max)
    if spec.constraint_text is not None:
        config['prior_constraint'] = spec.constraint_text
    config.update(counts)
    config['timestamp'] = datetime.datetime.now().isoformat()
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=2)


def _next_index(output_dir: str, stem: str, ext: str) -> int:
    existing = glob(os.path.join(output_dir, f'{stem}.*.{ext}'))
    indices = [int(p.split('.')[-2]) for p in existing
               if p.split('.')[-2].isdigit()]
    return max(indices, default=-1) + 1


def run(spec: ModelSpec, args: argparse.Namespace) -> None:
    """Simulate `args.n_sims` draws of `spec` into `args.output_dir`."""
    os.makedirs(args.output_dir, exist_ok=True)
    if args.seed is None:
        # record a concrete seed so an unseeded run can still be
        # reproduced from the saved config
        args.seed = int(np.random.SeedSequence().entropy)
    if args.n_stars_range is not None:
        args.n_stars = tuple(args.n_stars_range)
    elif args.n_stars is None:
        args.n_stars = spec.n_stars
    print(f'Spec: {spec.name} - {spec.description}')
    print(f'Using seed: {args.seed}')
    rng = np.random.default_rng(args.seed)
    param_stream = iter_params(spec, args.n_sims, args.n_stars, rng)

    if not args.append:
        file_idx = 0
        config_path = os.path.join(args.output_dir, 'config.0.json')
    else:
        if not os.path.exists(args.output_dir):
            raise ValueError(
                f'Output directory {args.output_dir} does not exist, '
                'cannot append.')
        file_idx = _next_index(args.output_dir, 'data', 'h5')
        config_idx = _next_index(args.output_dir, 'config', 'json')
        config_path = os.path.join(
            args.output_dir, f'config.{config_idx:d}.json')
    save_config(args, spec, config_path)

    theta_buffer, posvel_buffer, extras_buffer = [], [], []
    n_success = 0
    n_reject: Counter = Counter()

    def _flush() -> None:
        nonlocal file_idx
        if not theta_buffer:
            return
        file_path = os.path.join(args.output_dir, f'data.{file_idx:d}.h5')
        save_simdata(spec, theta_buffer, posvel_buffer, extras_buffer,
                     file_path)
        file_idx += 1
        theta_buffer.clear()
        posvel_buffer.clear()
        extras_buffer.clear()

    # cap the number of in-flight tasks so memory use stays bounded
    # no matter how large n_sims is
    max_pending = args.max_pending or args.n_workers * 4
    with ProcessPoolExecutor(
        max_workers=args.n_workers, initializer=common.init_worker,
        initargs=(args.sample_threads,),
    ) as pool:
        pending = {
            pool.submit(simulate_one, spec.name, p, n, s)
            for p, n, s in itertools.islice(param_stream, max_pending)
        }
        with tqdm(total=args.n_sims, desc='Simulating') as pbar:
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    pbar.update(1)
                    params, pv, extras = fut.result()
                    if pv is not None:
                        theta_buffer.append(params)
                        posvel_buffer.append(pv)
                        extras_buffer.append(extras)
                        n_success += 1
                        if len(theta_buffer) >= args.galaxies_per_file:
                            _flush()
                    else:
                        n_reject[extras] += 1
                    next_item = next(param_stream, None)
                    if next_item is not None:
                        p, n, s = next_item
                        pending.add(
                            pool.submit(simulate_one, spec.name, p, n, s))
    _flush()
    # the rejection counts define the effective prior, so keep them with
    # the config
    save_config(args, spec, config_path, n_success=n_success,
                **{f'n_reject_{k}': v for k, v in n_reject.items()})
    print(f'Successful simulations: {n_success} / {args.n_sims} '
          f'({n_success / args.n_sims * 100:.1f}%)')
    for reason, count in sorted(n_reject.items()):
        print(f'  rejected ({reason}): {count} '
              f'({count / args.n_sims * 100:.1f}%)')
    print(f'Saved {file_idx} file(s) to {args.output_dir}')


def main(spec: ModelSpec, argv=None) -> None:
    """CLI entry point shared by simulate_batch.py and the legacy
    per-family scripts."""
    t1 = time.time()
    run(spec, parse_args(argv, description=spec.description))
    print(f'Time taken: {time.time() - t1:.2f} seconds')
