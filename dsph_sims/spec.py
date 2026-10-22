"""The one description of a simulator family, consumed by every pipeline.

A `ModelSpec` says what a dataset's galaxies were drawn from and how:
the sampling-space parameters and their uniform box, the physics that
turns a parameter vector into stars, the acceptance cuts, the columns
stored per galaxy (the NPE picks its labels from these), and -- when
the family supports it -- how tsnpe's truncated-proposal box maps onto
the model's units and back onto the simulator (`TsnpePrior`).

dsph_sims's batch runner, npe/train_npe.py and tsnpe all read the same
object, so a bound, a unit convention or a cut can only be changed in
one place. Adding a model means adding one spec module under `models/`
and registering it in `models/__init__.py`.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

# simulate(num_stars, params[, rng]) -> (posvel, extras), None, or a str.
# `extras` carries per-galaxy scalars that are not parameters but are
# stored as columns (e.g. the projected half-light radius of a non-Plummer
# tracer). A str is a rejection with a reason; the runner counts reasons
# into config.<i>.json, since rejections shape the effective prior. `rng`
# is passed only when `ModelSpec.seeded` is True.
SimulateFn = Callable[..., 'tuple[np.ndarray, dict] | None | str']
PreprocessFn = Callable[[np.ndarray, np.ndarray], 'np.ndarray | None']
ColumnsFn = Callable[[np.ndarray, dict], 'dict[str, float]']


@dataclass(frozen=True)
class TsnpePrior:
    """How tsnpe sees a spec: its box, the model's units, the way back.

    tsnpe draws its proposal uniformly in *box space* -- the inferred
    parameters with every radius relative to r_star -- because that is
    the only space in which the training prior is a box for all of the
    8-parameter family, then conditions on one axis, log10(r_half/kpc),
    taken from the target rather than drawn.

    Attributes:
        param_names: Inferred parameters (the NN's labels), model order.
        prior_min: Box-space lower bounds keyed by parameter name.
        prior_max: Upper bounds, same keys.
        kpc_offset_names: Parameters the model predicts as log10(r/kpc)
            although the box holds them as log10(r/r_star); tsnpe adds
            the conditioning value on the way to the model and
            subtracts it on the way back. Empty when the model itself
            works in r_star units.
        cond_columns: Stored dataset columns whose value equals tsnpe's
            conditioning axis, so a checkpoint's `cond_labels` can be
            checked against the spec.
        model_to_sim: Maps (N, len(param_names) + 1) rows in model
            units, conditioning last, to (N, len(ModelSpec.param_names))
            sampling-space rows the family's `simulate`/`preprocess`
            take. Must be a module-level function (workers unpickle it).
        cond_name: Name tsnpe gives the conditioning axis.
    """

    param_names: tuple[str, ...]
    prior_min: dict[str, float]
    prior_max: dict[str, float]
    kpc_offset_names: tuple[str, ...]
    cond_columns: tuple[str, ...]
    model_to_sim: Callable[[np.ndarray], np.ndarray]
    cond_name: str = 'stellar_log_rstar'

    def to_dict(self, model_spec: str) -> dict:
        """The pinned form `tsnpe.prior.Prior.from_dict` rebuilds from.

        Args:
            model_spec: The owning `ModelSpec.name`, recorded for
                provenance and for tsnpe to find the physics again.

        Returns:
            JSON-serializable dict.
        """
        return {
            'model_spec': model_spec,
            'param_names': list(self.param_names),
            'cond_name': self.cond_name,
            'prior_min': {k: float(v) for k, v in self.prior_min.items()},
            'prior_max': {k: float(v) for k, v in self.prior_max.items()},
            'kpc_offset_names': list(self.kpc_offset_names),
        }


@dataclass(frozen=True)
class ModelSpec:
    """One simulator family, or one prior variant of it.

    Attributes:
        name: Registry key, e.g. '8p_ZhaoPlumCOM_v4'.
        description: One line on what distinguishes this spec.
        param_names: Sampling-space parameters in the order `simulate`,
            `preprocess`, `constraint` and the batch runner use.
        prior_min: Lower bounds of the uniform sampling box, same order.
        prior_max: Upper bounds, same order.
        column_names: Graph-level columns stored per galaxy, file order.
            `columns` must return exactly these keys.
        labels: The NPE labels this family is trained with by default.
        cond_labels: The NPE conditioning columns, likewise.
        datasets: Dataset directory names produced by this spec, so a
            checkpoint that only records `data_name` can be mapped back.
        simulate: See `SimulateFn`. Module-level function or partial.
        preprocess: Acceptance cuts on a simulated galaxy; None rejects.
        columns: Stored scalars from (sampling-space params, extras).
        constraint: Joint prior constraint the box does not express;
            draws violating it are redrawn. None when the prior is a box.
        constraint_text: Human-readable `constraint`, recorded in the
            dataset config.
        tsnpe: How tsnpe uses this spec, or None if it cannot yet
            (multi-dimensional conditioning, non-Plummer r_half).
        n_stars: Default mean number of stars per galaxy.
        min_stars: Floor on the Poisson star count (0 = none).
        seeded: True if `simulate` draws its own random numbers from a
            NumPy generator (DF-free simulators) rather than through
            agama. The runner then draws one seed per galaxy after the
            star count and passes `rng=np.random.default_rng(seed)`.

    Raises:
        ValueError: On inconsistent lengths or unknown column names.
    """

    name: str
    description: str
    param_names: tuple[str, ...]
    prior_min: tuple[float, ...]
    prior_max: tuple[float, ...]
    column_names: tuple[str, ...]
    labels: tuple[str, ...]
    cond_labels: tuple[str, ...]
    datasets: tuple[str, ...]
    simulate: SimulateFn
    preprocess: PreprocessFn
    columns: ColumnsFn
    constraint: 'Callable[[np.ndarray], bool] | None' = None
    constraint_text: 'str | None' = None
    tsnpe: 'TsnpePrior | None' = None
    n_stars: int = 100
    min_stars: int = 0
    seeded: bool = False

    def __post_init__(self):
        n = len(self.param_names)
        if len(self.prior_min) != n or len(self.prior_max) != n:
            raise ValueError(
                f'{self.name}: prior bounds must have {n} entries')
        bad = [p for p, lo, hi in zip(
            self.param_names, self.prior_min, self.prior_max) if lo >= hi]
        if bad:
            raise ValueError(f'{self.name}: prior_min >= prior_max for {bad}')
        for attr in ('labels', 'cond_labels'):
            unknown = set(getattr(self, attr)) - set(self.column_names)
            if unknown:
                raise ValueError(
                    f'{self.name}: {attr} {sorted(unknown)} not in '
                    'column_names')
        if self.tsnpe is not None:
            unknown = set(self.tsnpe.param_names) - set(self.column_names)
            unknown |= set(self.tsnpe.cond_columns) - set(self.column_names)
            if unknown:
                raise ValueError(
                    f'{self.name}: tsnpe names {sorted(unknown)} not in '
                    'column_names')
        if (self.constraint is None) != (self.constraint_text is None):
            raise ValueError(
                f'{self.name}: constraint and constraint_text go together')

    def draw(self, rng: np.random.Generator, n_stars: int
             ) -> tuple[np.ndarray, int]:
        """One prior draw and its star count.

        Same RNG consumption as the original scripts -- uniform box draws
        until `constraint` admits one, then one Poisson draw (floored at
        `min_stars`) -- so a dataset is reproducible from the seed in its
        config. A seeded spec's per-galaxy seed is drawn by the caller
        right after this, again as the scripts did.

        Args:
            rng: Generator to draw from.
            n_stars: Mean number of stars (Poisson), or a (lo, hi) pair
                for a count drawn uniformly over the closed range.

        Returns:
            (params, num_stars) in sampling-space order.
        """
        lo = np.asarray(self.prior_min)
        hi = np.asarray(self.prior_max)
        while True:
            params = rng.uniform(lo, hi)
            if self.constraint is None or self.constraint(params):
                break
        if np.ndim(n_stars) == 1:
            # Reason: a (lo, hi) pair draws the count uniformly over the
            # closed range instead of Poisson about one mean, so one
            # dataset spans e.g. 100-1000 stars per galaxy.
            lo_n, hi_n = (int(v) for v in n_stars)
            return params, max(int(rng.integers(lo_n, hi_n + 1)),
                               self.min_stars)
        return params, max(int(rng.poisson(n_stars)), self.min_stars)
