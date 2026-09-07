"""Prior box for the 8-parameter Zhao-Plummer dSph model.

Order: (dm_alpha, dm_beta, dm_gamma, dm_log_rdm, dm_log_rho0, df_beta0,
df_log_ra), plus stellar_log_rstar as a conditioning dimension.

The bounds and the radius convention both belong to the *training set*,
so they live on a `Prior` instance built from `config.prior` rather than
in module constants: a run whose proposal box disagrees with the data its
round-0 model saw is silently wrong, and there is now more than one
training set (priorA and priorB). `register_run.py` writes the resolved
prior into `round_0/prior_config.json` and every later round reads it
back, the same way norm_dict and the model architecture are pinned.

What does *not* vary is the parameter list itself, so PARAM_NAMES and
friends stay constants.

Radius units
------------
The box always draws dm_log_rdm as log10(r_dm / r_star) -- an offset from
the conditioning value -- because that is the only space in which the box
is a box: r_dm's kpc bounds slide with each row's own r_star draw. What
differs between training sets is the units the *model* predicts and
tsnpe/sims.py consumes, selected by `radius_units`:

  'kpc'   (priorA / 8p_ZhaoPlumCOM) - the model predicts
          log10(r_dm / kpc), so to_kpc() adds the conditioning value on
          the way out and to_rstar_units() subtracts it on the way back.

  'rstar' (priorB / 8p_ZhaoPlumCOM_v3) - the model predicts
          log10(r_dm / r_star), the same units the box is defined in.
          Nothing converts: `rstar_scaled_names` is empty and both
          functions become the identity.

df_log_ra is in r_star units under *both* conventions -- tsnpe/sims.py
multiplies it by r_star itself -- so it never takes part in either, and
must not be listed in `rstar_scaled_names` or the offset gets applied
twice.
"""

import numpy as np
import torch

from .target import TargetData

PARAM_NAMES = [
    'dm_alpha', 'dm_beta', 'dm_gamma', 'dm_log_rdm',
    'dm_log_rho0', 'df_beta0', 'df_log_ra',
]

CONDITIONING_NAME = 'stellar_log_rstar'
CONDITIONING_INDEX = len(PARAM_NAMES)  # conditioning dim is always appended last
ALL_PARAM_NAMES = PARAM_NAMES + [CONDITIONING_NAME]

RADIUS_UNITS_CHOICES = ('kpc', 'rstar')

# priorA / 8p_ZhaoPlumCOM, i.e. that set's config.<n>.json prior_min /
# prior_max with the conditioning column dropped. Used when a config
# gives no explicit bounds, so existing runs keep working untouched.
DEFAULT_PRIOR_MIN = {
    'dm_alpha': 0.5, 'dm_beta': 1.0, 'dm_gamma': -1.0, 'dm_log_rdm': 0.0,
    'dm_log_rho0': 3.0, 'df_beta0': -0.499, 'df_log_ra': -1.0,
}
DEFAULT_PRIOR_MAX = {
    'dm_alpha': 3.0, 'dm_beta': 10.0, 'dm_gamma': 2.0, 'dm_log_rdm': 3.0,
    'dm_log_rho0': 10.0, 'df_beta0': 1.0, 'df_log_ra': 3.0,
}


class Prior:
    """Prior box for one TSNPE run, pinned to its training set.

    Args:
        prior_min: Lower bounds, keyed by parameter name. None uses
            `DEFAULT_PRIOR_MIN` (priorA). Every name in `PARAM_NAMES`
            must be present -- a missing one is a config error, not
            something to fill in from a default, since a half-specified
            box is exactly the silent mismatch this class exists to stop.
        prior_max: Upper bounds, same rules.
        radius_units: 'kpc' or 'rstar'; see the module docstring.

    Raises:
        ValueError: If `radius_units` is unknown, if either bound dict
            names an unknown parameter or omits a known one, or if any
            lower bound is not strictly below its upper bound.
    """

    def __init__(self, prior_min=None, prior_max=None, radius_units='kpc'):
        if radius_units not in RADIUS_UNITS_CHOICES:
            raise ValueError(
                f'radius_units={radius_units!r} not recognized; must be '
                f'one of {RADIUS_UNITS_CHOICES}')
        self.radius_units = radius_units

        prior_min = dict(DEFAULT_PRIOR_MIN if prior_min is None else prior_min)
        prior_max = dict(DEFAULT_PRIOR_MAX if prior_max is None else prior_max)
        self._validate(prior_min, 'prior_min')
        self._validate(prior_max, 'prior_max')

        bad = [n for n in PARAM_NAMES if prior_min[n] >= prior_max[n]]
        if bad:
            raise ValueError(
                f'prior_min must be strictly below prior_max for every '
                f'parameter; violated by {bad}')

        self.prior_min_dict = prior_min
        self.prior_max_dict = prior_max
        self.prior_min = np.array([prior_min[n] for n in PARAM_NAMES])
        self.prior_max = np.array([prior_max[n] for n in PARAM_NAMES])

        # Columns the box holds in r_star units but the model and
        # tsnpe/sims.py want in kpc. Empty under 'rstar' - that emptiness
        # is what makes to_kpc/to_rstar_units the identity, with no
        # branching in either.
        self.rstar_scaled_names = (
            ('dm_log_rdm',) if radius_units == 'kpc' else ())
        self.rstar_scaled_indices = tuple(
            PARAM_NAMES.index(name) for name in self.rstar_scaled_names)

    @staticmethod
    def _validate(bounds: dict, label: str) -> None:
        """Reject a bounds dict that doesn't name exactly PARAM_NAMES."""
        unknown = sorted(set(bounds) - set(PARAM_NAMES))
        if unknown:
            raise ValueError(
                f'config.prior.{label} has unknown parameter(s) {unknown}; '
                f'expected names from {PARAM_NAMES}')
        missing = [n for n in PARAM_NAMES if n not in bounds]
        if missing:
            raise ValueError(
                f'config.prior.{label} is missing {missing}; give every '
                'parameter or leave the whole field unset to take the '
                'priorA defaults')

    # ------------------------------------------------------------------
    # Serialization - what register_run.py pins into the run directory
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """JSON-serializable form, as written to round_0/prior_config.json."""
        return {
            'radius_units': self.radius_units,
            'prior_min': {n: float(v) for n, v in self.prior_min_dict.items()},
            'prior_max': {n: float(v) for n, v in self.prior_max_dict.items()},
        }

    @classmethod
    def from_dict(cls, config: dict) -> 'Prior':
        """Rebuild from `to_dict` output, or from a `config.prior` block.

        Args:
            config: Mapping with `radius_units`, `prior_min` and
                `prior_max`. Any of them may be absent, in which case the
                priorA defaults apply.

        Returns:
            The corresponding `Prior`.
        """
        return cls(
            prior_min=config.get('prior_min') or None,
            prior_max=config.get('prior_max') or None,
            radius_units=config.get('radius_units', 'kpc'),
        )

    def __repr__(self) -> str:
        return (f'Prior(radius_units={self.radius_units!r}, '
                f'rstar_scaled={self.rstar_scaled_names})')

    # ------------------------------------------------------------------
    # Conditioning dimension
    # ------------------------------------------------------------------

    def conditioning_bounds(
        self, target: TargetData, n_sigma: float = 5.0,
    ) -> tuple[float, float]:
        """log10(r_half [kpc]) window, `n_sigma` wide, from the target.

        Args:
            target: Target snapshot providing rhalf_kpc and its uncertainty.
            n_sigma: Half-width of the window, in units of the
                (symmetrized) half-light-radius uncertainty.

        Returns:
            (log_min, log_max) bounds for the conditioning dimension.
        """
        err = 0.5 * (target.rhalf_kpc_em + target.rhalf_kpc_ep)
        return (
            float(np.log10(target.rhalf_kpc - n_sigma * err)),
            float(np.log10(target.rhalf_kpc + n_sigma * err)),
        )

    def sample_conditioning(
        self, target: TargetData, n_samples: int, n_sigma: float = 5.0,
    ) -> np.ndarray:
        """Draw the conditioning dimension from its *prior*.

        Uniform in log10(r_half), i.e. exactly the conditioning marginal
        prior_box hands sample_prior_box. The single definition of the
        conditioning prior, so the proposal samplers cannot drift into a
        different one: an importance weight of 1{...}/q is only correct
        when the conditioning draws come from this, and drawing from
        anything else silently tilts the proposal in the conditioning
        dimension.

        Not the spectroscopic likelihood, deliberately. That belongs to
        the posterior (see _gaussian_conditioning_draws); the proposal is
        drawn from the prior, and the measured r_half is the unreliable
        quantity this whole conditioning scheme exists to keep out of it.

        Args:
            target: Target snapshot resolving the conditioning bounds.
            n_samples: Number of draws.
            n_sigma: Passed to `conditioning_bounds`; must match the
                `prior_n_sigma` the proposal samplers use.

        Returns:
            (n_samples,) log10(r_half [kpc]) draws.
        """
        cond_min, cond_max = self.conditioning_bounds(target, n_sigma=n_sigma)
        return np.random.uniform(cond_min, cond_max, n_samples)

    # ------------------------------------------------------------------
    # The box
    # ------------------------------------------------------------------

    def prior_box(
        self, target: TargetData, n_sigma: float = 5.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full 8D prior box: this run's bounds + conditioning from `target`.

        Args:
            target: Target snapshot used to resolve the conditioning bounds.
            n_sigma: Passed to `conditioning_bounds`.

        Returns:
            (prior_min, prior_max) tensors, length 8.
        """
        cond_min, cond_max = self.conditioning_bounds(target, n_sigma=n_sigma)
        prior_min = torch.tensor(
            np.append(self.prior_min, cond_min), dtype=torch.float32)
        prior_max = torch.tensor(
            np.append(self.prior_max, cond_max), dtype=torch.float32)
        return prior_min, prior_max

    def sample_prior_box(
        self, n_samples: int, target: TargetData, n_sigma: float = 5.0,
    ) -> np.ndarray:
        """Draw `n_samples` uniform samples from the prior box, in box space.

        Box space means radii in r_star units; pass the result through
        `to_kpc` before it reaches the model or the simulator.

        Args:
            n_samples: Number of samples to draw.
            target: Target snapshot used to resolve the conditioning bounds.
            n_sigma: Passed to `conditioning_bounds`.

        Returns:
            (n_samples, 8) ndarray of box-space samples.
        """
        prior_min, prior_max = self.prior_box(target, n_sigma=n_sigma)
        dist = torch.distributions.Uniform(prior_min, prior_max)
        return dist.sample((n_samples,)).numpy()

    def in_prior_box(
        self, theta_kpc: np.ndarray, target: TargetData, n_sigma: float = 5.0,
    ) -> np.ndarray:
        """Mask of model-unit rows lying inside the prior box.

        Converted back to box space first, which is where the box is
        defined. Under radius_units='kpc' a cut on kpc theta cannot
        express it: dm_log_rdm is an offset from the conditioning value,
        so its kpc bounds slide with each row's own conditioning draw, and
        testing kpc values against fixed bounds admits rows whose offset
        is below the box's floor -- i.e. r_dm under r_star, which the box
        exists to forbid. Under 'rstar' the conversion is the identity and
        the cut is direct.

        Args:
            theta_kpc: (N, 8) rows in the model's radius units,
                ALL_PARAM_NAMES order.
            target: Target snapshot resolving the conditioning bounds.
            n_sigma: Passed to `conditioning_bounds`.

        Returns:
            (N,) boolean mask.
        """
        theta_box = self.to_rstar_units(np.asarray(theta_kpc, dtype=float))
        prior_min, prior_max = self.prior_box(target, n_sigma=n_sigma)
        lo = prior_min.numpy()
        hi = prior_max.numpy()
        return np.all((theta_box >= lo) & (theta_box <= hi), axis=-1)

    # ------------------------------------------------------------------
    # Radius-unit conversion
    # ------------------------------------------------------------------

    def to_kpc(self, theta_rstar: np.ndarray) -> np.ndarray:
        """Map box-space samples to the units the model and simulator use.

        `theta_rstar[:, i]` for `i in rstar_scaled_indices` holds
        log10(r / r_star), an offset from the conditioning value; the kpc
        value is offset + conditioning. Under radius_units='rstar' there
        are no such columns and this is the identity (a copy).

        Args:
            theta_rstar: (N, 8) box-space samples.

        Returns:
            (N, 8) samples in the model's radius units.
        """
        theta_kpc = np.asarray(theta_rstar, dtype=float).copy()
        cond = theta_kpc[:, CONDITIONING_INDEX]
        for i in self.rstar_scaled_indices:
            theta_kpc[:, i] = theta_kpc[:, i] + cond
        return theta_kpc

    def to_rstar_units(self, theta_kpc: np.ndarray) -> np.ndarray:
        """Inverse of `to_kpc`: map model-unit samples back to box space.

        Args:
            theta_kpc: (N, 8) samples in the model's radius units.

        Returns:
            (N, 8) box-space samples.
        """
        theta_rstar = np.asarray(theta_kpc, dtype=float).copy()
        cond = theta_rstar[:, CONDITIONING_INDEX]
        for i in self.rstar_scaled_indices:
            theta_rstar[:, i] = theta_rstar[:, i] - cond
        return theta_rstar

    # ------------------------------------------------------------------

    def default_norm_dict(self) -> dict:
        """Generic norm_dict for the random_init debug model.

        Returns:
            norm_dict with theta_loc/theta_scale (length 7, spanning this
            prior's box), cond_loc/cond_scale (length 1, unit scale), and
            x_loc/x_scale (length 3, unit scale).
        """
        theta_loc = (self.prior_max + self.prior_min) / 2
        theta_scale = (self.prior_max - self.prior_min) / 2
        return {
            'theta_loc': theta_loc.tolist(),
            'theta_scale': theta_scale.tolist(),
            'cond_loc': [0.0],
            'cond_scale': [1.0],
            'x_loc': [0.0, 0.0, 0.0],
            'x_scale': [1.0, 1.0, 1.0],
        }
