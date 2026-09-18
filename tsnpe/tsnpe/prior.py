"""Prior box for one TSNPE run, pinned to its round-0 model's training set.

Which box, which units and which simulator physics belong together is
described once, in dsph_sims (`ModelSpec.tsnpe`). `register_run.py`
turns that into a `Prior`, writes it to round_0/prior_config.json, and
every later round reads it back -- the same way norm_dict and the model
architecture are pinned, and for the same reason: a run whose proposal
box disagrees with the data its round-0 model saw is silently wrong.

This module deliberately does not import dsph_sims: the pinned file
carries everything a Prior needs, so consumers that only sample
posteriors (plotting/, the webapp's vendored copy, npe_inference) can
rebuild one without agama. The 8-parameter family's names live here as
constants for the same consumers; a Prior's own `param_names` is what
the pipeline code uses.

Box space vs model units
------------------------
The box always holds `dm_log_rdm` as log10(r_dm / r_star) -- an offset
from the conditioning value -- because that is the only space in which
the training prior is a box: r_dm's kpc bounds slide with each row's own
r_star draw. What differs between training sets is the units the *model*
predicts, recorded in `kpc_offset_names`: a parameter listed there is
predicted as log10(r / kpc), so `box_to_model()` adds the conditioning
value on the way out and `model_to_box()` subtracts it on the way back.
Under the r_star convention (priorB, priorC) the tuple is empty and both
maps are the identity.

`df_log_ra` is in r_star units under every convention -- the simulator
multiplies it by r_star itself -- so it is never listed.

The conditioning axis is always log10(r_half / kpc), taken from the
target rather than drawn, and named `stellar_log_rstar`.
"""

import numpy as np
import torch

from .target import TargetData

# The 8-parameter Zhao-Plummer family, for consumers that predate model
# specs. Pipeline code reads the same things off a Prior instance.
PARAM_NAMES = [
    'dm_alpha', 'dm_beta', 'dm_gamma', 'dm_log_rdm',
    'dm_log_rho0', 'df_beta0', 'df_log_ra',
]
CONDITIONING_NAME = 'stellar_log_rstar'
CONDITIONING_INDEX = len(PARAM_NAMES)  # conditioning dim is always appended last
ALL_PARAM_NAMES = PARAM_NAMES + [CONDITIONING_NAME]

RADIUS_UNITS_CHOICES = ('kpc', 'rstar')

# priorA / 8p_ZhaoPlumCOM in box space: what a bare `Prior()` is.
DEFAULT_PRIOR_MIN = {
    'dm_alpha': 0.5, 'dm_beta': 1.0, 'dm_gamma': -1.0, 'dm_log_rdm': 0.0,
    'dm_log_rho0': 3.0, 'df_beta0': -0.499, 'df_log_ra': -1.0,
}
DEFAULT_PRIOR_MAX = {
    'dm_alpha': 3.0, 'dm_beta': 10.0, 'dm_gamma': 2.0, 'dm_log_rdm': 3.0,
    'dm_log_rho0': 10.0, 'df_beta0': 1.0, 'df_log_ra': 3.0,
}

# Legacy (radius_units, sim_variant) -> model spec, for prior_config.json
# files and callers written before model specs existed. 'rstar' alone
# cannot tell priorB from priorC and resolves to priorB, which is right
# for everything that only samples posteriors (same box space, same
# units); a TSNPE run names its spec through register_run.py instead.
LEGACY_MODEL_SPECS = {
    'priorA': '8p_ZhaoPlumCOM',
    'priorB': '8p_ZhaoPlumCOM_v3',
    'priorC': '8p_ZhaoPlumCOM_v4',
}
_VARIANT_RADIUS_UNITS = {'priorA': 'kpc', 'priorB': 'rstar', 'priorC': 'rstar'}
_RADIUS_UNITS_VARIANT = {'kpc': 'priorA', 'rstar': 'priorB'}


class Prior:
    """Prior box for one TSNPE run.

    Build one with `from_spec` (a dsph_sims `ModelSpec`, the normal
    pipeline path), `from_dict` (a pinned prior_config.json, new or
    legacy form), or directly with the legacy 8-parameter keywords.

    Args:
        prior_min: Box-space lower bounds keyed by parameter name. None
            takes priorA's, which is only meaningful for the 8-parameter
            family. Every parameter must be present -- a half-specified
            box is exactly the silent mismatch this class exists to stop.
        prior_max: Upper bounds, same rules.
        radius_units: Legacy: 'kpc' (priorA) or 'rstar' (priorB/C).
        sim_variant: Legacy: 'priorA', 'priorB' or 'priorC'.
        model_spec: The dsph_sims spec name this box belongs to. When
            given, `param_names` and `kpc_offset_names` describe the box
            directly and the legacy keywords are ignored.
        param_names: Inferred parameters, model order.
        cond_name: Name of the conditioning axis.
        kpc_offset_names: Parameters the model predicts in kpc although
            the box holds them relative to r_star (see module docstring).

    Raises:
        ValueError: On unknown or inconsistent legacy keywords, a bounds
            dict that does not name exactly `param_names`, a lower bound
            not strictly below its upper bound, or an unknown
            `kpc_offset_names` entry.
    """

    def __init__(self, prior_min=None, prior_max=None, radius_units=None,
                 sim_variant=None, *, model_spec=None, param_names=None,
                 cond_name=CONDITIONING_NAME, kpc_offset_names=None):
        if model_spec is None:
            model_spec, param_names, kpc_offset_names = self._resolve_legacy(
                radius_units, sim_variant, param_names, kpc_offset_names)
        self.model_spec = model_spec
        self.param_names = tuple(PARAM_NAMES if param_names is None
                                 else param_names)
        self.cond_name = cond_name
        self.all_names = self.param_names + (cond_name,)
        self.cond_index = len(self.param_names)
        self.kpc_offset_names = tuple(kpc_offset_names or ())

        unknown = set(self.kpc_offset_names) - set(self.param_names)
        if unknown or cond_name in self.param_names:
            raise ValueError(
                f'kpc_offset_names {sorted(unknown)} not in param_names, or '
                f'cond_name {cond_name!r} collides with a parameter')
        self.kpc_offset_indices = tuple(
            self.param_names.index(n) for n in self.kpc_offset_names)

        if prior_min is None or prior_max is None:
            if self.param_names != tuple(PARAM_NAMES):
                raise ValueError(
                    'prior_min/prior_max are required unless param_names is '
                    'the 8-parameter family')
        prior_min = dict(DEFAULT_PRIOR_MIN if prior_min is None else prior_min)
        prior_max = dict(DEFAULT_PRIOR_MAX if prior_max is None else prior_max)
        self._validate(prior_min, 'prior_min')
        self._validate(prior_max, 'prior_max')
        bad = [n for n in self.param_names if prior_min[n] >= prior_max[n]]
        if bad:
            raise ValueError(
                f'prior_min must be strictly below prior_max for every '
                f'parameter; violated by {bad}')
        self.prior_min_dict = prior_min
        self.prior_max_dict = prior_max
        self.prior_min = np.array([prior_min[n] for n in self.param_names])
        self.prior_max = np.array([prior_max[n] for n in self.param_names])

    @staticmethod
    def _resolve_legacy(radius_units, sim_variant, param_names,
                        kpc_offset_names):
        """Map the pre-spec keywords onto (model_spec, names, offsets)."""
        if (radius_units is not None
                and radius_units not in RADIUS_UNITS_CHOICES):
            raise ValueError(
                f'radius_units={radius_units!r} not recognized; must be '
                f'one of {RADIUS_UNITS_CHOICES}')
        if sim_variant is not None and sim_variant not in LEGACY_MODEL_SPECS:
            raise ValueError(
                f'sim_variant={sim_variant!r} not recognized; must be one '
                f'of {tuple(LEGACY_MODEL_SPECS)}')
        variant = sim_variant or _RADIUS_UNITS_VARIANT[radius_units or 'kpc']
        if (radius_units is not None
                and _VARIANT_RADIUS_UNITS[variant] != radius_units):
            raise ValueError(
                f'sim_variant={variant!r} draws dm_log_rdm in '
                f'{_VARIANT_RADIUS_UNITS[variant]!r} units, but '
                f'radius_units={radius_units!r}')
        if kpc_offset_names is None:
            kpc_offset_names = (
                ('dm_log_rdm',) if _VARIANT_RADIUS_UNITS[variant] == 'kpc'
                else ())
        return LEGACY_MODEL_SPECS[variant], param_names, kpc_offset_names

    def _validate(self, bounds: dict, label: str) -> None:
        """Reject a bounds dict that doesn't name exactly param_names."""
        unknown = sorted(set(bounds) - set(self.param_names))
        if unknown:
            raise ValueError(
                f'{label} has unknown parameter(s) {unknown}; expected '
                f'names from {list(self.param_names)}')
        missing = [n for n in self.param_names if n not in bounds]
        if missing:
            raise ValueError(
                f'{label} is missing {missing}; give every parameter')

    @property
    def radius_units(self) -> str:
        """'kpc' if the model predicts dm_log_rdm in kpc, else 'rstar'.

        The 8-parameter family's summary of `kpc_offset_names`, kept for
        plotting/ and npe_inference, which key their kpc conversion on it.
        """
        return 'kpc' if 'dm_log_rdm' in self.kpc_offset_names else 'rstar'

    # ------------------------------------------------------------------
    # Serialization - what register_run.py pins into the run directory
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """JSON-serializable form, as written to round_0/prior_config.json.

        Identical to `dsph_sims.TsnpePrior.to_dict`, so a spec and its
        pinned copy compare equal.
        """
        return {
            'model_spec': self.model_spec,
            'param_names': list(self.param_names),
            'cond_name': self.cond_name,
            'prior_min': {n: float(v) for n, v in self.prior_min_dict.items()},
            'prior_max': {n: float(v) for n, v in self.prior_max_dict.items()},
            'kpc_offset_names': list(self.kpc_offset_names),
        }

    @classmethod
    def from_dict(cls, config: dict) -> 'Prior':
        """Rebuild from `to_dict` output or a legacy pinned/config dict.

        Args:
            config: Mapping with `model_spec`, `param_names`, `cond_name`,
                `prior_min`, `prior_max`, `kpc_offset_names` (new form),
                or `radius_units` / `sim_variant` plus bounds (legacy).
                Missing bounds take the priorA defaults.

        Returns:
            The corresponding `Prior`.
        """
        bounds = dict(prior_min=config.get('prior_min') or None,
                      prior_max=config.get('prior_max') or None)
        if config.get('model_spec'):
            return cls(
                model_spec=config['model_spec'],
                param_names=tuple(config.get('param_names') or PARAM_NAMES),
                cond_name=config.get('cond_name', CONDITIONING_NAME),
                kpc_offset_names=tuple(config.get('kpc_offset_names') or ()),
                **bounds)
        return cls(radius_units=config.get('radius_units'),
                   sim_variant=config.get('sim_variant'), **bounds)

    @classmethod
    def from_spec(cls, spec) -> 'Prior':
        """The prior a dsph_sims `ModelSpec` declares for tsnpe.

        Args:
            spec: A `dsph_sims.ModelSpec` with a `tsnpe` adapter.

        Returns:
            The corresponding `Prior`.

        Raises:
            ValueError: If the spec has no tsnpe adapter (multi-dimensional
                conditioning, a non-box prior, ...).
        """
        if getattr(spec, 'tsnpe', None) is None:
            raise ValueError(
                f'model spec {spec.name!r} has no tsnpe adapter; tsnpe '
                'cannot run it yet')
        return cls.from_dict(spec.tsnpe.to_dict(spec.name))

    def __repr__(self) -> str:
        return (f'Prior(model_spec={self.model_spec!r}, '
                f'n_params={len(self.param_names)}, '
                f'kpc_offsets={self.kpc_offset_names})')

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
        when the conditioning draws come from this.

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
        """Full prior box: this run's bounds + conditioning from `target`.

        Args:
            target: Target snapshot used to resolve the conditioning bounds.
            n_sigma: Passed to `conditioning_bounds`.

        Returns:
            (prior_min, prior_max) tensors, length len(all_names).
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
        `box_to_model` before it reaches the model.

        Args:
            n_samples: Number of samples to draw.
            target: Target snapshot used to resolve the conditioning bounds.
            n_sigma: Passed to `conditioning_bounds`.

        Returns:
            (n_samples, len(all_names)) ndarray of box-space samples.
        """
        prior_min, prior_max = self.prior_box(target, n_sigma=n_sigma)
        dist = torch.distributions.Uniform(prior_min, prior_max)
        return dist.sample((n_samples,)).numpy()

    def in_prior_box(
        self, theta_model: np.ndarray, target: TargetData,
        n_sigma: float = 5.0,
    ) -> np.ndarray:
        """Mask of model-unit rows lying inside the prior box.

        Converted back to box space first, which is where the box is
        defined: a kpc-predicted dm_log_rdm is an offset from the
        conditioning value, so its kpc bounds slide with each row's own
        conditioning draw, and testing kpc values against fixed bounds
        admits rows whose offset is below the box's floor -- r_dm under
        r_star, which the box exists to forbid.

        Args:
            theta_model: (N, len(all_names)) rows in model units.
            target: Target snapshot resolving the conditioning bounds.
            n_sigma: Passed to `conditioning_bounds`.

        Returns:
            (N,) boolean mask.
        """
        theta_box = self.model_to_box(np.asarray(theta_model, dtype=float))
        prior_min, prior_max = self.prior_box(target, n_sigma=n_sigma)
        lo = prior_min.numpy()
        hi = prior_max.numpy()
        return np.all((theta_box >= lo) & (theta_box <= hi), axis=-1)

    # ------------------------------------------------------------------
    # Box space <-> model units
    # ------------------------------------------------------------------

    def box_to_model(self, theta_box: np.ndarray) -> np.ndarray:
        """Map box-space rows to the units the model predicts.

        Columns in `kpc_offset_indices` hold log10(r / r_star), an offset
        from the conditioning value; the model value is offset +
        conditioning. With no such columns this is the identity (a copy).

        Args:
            theta_box: (N, len(all_names)) box-space rows.

        Returns:
            (N, len(all_names)) rows in model units.
        """
        theta = np.asarray(theta_box, dtype=float).copy()
        cond = theta[:, self.cond_index]
        for i in self.kpc_offset_indices:
            theta[:, i] = theta[:, i] + cond
        return theta

    def model_to_box(self, theta_model: np.ndarray) -> np.ndarray:
        """Inverse of `box_to_model`.

        Args:
            theta_model: (N, len(all_names)) rows in model units.

        Returns:
            (N, len(all_names)) box-space rows.
        """
        theta = np.asarray(theta_model, dtype=float).copy()
        cond = theta[:, self.cond_index]
        for i in self.kpc_offset_indices:
            theta[:, i] = theta[:, i] - cond
        return theta

    # Names these two went by before there was a model that predicted
    # anything but kpc; npe_inference and the webapp still call them.
    to_kpc = box_to_model
    to_rstar_units = model_to_box

    # ------------------------------------------------------------------

    def default_norm_dict(self) -> dict:
        """Generic norm_dict for the random_init debug model.

        Returns:
            norm_dict with theta_loc/theta_scale spanning this prior's
            box, cond_loc/cond_scale (length 1, unit scale), and
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
