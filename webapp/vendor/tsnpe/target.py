"""Target observational data: load from a survey catalog once, then keep a
minimal, self-contained copy inside the run directory.

Only the fields the TSNPE proposal sampler needs (tsnpe/proposal.py) are
kept: sky position, LOS velocity + its uncertainty, projected radius, and
the half-light-radius window used by the rstar-conditioning prior
(tsnpe/prior.py). Observational data is small, so this snapshot is a single
lightweight .npz — after registration, nothing downstream needs the
original catalog file or the `dsph_analysis` package again.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class TargetData:
    """Self-contained snapshot of one target's observational data.

    Attributes:
        key: Target identifier (e.g. 'draco_1').
        ra_deg, dec_deg: Sky position of each star, degrees.
        vlos_kms, vlos_err_kms: Line-of-sight velocity and its uncertainty, km/s.
        R_proj_kpc: Projected radius of each star, kpc.
        rhalf_kpc, rhalf_kpc_em, rhalf_kpc_ep: Half-light radius and its
            (possibly asymmetric) uncertainty, kpc.
    """
    key: str
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    vlos_kms: np.ndarray
    vlos_err_kms: np.ndarray
    R_proj_kpc: np.ndarray
    rhalf_kpc: float
    rhalf_kpc_em: float
    rhalf_kpc_ep: float

    @classmethod
    def from_catalog(cls, target_config) -> 'TargetData':
        """Load target data from a survey catalog via `dsph_analysis`.

        Args:
            target_config: config.target ConfigDict with `key`,
                `catalog_path`, and `catalog_kwargs` (forwarded to
                `kinematic_io.load_kinematic_data`).

        Returns:
            A self-contained TargetData snapshot.
        """
        from astropy import units as auni
        from dsph_analysis import kinematic_io

        meta = kinematic_io.load_meta(target_config.key)
        data = kinematic_io.load_kinematic_data(
            target_config.catalog_path, meta, **target_config.catalog_kwargs)

        return cls(
            key=target_config.key,
            ra_deg=data.ra.to_value(auni.deg).astype('float32'),
            dec_deg=data.dec.to_value(auni.deg).astype('float32'),
            vlos_kms=data.vlos.to_value(auni.km / auni.s).astype('float32'),
            vlos_err_kms=data.vlos_err.to_value(auni.km / auni.s).astype('float32'),
            R_proj_kpc=data.R_proj.to_value(auni.kpc).astype('float32'),
            rhalf_kpc=float(meta.rhalf_kpc.to_value(auni.kpc)),
            rhalf_kpc_em=float(meta.rhalf_kpc_em.to_value(auni.kpc)),
            rhalf_kpc_ep=float(meta.rhalf_kpc_ep.to_value(auni.kpc)),
        )

    def save(self, path) -> None:
        """Save this snapshot to a single .npz file.

        Args:
            path: Destination .npz path.
        """
        np.savez(
            path, key=self.key,
            ra_deg=self.ra_deg, dec_deg=self.dec_deg,
            vlos_kms=self.vlos_kms, vlos_err_kms=self.vlos_err_kms,
            R_proj_kpc=self.R_proj_kpc,
            rhalf_kpc=self.rhalf_kpc, rhalf_kpc_em=self.rhalf_kpc_em,
            rhalf_kpc_ep=self.rhalf_kpc_ep,
        )

    @classmethod
    def load(cls, path) -> 'TargetData':
        """Load a snapshot previously written by `save`.

        Args:
            path: Source .npz path.

        Returns:
            The loaded TargetData snapshot.
        """
        raw = np.load(path)
        return cls(
            key=str(raw['key']),
            ra_deg=raw['ra_deg'], dec_deg=raw['dec_deg'],
            vlos_kms=raw['vlos_kms'], vlos_err_kms=raw['vlos_err_kms'],
            R_proj_kpc=raw['R_proj_kpc'],
            rhalf_kpc=float(raw['rhalf_kpc']),
            rhalf_kpc_em=float(raw['rhalf_kpc_em']),
            rhalf_kpc_ep=float(raw['rhalf_kpc_ep']),
        )
