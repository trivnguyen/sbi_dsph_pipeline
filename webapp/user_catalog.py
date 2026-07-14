"""Fixed-format user catalog parsing for the posterior web app.

Defines the upload data format the web app (webapp/app.py) accepts and
turns an uploaded catalog into the `TargetData` snapshot the inference
stack consumes. The per-star format is fixed: `ra`, `dec` [deg], `vr`,
`vr_err` [km/s], plus a per-star `distance` [kpc] or `dm` [mag] column
(common alias spellings are tolerated, case-insensitive - see
COLUMN_ALIASES). Anything beyond that is optional and auto-detected: a
membership-probability column (threshold cut) and arbitrarily-named
boolean "flag" columns (each can be ignored, required True, or required
False).

System-level metadata the model needs but a per-star table can't carry
(half-light radius + uncertainty for the conditioning prior, center,
systemic velocity, proper motion for the perspective correction) comes
in as explicit arguments to `build_target`, with sensible data-driven
defaults where possible (center/systemic velocity from the member stars
themselves).
"""

import sys
from pathlib import Path

import astropy.table as at
import astropy.units as auni
import numpy as np
import pandas as pd

# In the packaged bundle, `tsnpe`/`dsph_analysis` sit next to this
# file; in the repo, the tsnpe package lives in ../tsnpe.
_APP_DIR = Path(__file__).resolve().parent
for _p in (_APP_DIR, _APP_DIR.parent / 'tsnpe'):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from tsnpe.target import TargetData

from dsph_analysis import data_utils
from dsph_analysis.kinematic_io import DwarfMeta

# Canonical role -> accepted (normalized) column spellings. Normalization
# lowercases and strips '_'/'-'/' ' (see _normalize), so e.g. 'VR_ERR',
# 'vr err' and 'vrerr' all match 'vr_err'.
COLUMN_ALIASES = {
    'ra': ('ra', 'radeg', 'raj2000', 'alpha'),
    'dec': ('dec', 'decdeg', 'dej2000', 'de', 'delta'),
    'distance': ('distance', 'distancekpc', 'dist', 'distkpc', 'dkpc'),
    'dm': ('dm', 'dmod', 'distmod', 'distancemodulus', 'mu'),
    'vr': ('vr', 'vlos', 'vrad', 'rv', 'vhel', 'vhelio'),
    'vr_err': ('vrerr', 'vloserr', 'vraderr', 'evr', 'rverr', 'verr',
               'errvr', 'evlos', 'vrade'),
    'mem_prob': ('memprob', 'prob', 'pmem', 'membershipprob', 'memp',
                 'pmember', 'membership'),
}
REQUIRED_ROLES = ('ra', 'dec', 'vr', 'vr_err')

_TRUE_STRINGS = {'true', 't', 'yes', 'y', '1'}
_FALSE_STRINGS = {'false', 'f', 'no', 'n', '0'}


def _normalize(name: str) -> str:
    """Normalize a column name for alias matching."""
    return (name.strip().lower()
            .replace('_', '').replace('-', '').replace(' ', ''))


def read_catalog(path: str) -> pd.DataFrame:
    """Read an uploaded catalog file into a DataFrame.

    Args:
        path: Path to the uploaded file. `.ecsv`/`.fits` go through
            astropy Table; anything else is parsed as CSV.

    Returns:
        The catalog as a pandas DataFrame.
    """
    suffix = Path(path).suffix.lower()
    if suffix in ('.ecsv', '.fits'):
        return at.Table.read(path).to_pandas()
    return pd.read_csv(path)


def _is_flag_column(series: pd.Series) -> bool:
    """Whether a column looks boolean: bool dtype, {0, 1} numerics, or
    true/false-like strings.
    """
    if pd.api.types.is_bool_dtype(series):
        return True
    values = series.dropna().unique()
    if len(values) == 0:
        return False
    if pd.api.types.is_numeric_dtype(series):
        return set(np.unique(values.astype(float))) <= {0.0, 1.0}
    if series.dtype == object:
        lowered = {str(v).strip().lower() for v in values}
        return lowered <= (_TRUE_STRINGS | _FALSE_STRINGS)
    return False


def _flag_values(series: pd.Series) -> np.ndarray:
    """Parse a flag column (see _is_flag_column) to a boolean array.

    NaN entries parse as False, so 'require True' drops them and
    'require False' keeps them - consistent with treating NaN as
    "flag not set".
    """
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy()
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).to_numpy().astype(float) == 1.0
    return np.array([
        str(v).strip().lower() in _TRUE_STRINGS for v in series
    ])


def inspect_catalog(df: pd.DataFrame) -> dict:
    """Auto-detect the catalog's column roles for the upload UI.

    Args:
        df: Catalog as returned by `read_catalog`.

    Returns:
        JSON-friendly dict with `n_rows`, `columns` (all column names),
        `mapping` (role -> matched column name for every recognized
        role), `missing` (required roles with no match), `has_distance`
        (a distance or dm column matched), and `flag_columns`
        (boolean-like columns not consumed by any role).
    """
    mapping = {}
    for column in df.columns:
        key = _normalize(str(column))
        for role, aliases in COLUMN_ALIASES.items():
            if role not in mapping and key in aliases:
                mapping[role] = column
                break

    consumed = set(mapping.values())
    flag_columns = [
        str(c) for c in df.columns
        if c not in consumed and _is_flag_column(df[c])
    ]
    missing = [r for r in REQUIRED_ROLES if r not in mapping]
    return dict(
        n_rows=int(len(df)),
        columns=[str(c) for c in df.columns],
        mapping={k: str(v) for k, v in mapping.items()},
        missing=missing,
        has_distance=('distance' in mapping or 'dm' in mapping),
        flag_columns=flag_columns,
    )


def _median_center(ra_deg: np.ndarray, dec_deg: np.ndarray) -> tuple:
    """Median sky position, handling the RA wrap at 0/360.

    A field straddling RA=0 has raw values near both 0 and 360, whose
    plain median lands near 180 - shifting by 180 deg first moves the
    wrap to RA=180, far from any such field's stars.
    """
    if np.ptp(ra_deg) > 180.0:
        ra_center = (np.median((ra_deg + 180.0) % 360.0) - 180.0) % 360.0
    else:
        ra_center = float(np.median(ra_deg))
    return float(ra_center), float(np.median(dec_deg))


def resolve_mapping(df: pd.DataFrame, columns: dict = None) -> dict:
    """Final role -> column mapping: auto-detection plus user choices.

    Args:
        df: Catalog as returned by `read_catalog`.
        columns: Optional overrides from the UI's per-role column
            dropdowns - {role: column name}, where an empty/None column
            name unsets an auto-detected role (e.g. "don't use the
            membership column after all").

    Returns:
        {role: column name} with overrides applied.

    Raises:
        ValueError: If an override names an unknown role or a column
            not in the catalog, or a required role ends up unmapped.
    """
    mapping = inspect_catalog(df)['mapping']
    for role, column in (columns or {}).items():
        if role not in COLUMN_ALIASES:
            raise ValueError(f'Unknown column role: {role!r}')
        if column:
            if column not in df.columns:
                raise ValueError(
                    f'Column {column!r} (chosen for {role!r}) not in '
                    f'the catalog.')
            mapping[role] = column
        else:
            mapping.pop(role, None)

    missing = [r for r in REQUIRED_ROLES if r not in mapping]
    if missing:
        raise ValueError(
            f'No column assigned for required role(s): {missing}. '
            f'Catalog columns: {[str(c) for c in df.columns]}')
    return mapping


def build_target(
    df: pd.DataFrame,
    label: str,
    rhalf_kpc: float,
    rhalf_kpc_em: float,
    rhalf_kpc_ep: float,
    columns: dict = None,
    center_ra_deg: float = None,
    center_dec_deg: float = None,
    distance_kpc: float = None,
    vlos_systemic_kms: float = None,
    pmra_masyr: float = None,
    pmdec_masyr: float = None,
    vlos_abs_max: float = None,
    mem_prob_min: float = None,
    flag_requirements: dict = None,
    apply_perspective_corr: bool = True,
) -> tuple[TargetData, dict]:
    """Build a TargetData snapshot from a fixed-format user catalog.

    Args:
        df: Catalog as returned by `read_catalog`.
        label: Display name for this run (becomes TargetData.key).
        rhalf_kpc: Projected half-light radius [kpc] - required, the
            model conditions on it.
        rhalf_kpc_em: Lower (minus) uncertainty on rhalf_kpc [kpc].
        rhalf_kpc_ep: Upper (plus) uncertainty on rhalf_kpc [kpc].
        columns: Optional role -> column-name overrides on top of the
            auto-detected mapping (see `resolve_mapping`).
        center_ra_deg: System center RA [deg]; None = median of the
            selected member stars.
        center_dec_deg: System center Dec [deg]; None = median.
        distance_kpc: System distance [kpc]; None = median of the
            catalog's distance (or dm) column. Required if the catalog
            has neither column.
        vlos_systemic_kms: Systemic LOS velocity [km/s]; None = median
            of the selected stars' vr.
        pmra_masyr: Systemic proper motion in RA*cos(Dec) [mas/yr];
            needed (with pmdec_masyr) for the perspective correction.
        pmdec_masyr: Systemic proper motion in Dec [mas/yr].
        vlos_abs_max: Optional cut on |vr - v_sys| [km/s].
        mem_prob_min: Optional membership-probability threshold; only
            applied if the catalog has a membership column.
        flag_requirements: {column name: required bool value} for the
            catalog's flag columns; unlisted flags are ignored.
        apply_perspective_corr: Apply the full perspective-rotation
            correction (needs both proper motions); otherwise only the
            systemic velocity is subtracted.

    Returns:
        (target, info) - the TargetData snapshot plus a JSON-friendly
        dict recording the star counts and the metadata values actually
        used (including defaults filled in from the data).

    Raises:
        ValueError: If required columns are missing, no distance is
            available, an unknown flag column is named, or every star is
            cut.
    """
    mapping = resolve_mapping(df, columns)

    mask = np.ones(len(df), dtype=bool)
    for column, required in (flag_requirements or {}).items():
        if column not in df.columns:
            raise ValueError(f'Unknown flag column: {column!r}')
        mask &= _flag_values(df[column]) == bool(required)

    mem_prob = None
    if 'mem_prob' in mapping:
        mem_prob = df[mapping['mem_prob']].to_numpy().astype(float)
        if mem_prob_min is not None:
            mask &= np.nan_to_num(mem_prob) > mem_prob_min
    data = df[mask]
    if len(data) == 0:
        raise ValueError('All stars removed by flag/membership cuts.')

    ra = data[mapping['ra']].to_numpy().astype(float)
    dec = data[mapping['dec']].to_numpy().astype(float)
    vr = data[mapping['vr']].to_numpy().astype(float)
    vr_err = data[mapping['vr_err']].to_numpy().astype(float)
    mem_prob = (mem_prob[mask] if mem_prob is not None
                else np.ones(len(data)))

    if distance_kpc is None:
        if 'distance' in mapping:
            distance_kpc = float(np.nanmedian(
                data[mapping['distance']].to_numpy().astype(float)))
        elif 'dm' in mapping:
            dm = float(np.nanmedian(
                data[mapping['dm']].to_numpy().astype(float)))
            distance_kpc = 10.0 ** (dm / 5.0 - 2.0)
        else:
            raise ValueError(
                'Catalog has no distance/dm column - provide the '
                'system distance explicitly.')

    if center_ra_deg is None or center_dec_deg is None:
        auto_ra, auto_dec = _median_center(ra, dec)
        center_ra_deg = auto_ra if center_ra_deg is None else center_ra_deg
        center_dec_deg = (auto_dec if center_dec_deg is None
                          else center_dec_deg)
    if vlos_systemic_kms is None:
        vlos_systemic_kms = float(np.nanmedian(vr))

    # The perspective correction needs a systemic proper motion; without
    # one, quietly fall back to plain systemic-velocity subtraction.
    have_pm = pmra_masyr is not None and pmdec_masyr is not None
    use_corr = bool(apply_perspective_corr and have_pm)

    meta = DwarfMeta(
        key=label,
        ra=center_ra_deg * auni.deg,
        dec=center_dec_deg * auni.deg,
        distance=distance_kpc * auni.kpc,
        pmra=(pmra_masyr or 0.0) * auni.mas / auni.yr,
        pmdec=(pmdec_masyr or 0.0) * auni.mas / auni.yr,
        vlos_systemic=vlos_systemic_kms * auni.km / auni.s,
        rhalf_arcmin=np.nan * auni.arcmin,
        rhalf_arcmin_em=np.nan * auni.arcmin,
        rhalf_arcmin_ep=np.nan * auni.arcmin,
        rhalf_kpc=rhalf_kpc * auni.kpc,
        rhalf_kpc_em=rhalf_kpc_em * auni.kpc,
        rhalf_kpc_ep=rhalf_kpc_ep * auni.kpc,
    )

    ra, dec, _, vr_err, mem_prob, vlos, r_proj = (
        data_utils.preprocess_kinematic_data(
            ra, dec, vr, vr_err, mem_prob, meta,
            vlos_abs_max=vlos_abs_max,
            apply_perspective_corr=use_corr,
        ))
    if len(ra) == 0:
        raise ValueError('All stars removed by NaN/vlos_abs_max cuts.')

    target = TargetData(
        key=label,
        ra_deg=ra.astype('float32'),
        dec_deg=dec.astype('float32'),
        vlos_kms=vlos.astype('float32'),
        vlos_err_kms=vr_err.astype('float32'),
        R_proj_kpc=r_proj.astype('float32'),
        rhalf_kpc=float(rhalf_kpc),
        rhalf_kpc_em=float(rhalf_kpc_em),
        rhalf_kpc_ep=float(rhalf_kpc_ep),
    )
    info = dict(
        n_uploaded=int(len(df)),
        n_after_cuts=int(len(ra)),
        center_ra_deg=float(center_ra_deg),
        center_dec_deg=float(center_dec_deg),
        distance_kpc=float(distance_kpc),
        vlos_systemic_kms=float(vlos_systemic_kms),
        perspective_corr_applied=use_corr,
        rhalf_kpc=float(rhalf_kpc),
    )
    return target, info
