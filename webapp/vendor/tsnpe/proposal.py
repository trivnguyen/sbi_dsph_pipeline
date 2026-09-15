"""TSNPE (Deistler et al. 2022) truncated-proposal sampler.

Reads the real target's observation, applies the r_star-units/conditioning
reparametrization (tsnpe/prior.py), and computes the truncated proposal:
1. sample the posterior at Monte-Carlo conditioning values
2. estimate tau from those samples (estimate_tau)
3. sample the prior and cut at tau (sample_tsnpe_proposal)
"""

import numpy as np
import torch
import warnings

from scipy.special import logsumexp
from torch_geometric.data import Data, Batch
from tqdm import tqdm

from jgnn.transforms import build_transformation

from . import prior as prior_lib
from .target import TargetData

_LOG_EPS = 1e-6


def build_obs_pre_transforms(pre_transforms_config: dict, norm_dict: dict):
    """Pre-transforms for a real observation graph.

    Graph construction and normalization exactly match the model's own
    training config (`pre_transforms_config`, e.g. config.pre_transforms) -
    only the augmentations that assume raw pos/vel or a training-style
    batch (projection, selection, uncertainty, node-feature recompute) are
    forced off, since the observation's x is already computed manually
    (_x_obs_features) and it's a single graph, not a batch.
    """
    return build_transformation(norm_dict=norm_dict, **{
        **pre_transforms_config,
        'apply_projection': False,
        'apply_selection': False,
        'apply_uncertainty': False,
        'recompute_node_features': False,
    })


def _x_obs_features(target: TargetData) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the (x, pos) node-feature tensors shared by every conditioning draw."""
    x = torch.tensor(np.column_stack([
        np.log10(target.R_proj_kpc + _LOG_EPS),
        target.vlos_kms,
        target.vlos_err_kms,
    ]), dtype=torch.float32)
    pos = torch.tensor(
        np.column_stack([target.ra_deg, target.dec_deg]), dtype=torch.float32)
    return x, pos


def _gaussian_conditioning_draws(
    target: TargetData, n_mc_conditioning: int,
) -> np.ndarray:
    """Conditioning draws from the target's actual measured half-light
    radius and uncertainty (rhalf_kpc +- rhalf_err) - sampled in linear
    rhalf space, then log10-transformed per draw. Shared by estimate_tau
    and sample_posterior: tau is a quantile of the posterior's own
    log-density, so it must be estimated from the same posterior draws,
    not a different conditioning distribution.
    """
    err = 0.5 * (target.rhalf_kpc_em + target.rhalf_kpc_ep)
    rhalf_mc = np.clip(np.random.normal(target.rhalf_kpc, err, n_mc_conditioning), 1e-6, None)
    return np.log10(rhalf_mc)

def _prior_conditioning_draws(
    prior, target: TargetData, n_mc_conditioning: int, n_sigma: float = 5.0,
) -> np.ndarray:
    """Conditioning draws from the conditioning prior (prior.sample_conditioning).

    What the *proposal* samplers use. Same distribution prior_box hands
    sample_prior_box, so the p(cond)/p_mc(cond) ratio in an importance weight
    is identically 1 and the plain 1{in S}/q weight is correct. It is also
    the better importance proposal: the target is uniform over the whole
    conditioning window, and drawing from the Gaussian instead would put
    nearly every draw in the middle fifth of a +-5 sigma window and leave
    the rest carrying real target mass with almost no samples.
    """
    return prior.sample_conditioning(
        target, n_mc_conditioning, n_sigma=n_sigma)


def _uniform_conditioning_draws(
    target: TargetData, n_mc_conditioning: int, n_sigma=5
) -> np.ndarray:
    """Conditioning draws from the target's actual measured half-light
    radius and uncertainty (rhalf_kpc +- rhalf_err) - sampled in linear
    rhalf space, then log10-transformed per draw. Shared by estimate_tau
    and sample_posterior: tau is a quantile of the posterior's own
    log-density, so it must be estimated from the same posterior draws,
    not a different conditioning distribution.
    """
    lo = target.rhalf_kpc - n_sigma * target.rhalf_kpc_em
    hi = target.rhalf_kpc + n_sigma * target.rhalf_kpc_ep
    rhalf_mc = np.clip(np.random.uniform(lo, hi, n_mc_conditioning), 1e-6, None)
    return np.log10(rhalf_mc)


def _supports_fast_embedding(model) -> bool:
    """Whether model matches GNNEmbedding.forward's architecture:
    embedding = mlp(gnn(x, edge_index, batch)) + conditional_mlp(cond) (see
    jgnn.models.lightning.gnn_embedding.GNNEmbedding.forward). This is the
    assumption the *_fast functions below rely on to skip replicating the
    observation graph per conditioning draw/candidate - the graph term is
    independent of cond, so it only needs computing once. False for any
    other embedding architecture (e.g. no conditional_mlp); callers must
    fall back to the *_safe path in that case.
    """
    embedding_nn = getattr(model, 'embedding_nn', None)
    return (
        embedding_nn is not None
        and hasattr(embedding_nn, 'gnn')
        and hasattr(embedding_nn, 'mlp')
        and getattr(embedding_nn, 'conditional_mlp', None) is not None
    )


def _embed_observation(model, obs_graph) -> torch.Tensor:
    """The cond-independent half of GNNEmbedding.forward (mlp(gnn(...)))
    for a single observation graph. Only valid when
    _supports_fast_embedding(model) is True - callers must check that
    first. `pre_transforms=None` is implicit here (embedding_nn's own
    pre_transforms is relied on to be unset, same as elsewhere in this
    module - see _sample_posterior_mc_safe).
    """
    obs_batch = Batch.from_data_list([obs_graph])
    embedding_nn = model.embedding_nn
    bd = embedding_nn._prepare_batch(obs_batch)
    with torch.no_grad():
        return embedding_nn.mlp(
            embedding_nn.gnn(
                bd['x'], bd['edge_index'], batch=bd['batch'],
                edge_attr=bd['edge_attr'], edge_weight=bd['edge_weight'],
            )
        )


def _sample_posterior_mc_fast(
    model, prior, target: TargetData, norm_dict: dict, pre_transforms,
    cond_mc: np.ndarray, n_samples: int, return_log_prob: bool = False,
):
    """Fast path for _sample_posterior_mc - see _embed_observation and
    _supports_fast_embedding. The observation's graph embedding is
    computed exactly once (not once per conditioning draw); each draw
    then only needs the cheap conditional_mlp(cond) + a broadcast add, so
    there's no chunking/batch_size to bound - unlike replicated graphs,
    this scales trivially in len(cond_mc). Numerically verified against
    _sample_posterior_mc_safe (max abs diff ~1e-7, float32 rounding).
    """
    x, pos = _x_obs_features(target)
    obs_graph = pre_transforms(Data(x=x, pos=pos))
    graph_embedding = _embed_observation(model, obs_graph)

    n_per_draw = max(n_samples // len(cond_mc), 1)

    cond_loc = np.asarray(norm_dict['cond_loc'])
    cond_scale = np.asarray(norm_dict['cond_scale'])
    cond_norm = (np.asarray(cond_mc).reshape(-1, 1) - cond_loc) / cond_scale

    with torch.no_grad():
        cond_tensor = torch.tensor(
            cond_norm, dtype=torch.float32, device=graph_embedding.device)
        cond_embedding = model.embedding_nn.conditional_mlp(cond_tensor)
        embedding = graph_embedding + cond_embedding

        dist = model.flows(embedding)
        if return_log_prob:
            post, logq = dist.rsample_and_log_prob((n_per_draw,))
        else:
            post = dist.rsample((n_per_draw,))

    post_all = post.reshape(-1, post.shape[-1]).cpu().numpy()
    cond_all = np.repeat(cond_mc, n_per_draw)
    if return_log_prob:
        log_q_all = logq.reshape(-1).cpu().numpy()
        return post_all, cond_all, log_q_all
    return post_all, cond_all


def _sample_posterior_mc_safe(
    model, prior, target: TargetData, norm_dict: dict, pre_transforms,
    cond_mc: np.ndarray, n_samples: int, return_log_prob: bool = False,
    batch_size: int = 64,
):
    """Draw posterior samples at each of the given conditioning values.

    The conditioning dimension (stellar_log_r_star) is itself uncertain, so
    posterior samples are drawn across a spread of conditioning values
    (`cond_mc`, from _gaussian_conditioning_draws), not at one fixed point;
    shared by estimate_tau and sample_posterior.

    General path via the model's public API (model.sample_from_batch) -
    used when _supports_fast_embedding(model) is False. x/pos and
    pre_transforms never vary across draws - only cond does - so the
    observation graph is built and transformed exactly once, then cloned
    per conditioning draw and sampled in batches of up to `batch_size`
    draws per model call, rather than looping over len(cond_mc) separate
    single-graph calls - looping would recompute the (identical) graph
    embedding once per draw for no reason. Chunking (instead of one batch
    of all of len(cond_mc)) matters because n_mc_conditioning is only ever
    an approximation of the true conditioning-uncertainty marginalization
    and may be pushed arbitrarily high; this keeps memory bounded the same
    way _log_prob_candidates_safe already does for rejection-sampling
    candidates. `pre_transforms=None` below is intentional: re-passing it
    would transform an already-transformed batch a second time. This
    relies on the model itself having no pre_transforms of its own (always
    true here - see simulate_round.py/register_run.py).

    Returns:
        (post_norm, cond_phys) if return_log_prob is False, else
        (post_norm, cond_phys, log_q) - all three are ndarrays, post_norm
        is (len(cond_mc) * n_per_draw, 7), cond_phys/log_q are
        matching-length 1-D arrays (one physical conditioning value / log-prob
        per posterior sample). Torch tensors never leave this function - the
        model may live on GPU, so results are moved to CPU/numpy here rather
        than by callers guessing when that's needed.
    """
    x, pos = _x_obs_features(target)
    obs_graph = pre_transforms(Data(x=x, pos=pos))

    n_per_draw = max(n_samples // len(cond_mc), 1)

    cond_loc = np.asarray(norm_dict['cond_loc'])
    cond_scale = np.asarray(norm_dict['cond_scale'])
    cond_norm = (np.asarray(cond_mc).reshape(-1, 1) - cond_loc) / cond_scale

    n = len(cond_mc)
    post_chunks, log_q_chunks = [], []
    for start in tqdm(range(0, n, batch_size), desc='Posterior MC draws', unit='batch'):
        end = min(start + batch_size, n)
        graphs = []
        for c in cond_norm[start:end]:
            g = obs_graph.clone()
            g.cond = torch.tensor(c, dtype=torch.float32).view(1, -1)
            graphs.append(g)
        batch = Batch.from_data_list(graphs)

        out = model.sample_from_batch(
            batch, num_samples=n_per_draw, pre_transforms=None,
            return_log_prob=return_log_prob)
        post, logq = out if return_log_prob else (out, None)
        # post: (chunk_size, n_per_draw, 7) -> flatten to (chunk_size *
        # n_per_draw, 7), consistent with cond_all's per-draw repeat order.
        post_chunks.append(post.reshape(-1, post.shape[-1]).cpu().numpy())
        if return_log_prob:
            log_q_chunks.append(logq.reshape(-1).cpu().numpy())

    post_all = np.concatenate(post_chunks, axis=0)
    cond_all = np.repeat(cond_mc, n_per_draw)
    if return_log_prob:
        log_q_all = np.concatenate(log_q_chunks)
        return post_all, cond_all, log_q_all
    return post_all, cond_all


def _sample_posterior_mc(
    model, prior, target: TargetData, norm_dict: dict, pre_transforms,
    cond_mc: np.ndarray, n_samples: int, return_log_prob: bool = False,
    batch_size: int = 64,
):
    """Dispatch to _sample_posterior_mc_fast when model's architecture
    supports it (see _supports_fast_embedding), else _sample_posterior_mc_safe.
    Same contract as either: see _sample_posterior_mc_safe's docstring.
    """
    if _supports_fast_embedding(model):
        return _sample_posterior_mc_fast(
            model, prior, target, norm_dict, pre_transforms, cond_mc, n_samples,
            return_log_prob=return_log_prob)
    return _sample_posterior_mc_safe(
        model, prior, target, norm_dict, pre_transforms, cond_mc, n_samples,
        return_log_prob=return_log_prob, batch_size=batch_size)


def _normalize(prior, theta_box: np.ndarray, norm_dict: dict):
    """Map box-space samples to the model's (theta_norm, cond_norm)."""
    theta_phys = prior.to_kpc(theta_box)
    n_base = len(prior_lib.PARAM_NAMES)
    theta_loc = np.asarray(norm_dict['theta_loc'])
    theta_scale = np.asarray(norm_dict['theta_scale'])
    theta_norm = (theta_phys[:, :n_base] - theta_loc) / theta_scale

    cond_loc = np.asarray(norm_dict['cond_loc'])
    cond_scale = np.asarray(norm_dict['cond_scale'])
    cond_phys = theta_phys[:, prior_lib.CONDITIONING_INDEX].reshape(-1, 1)
    cond_norm = (cond_phys - cond_loc) / cond_scale
    return theta_norm, cond_norm


def _log_prob_candidates_fast(
    model, prior, graph_embedding: torch.Tensor, norm_dict: dict,
    theta_box_batch,
):
    """Fast path for _log_prob_candidates - see _embed_observation and
    _supports_fast_embedding. theta only ever feeds dist.log_prob(theta),
    never the embedding, so unlike the graph term no replication is needed
    at all: graph_embedding is computed once by the caller (shared across
    the *entire* rejection-sampling loop in sample_tsnpe_proposal, not just
    one chunk) and broadcast-added to every candidate's cheap
    conditional_mlp(cond) output.
    """
    theta_norm, cond_norm = _normalize(prior, theta_box_batch, norm_dict)
    with torch.no_grad():
        cond_tensor = torch.tensor(
            cond_norm, dtype=torch.float32, device=graph_embedding.device)
        cond_embedding = model.embedding_nn.conditional_mlp(cond_tensor)
        embedding = graph_embedding + cond_embedding

        theta_tensor = torch.tensor(
            theta_norm, dtype=torch.float32, device=graph_embedding.device)
        log_prob = model.flows(embedding).log_prob(theta_tensor)
    return log_prob.cpu().numpy()


def _log_prob_candidates_safe(
    model, prior, obs_graph, norm_dict, theta_box_batch, batch_size=64,
):
    """Evaluate log q(theta | conditioning) for a batch of box-space candidates.

    General path via the model's public API (model.logprob_from_batch) -
    used when _supports_fast_embedding(model) is False. obs_graph is the
    target's pre-transformed observation graph, built once by the caller -
    candidates only differ in cond/theta, not in x/pos, so each candidate
    clones obs_graph instead of rebuilding+retransforming it from scratch
    (same reasoning as _sample_posterior_mc_safe: graph construction, e.g.
    KNN, would otherwise be redone on every chunk for no reason).
    `pre_transforms=None` below is intentional; relies on the model itself
    having no pre_transforms of its own (see _sample_posterior_mc_safe).
    """
    theta_norm, cond_norm = _normalize(prior, theta_box_batch, norm_dict)
    n = len(theta_norm)
    log_probs = []
    for start in tqdm(range(0, n, batch_size), desc='Log-prob candidates', unit='batch'):
        end = min(start + batch_size, n)
        graphs = []
        for i in range(start, end):
            g = obs_graph.clone()
            g.cond = torch.tensor(cond_norm[i:i + 1], dtype=torch.float32)
            g.theta = torch.tensor(theta_norm[i:i + 1], dtype=torch.float32)
            graphs.append(g)
        batch = Batch.from_data_list(graphs)
        lq = model.logprob_from_batch(batch, pre_transforms=None)
        log_probs.append(lq.cpu().numpy())
    return np.concatenate(log_probs)


def _posterior_to_phys(
    post_norm: np.ndarray, cond_phys: np.ndarray, norm_dict: dict,
    concat=False
) -> np.ndarray:
    """Convert normalized posterior samples + their conditioning values to physical units.

    Returns:
        (N, 8) ndarray, physical units, columns matching tsnpe.prior.ALL_PARAM_NAMES.
    """
    theta_loc = np.asarray(norm_dict['theta_loc'])
    theta_scale = np.asarray(norm_dict['theta_scale'])
    theta_phys = post_norm * theta_scale + theta_loc
    return np.concatenate([theta_phys, cond_phys.reshape(-1, 1)], axis=1)


def sample_posterior(
    model,
    prior,
    target: TargetData,
    norm_dict: dict,
    pre_transforms_config: dict,
    n_samples: int = 2000,
    n_mc_conditioning: int = 100,
    conditioning_dist: str = 'gaussian',
    return_log_prob: bool = False,
    batch_size: int = 64,
    prior_n_sigma: float = 5.0,
) -> np.ndarray:
    """Draw posterior samples at the real target's observation, for diagnostics.

    The conditioning dimension is marginalized over Monte-Carlo draws, not
    fixed at a point estimate - stellar_log_r_star is itself uncertain, so
    fixing it would understate the posterior's spread. Draws come from a
    Gaussian on the target's actual rhalf_kpc +- uncertainty (see
    _gaussian_conditioning_draws) - the same draws estimate_tau uses.

    Args:
        model: Trained NPE model (jgnn.models.NPE).
        target: Observational data snapshot.
        norm_dict: Normalization dict matching the model.
        pre_transforms_config: The model's own pre_transforms config (e.g.
            config.pre_transforms) - see build_obs_pre_transforms.
        n_samples: Number of posterior draws.
        n_mc_conditioning: Number of Monte-Carlo conditioning draws - only
            ever an approximation of the true conditioning-uncertainty
            marginalization, so may be set arbitrarily high; batch_size
            keeps memory bounded regardless (see _sample_posterior_mc).
        conditioning_dist: 'gaussian' (default) or 'uniform' - which distribution
            to draw the Monte-Carlo conditioning values from. Default to Gaussian.
        return_log_prob: If True, also return the log-density of each
            posterior sample under the model (log q(theta | conditioning)).
        batch_size: Max conditioning draws per model call.

    Returns:
        (num_samples, 8) ndarray, physical units, columns matching
        tsnpe.prior.ALL_PARAM_NAMES.
    """
    model.eval()
    pre_transforms = build_obs_pre_transforms(pre_transforms_config, norm_dict)
    if conditioning_dist == 'gaussian':
        cond_mc = _gaussian_conditioning_draws(target, n_mc_conditioning)
    elif conditioning_dist == 'prior':
        cond_mc = _prior_conditioning_draws(prior, 
            target, n_mc_conditioning, n_sigma=prior_n_sigma)
    elif conditioning_dist == 'uniform':
        cond_mc = _uniform_conditioning_draws(
            target, n_mc_conditioning, n_sigma=prior_n_sigma)
    else:
        raise ValueError(
            f'conditioning_dist={conditioning_dist} not recognized;'
            f'must be "gaussian", "prior" or "uniform".'
        )

    out = _sample_posterior_mc(
        model, prior, target, norm_dict, pre_transforms, cond_mc, n_samples,
        return_log_prob=return_log_prob, batch_size=batch_size)
    post_all, cond_all, log_q_all = out if return_log_prob else (*out, None)

    # Convert first, then cut in box space via prior.in_prior_box.
    # Under RADIUS_UNITS='kpc' a [-1, 1] cut on normalized theta cannot
    # express this box: dm_log_rdm is an offset from the conditioning value,
    # so its kpc bounds slide with each row's own cond, and the fixed-bound
    # test admits offsets outside [0, 3] -- r_dm below r_star, which the box
    # exists to forbid. (Measured on draco_desi_CoreOM: the old cut admitted
    # offsets down to -1.40.) 'rejection' was never affected; it draws in
    # box space already.
    post_phys_all = _posterior_to_phys(post_all, cond_all, norm_dict, concat=False)
    in_box = prior.in_prior_box(post_phys_all, target, n_sigma=prior_n_sigma)
    if not in_box.any():
        raise RuntimeError(
            'All posterior samples fell outside the prior box. The '
            'previous round\'s model may not have converged.')
    post_phys_all = post_phys_all[in_box]

    if return_log_prob:
        log_q_all = log_q_all[in_box]
        return post_phys_all, log_q_all
    return post_phys_all

def estimate_tau(
    model,
    prior,
    target: TargetData,
    norm_dict: dict,
    pre_transforms_config: dict,
    epsilon: float = 1e-3,
    n_post_samples: int = 50_000,
    n_mc_conditioning: int = 100,
    return_posterior: bool = False,
    batch_size: int = 64,
    prior_n_sigma: float = 5.0,
):
    """Calibrate tau, the epsilon-quantile of in-box posterior log-density.

    Tau is a property of the posterior, so it's estimated from exactly the
    same posterior draws sample_posterior would produce (Gaussian
    conditioning on the target's measured rhalf_kpc +- uncertainty, see
    _gaussian_conditioning_draws) - the only extra step here is taking the
    epsilon-quantile of their log-density. `return_posterior` reuses those
    same draws instead of requiring a second, redundant sample_posterior
    call for diagnostics.

    Args:
        return_posterior: If True, also return the posterior samples
            already drawn for calibration (all of them, not just the
            in-box ones tau itself is computed from).

    Returns:
        tau, or (tau, posterior_phys) if return_posterior - posterior_phys
        is a (n_post_samples, 8) ndarray, physical units, columns matching
        tsnpe.prior.ALL_PARAM_NAMES.

    Raises:
        RuntimeError: If every posterior sample falls outside the prior box.
    """
    post_phys_all, log_q = sample_posterior(
        model, prior, target, norm_dict, pre_transforms_config,
        prior_n_sigma=prior_n_sigma,
        n_samples=n_post_samples, n_mc_conditioning=n_mc_conditioning,
        return_log_prob=True, batch_size=batch_size)
    post_all, cond_all = post_phys_all[:, :-1], post_phys_all[:, -1]

    tau = float(np.quantile(log_q, epsilon))
    print(f'  tau (eps={epsilon:.0e}): {tau:.4f} '
          f'[log-prob: {log_q.min():.2f} .. {log_q.max():.2f}]')

    if return_posterior:
        return tau, _posterior_to_phys(post_all, cond_all, norm_dict)
    return tau


def _sample_proposal_rejection(
    model, prior, target: TargetData, norm_dict: dict,
    pre_transforms_config: dict,
    tau: float, n_sims: int, draw_batch: int, batch_size: int,
    oversample_cap: int, prior_n_sigma: float,
):
    """Rejection-sample the prior, keeping candidates whose log-density
    under the model is >= tau (Deistler et al. 2022). Because the proposal
    is the prior truncated to this set, it is a proper distribution and
    standard NLL training needs no importance correction.

    Returns:
        (proposal_phys, diagnostics) - proposal_phys: (n_sims, 8) ndarray,
        physical units, columns matching tsnpe.prior.ALL_PARAM_NAMES;
        diagnostics: dict with acceptance_rate, n_drawn, n_accepted.

    Raises:
        RuntimeError: If no prior candidates pass the tau filter.
    """
    pre_transforms = build_obs_pre_transforms(pre_transforms_config, norm_dict)
    x, pos = _x_obs_features(target)
    obs_graph = pre_transforms(Data(x=x, pos=pos))

    # Checked once up front (not per-loop-iteration): if the model's
    # architecture supports it, the observation's graph embedding is
    # computed exactly once here and reused for the entire rejection-
    # sampling loop below (see _log_prob_candidates_fast), instead of
    # rebuilding+re-embedding obs_graph on every draw_batch iteration.
    use_fast = _supports_fast_embedding(model)
    if use_fast:
        graph_embedding = _embed_observation(model, obs_graph)
    n_max = n_sims * oversample_cap
    accepted = []
    n_accepted, n_drawn = 0, 0

    pbar = tqdm(total=n_sims, desc='Sampling proposal', unit='accepted')
    while n_accepted < n_sims and n_drawn < n_max:
        cands_box = prior.sample_prior_box(
            draw_batch, target, n_sigma=prior_n_sigma)
        if use_fast:
            lq = _log_prob_candidates_fast(model, prior, graph_embedding, norm_dict, cands_box)
        else:
            lq = _log_prob_candidates_safe(
                model, prior, obs_graph, norm_dict, cands_box, batch_size=batch_size)
        mask = lq >= tau
        if mask.any():
            accepted.append(prior.to_kpc(cands_box[mask]))
            n_accepted += int(mask.sum())
        n_drawn += draw_batch
        pbar.update(int(mask.sum()))
    pbar.close()

    if n_accepted == 0:
        raise RuntimeError(
            f'No candidates passed tau={tau:.3f} after {n_drawn:,} draws. '
            'Try raising epsilon or increasing oversample_cap.')

    acc_rate = n_accepted / n_drawn
    print(f'  Prior acceptance: {acc_rate:.4e} '
          f'(drawn={n_drawn:,}, accepted={n_accepted:,})')

    proposal_phys = np.concatenate(accepted)[:n_sims]
    if len(proposal_phys) < n_sims:
        # The loop also exits on n_drawn >= n_max, and the truncation above
        # then silently returns short. Loud, because the round trains on
        # whatever comes back and nothing else would notice.
        warnings.warn(
            f'Proposal is short: {len(proposal_phys):,} of {n_sims:,} '
            f'requested. Rejection sampling exhausted its budget of '
            f'n_sims * oversample_cap = {n_max:,} draws at an acceptance '
            f'rate of {acc_rate:.3e}; {int(np.ceil(n_sims / acc_rate)):,} '
            f'draws would be needed. Raise oversample_cap, raise epsilon, '
            f"or switch to sampling_mode='flow'.",
            RuntimeWarning, stacklevel=2)
        print(f'  WARNING: short proposal, {len(proposal_phys):,}/{n_sims:,}')

    diagnostics = dict(
        n_requested=int(n_sims), n_returned=int(len(proposal_phys)),
        acceptance_rate=float(acc_rate),
        n_drawn=int(n_drawn), n_accepted=int(n_accepted))
    return proposal_phys, diagnostics


def _sample_proposal_flow_rejection(
    model, prior, target: TargetData, norm_dict: dict,
    pre_transforms_config: dict,
    tau: float, n_sims: int, draw_batch: int, batch_size: int,
    oversample_cap: int, prior_n_sigma: float = 5.0, rng=None,
):
    """Exact rejection sampling with the model as the proposal density.

    Draws cond from its prior and theta from q(theta | cond), then keeps a
    draw with probability exp(tau - log q). Inside the truncated region
    q >= exp(tau) by construction, so the ratio (uniform target)/q is
    bounded by exp(-tau) and this is exact, not approximate.

    Two things make the conditioning come out right. The cond draws come
    from the prior (conditioning_dist='prior'), so the p(cond)/p_mc(cond)
    factor is 1 and never appears; and the box test runs in box space
    inside sample_posterior, so the r_dm-vs-r_star offset range is
    enforced. Drawing cond from the Gaussian instead would need an extra
    1/p_gauss(cond) factor and would leave the window's edges unsampled.

    Every accepted draw is independent and distinct, unlike
    _sample_proposal_sir, which resamples with replacement.

    Cost against _sample_proposal_rejection is 1 / (exp(tau) * V), V the
    prior volume in normalized theta -- independent of the truncated
    region, so this wins once tau > -log(V), i.e. once the posterior has
    sharpened enough that drawing from the prior stops working.
    """
    rng = np.random.default_rng() if rng is None else rng
    n_max = n_sims * oversample_cap
    accepted = []
    n_accepted, n_drawn = 0, 0

    pbar = tqdm(total=n_sims, desc='Sampling proposal (flow)', unit='accepted')
    while n_accepted < n_sims and n_drawn < n_max:
        theta, log_q = sample_posterior(
            model, prior, target, norm_dict, pre_transforms_config,
            n_samples=draw_batch, n_mc_conditioning=draw_batch,
            conditioning_dist='prior', return_log_prob=True,
            batch_size=batch_size, prior_n_sigma=prior_n_sigma)

        # The tau test is not optional: below tau, exp(tau - log q) > 1 and
        # the coin would accept everything.
        keep = (log_q >= tau) & (np.log(rng.random(len(log_q))) <= tau - log_q)
        if keep.any():
            accepted.append(theta[keep])
            n_accepted += int(keep.sum())
        n_drawn += draw_batch
        pbar.update(int(keep.sum()))
    pbar.close()

    if n_accepted == 0:
        raise RuntimeError(
            f'No draws survived rejection at tau={tau:.3f} after '
            f'{n_drawn:,} draws. Raise epsilon or oversample_cap.')

    acc_rate = n_accepted / n_drawn
    print(f'  Flow acceptance: {acc_rate:.4e} '
          f'(drawn={n_drawn:,}, accepted={n_accepted:,})')

    proposal_phys = np.concatenate(accepted)[:n_sims]
    if len(proposal_phys) < n_sims:
        warnings.warn(
            f'Proposal is short: {len(proposal_phys):,} of {n_sims:,} '
            f'requested. Flow rejection exhausted its budget of '
            f'n_sims * oversample_cap = {n_max:,} draws at an acceptance '
            f'rate of {acc_rate:.3e}. Raise oversample_cap or epsilon.',
            RuntimeWarning, stacklevel=2)
        print(f'  WARNING: short proposal, {len(proposal_phys):,}/{n_sims:,}')

    diagnostics = dict(
        n_requested=int(n_sims), n_returned=int(len(proposal_phys)),
        acceptance_rate=float(acc_rate), n_accepted=int(n_accepted),
        n_drawn=int(n_drawn))
    return proposal_phys, diagnostics


def calibrate_sampler(
    model, prior, target: TargetData, norm_dict: dict,
    pre_transforms_config: dict,
    tau: float, calibration_draws: int = 20_000, batch_size: int = 64,
    prior_n_sigma: float = 5.0, rng=None,
):
    """Measure both exact samplers' acceptance and pick the cheaper one.

    'rejection' and 'flow' are both exact and never duplicate, so the only
    thing separating them is cost, and cost is 1 / acceptance. Which one
    wins moves during a run: their acceptance ratio is exp(tau) * V, so
    the crossover sits at tau = -log(V) and tau climbs as the posterior
    sharpens. Early rounds favour 'rejection', later ones 'flow'.

    Returns:
        (mode, info) - mode is 'rejection' or 'flow'; info holds both
        measured rates and the draws they cost.
    """
    rng = np.random.default_rng() if rng is None else rng
    n = int(calibration_draws)

    pre_transforms = build_obs_pre_transforms(pre_transforms_config, norm_dict)
    x, pos = _x_obs_features(target)
    obs_graph = pre_transforms(Data(x=x, pos=pos))
    cands_box = prior.sample_prior_box(n, target, n_sigma=prior_n_sigma)
    if _supports_fast_embedding(model):
        emb = _embed_observation(model, obs_graph)
        lq = _log_prob_candidates_fast(model, prior, emb, norm_dict, cands_box)
    else:
        lq = _log_prob_candidates_safe(
            model, prior, obs_graph, norm_dict, cands_box, batch_size=batch_size)
    acc_rejection = float((lq >= tau).mean())

    _, log_q = sample_posterior(
        model, prior, target, norm_dict, pre_transforms_config,
        n_samples=n, n_mc_conditioning=n, conditioning_dist='prior',
        return_log_prob=True, batch_size=batch_size,
        prior_n_sigma=prior_n_sigma)
    keep = (log_q >= tau) & (np.log(rng.random(len(log_q))) <= tau - log_q)
    # Denominator is n, not len(log_q): sample_posterior already dropped
    # out-of-box draws, and those cost model calls just the same.
    acc_flow = float(keep.sum() / n)

    # Ties and double zeros go to 'flow': its acceptance barely moves across
    # rounds while 'rejection' is the one that collapses.
    mode = 'rejection' if acc_rejection > acc_flow else 'flow'
    info = dict(
        calibration_draws=2 * n,
        calibrated_acceptance_rejection=acc_rejection,
        calibrated_acceptance_flow=acc_flow,
    )
    print(f'  Calibration ({2 * n:,} draws): rejection={acc_rejection:.3e}, '
          f'flow={acc_flow:.3e} -> {mode}')
    if acc_rejection == 0.0 and acc_flow == 0.0:
        print('    WARNING: neither sampler accepted anything in the probe; '
              'raise epsilon or calibration_draws')
    return mode, info


def _sample_proposal_sir(
    model, prior, target: TargetData, norm_dict: dict,
    pre_transforms_config: dict,
    tau: float, n_sims: int, draw_batch: int, batch_size: int,
    oversample_cap: int, prior_n_sigma: float = 5.0,
):
    """Sampling-importance-resampling: draw theta directly from the
    posterior (uniform conditioning distribution, so draws aren't biased
    toward the target's measured rhalf), weight each by 1{log_q >= tau} / q
    (uniform prior, so weights need no extra prior-density factor), then
    resample n_sims draws proportional to those weights. Each draw_batch
    iteration sets n_mc_conditioning=draw_batch, i.e. one independently
    drawn conditioning value per theta (sample_posterior's n_per_draw=1
    case).

    Can be far more sample-efficient than _sample_proposal_rejection when
    the tau-truncated region is small relative to the prior box (low prior
    acceptance rate), since it draws from the model's own posterior rather
    than blindly from the prior - at the cost of the resampled draws being
    only approximately i.i.d. from the truncated prior (importance-weight
    degeneracy, tracked via ess_total below).

    Returns:
        (proposal_phys, diagnostics) - proposal_phys: (n_sims, 8) ndarray,
        physical units, columns matching tsnpe.prior.ALL_PARAM_NAMES;
        diagnostics: dict with ess_total (effective sample size of the
        accumulated weighted draws) and n_drawn/n_total.
    """
    ess_total = 0.0
    n_drawn = 0
    theta_running, logw_running = [], []

    pbar = tqdm(total=n_sims, desc='Sampling proposal (SIR)', unit='neff')
    while ess_total < n_sims and n_drawn < n_sims * oversample_cap:
        theta, log_q = sample_posterior(
            model, prior, target, norm_dict, pre_transforms_config,
            n_samples=draw_batch, n_mc_conditioning=draw_batch,
            conditioning_dist='prior', return_log_prob=True,
            batch_size=batch_size, prior_n_sigma=prior_n_sigma)

        in_tau = log_q >= tau
        logw = np.where(in_tau, -log_q, -np.inf)  # w propto 1{in S} / q (uniform prior)
        theta_running.append(theta)
        logw_running.append(logw)
        n_drawn += draw_batch

        logw_cat = np.concatenate(logw_running)
        w = np.exp(logw_cat - logsumexp(logw_cat))
        ess_total = 1.0 / np.sum(w ** 2)
        pbar.update(int(ess_total) - pbar.n)
    pbar.close()

    theta_running = np.concatenate(theta_running)
    logw_running = np.concatenate(logw_running)
    w = np.exp(logw_running - logsumexp(logw_running))

    idx = np.random.choice(len(theta_running), size=n_sims, replace=True, p=w)
    proposal_phys = theta_running[idx]

    print(f'  SIR effective sample size: {ess_total:.1f} '
          f'(drawn={n_drawn:,}, total={len(theta_running):,})')

    if ess_total < n_sims:
        # SIR always returns n_sims rows -- it resamples with replacement --
        # so a short budget shows up as duplicated draws, not a short array.
        warnings.warn(
            f'Proposal is effectively short: ESS {ess_total:.0f} of '
            f'{n_sims:,} requested. SIR exhausted its budget of '
            f'n_sims * oversample_cap = {n_sims * oversample_cap:,} draws, '
            f'so the {n_sims:,} returned rows contain repeats. Raise '
            f'oversample_cap or epsilon.', RuntimeWarning, stacklevel=2)
        print(f'  WARNING: ESS {ess_total:.0f} < n_sims {n_sims:,}')

    diagnostics = dict(
        n_requested=int(n_sims), ess_total=float(ess_total),
        n_drawn=int(n_drawn), n_total=int(len(theta_running)))
    return proposal_phys, diagnostics


def sample_tsnpe_proposal(
    model,
    prior,
    target: TargetData,
    norm_dict: dict,
    pre_transforms_config: dict,
    n_sims: int = 1000,
    epsilon: float = 1e-3,
    n_post_samples: int = 50_000,
    n_mc_conditioning: int = 100,
    draw_batch: int = 10_000,
    batch_size: int = 64,
    oversample_cap: int = 500,
    prior_n_sigma: float = 5.0,
    sampling_mode: str = 'rejection',
    calibration_draws: int = 20_000,
    return_posterior: bool = False,
):
    """Draw a TSNPE-truncated proposal for the next simulation round.

    Estimates tau (estimate_tau), then draws n_sims samples from the
    tau-truncated region via sampling_mode. All of them target the same
    distribution -- the prior truncated to {q >= tau} -- so it is a proper
    distribution either way and standard NLL training needs no importance
    correction.
    - 'rejection' (default): rejection-sample the prior. Exact and
      duplicate-free; degrades as the truncated region shrinks relative
      to the prior box.
    - 'flow': rejection-sample the *model*, exploiting q >= exp(tau)
      inside the region to bound the acceptance probability. Also exact
      and duplicate-free, and cheaper than 'rejection' once
      tau > -log(prior volume in normalized theta) -- i.e. in the later
      rounds where 'rejection' starts running out of budget.
    - 'sir': sampling-importance-resampling directly from the posterior.
      Same target, but resamples with replacement, so a sizable fraction
      of the returned rows are duplicates and the draws are only
      approximately i.i.d. Prefer 'flow'.
    - 'auto': measure both exact samplers on a short probe and run
      whichever accepts more (calibrate_sampler). The crossover moves
      between rounds, so this is the setting that does not need revisiting
      as a run progresses.

    Args:
        model: Trained NPE model (jgnn.models.NPE).
        target: Observational data snapshot.
        norm_dict: Fixed normalization dict (the same one used at round 0,
            never recomputed across rounds).
        pre_transforms_config: The model's own pre_transforms config (e.g.
            config.pre_transforms) - see build_obs_pre_transforms.
        n_sims: Number of proposal draws to return.
        epsilon: Posterior mass fraction excluded when calibrating tau.
        n_post_samples: Total posterior samples drawn (across all MC
            conditioning draws) to calibrate tau.
        n_mc_conditioning: Number of Monte-Carlo conditioning draws for
            tau calibration.
        draw_batch: Candidates/posterior draws per sampling_mode batch.
        batch_size: Max items per model call, both for tau calibration's
            conditioning draws and for sampling_mode's own model calls.
        oversample_cap: Hard ceiling on draws = n_sims * oversample_cap.
        prior_n_sigma: Half-width (in units of rhalf's uncertainty) of the
            conditioning dimension's prior window; see
            tsnpe.prior.conditioning_bounds. Only used by sampling_mode='rejection'.
        sampling_mode: 'auto', 'rejection', 'flow' or 'sir' - see above.
        calibration_draws: Draws given to each candidate sampler when
            sampling_mode='auto'. Costs 2x this in model calls, which is a
            rounding error against the sampling run it steers.
        return_posterior: If True, also return the posterior samples drawn
            for tau calibration (see estimate_tau) - avoids a second,
            redundant sample_posterior call for diagnostics.

    Returns:
        (proposal_phys, diagnostics), or (proposal_phys, diagnostics,
        posterior_phys) if return_posterior:
        - proposal_phys: (n_sims, 8) ndarray, physical units, columns
          matching tsnpe.prior.ALL_PARAM_NAMES.
        - diagnostics: dict with tau, n_drawn, plus sampling_mode-specific
          fields (acceptance_rate/n_accepted for 'rejection',
          ess_total/n_total for 'sir').
        - posterior_phys: (n_post_samples, 8) ndarray, same columns.

    Raises:
        RuntimeError: If every posterior sample falls outside the prior box,
            or (sampling_mode='rejection') no prior candidates pass the
            tau filter.
        ValueError: If sampling_mode isn't one of 'auto', 'rejection',
            'flow' or 'sir'.
    """
    model.eval()
    tau_result = estimate_tau(
        model, prior, target, norm_dict, pre_transforms_config,
        epsilon=epsilon, n_post_samples=n_post_samples,
        n_mc_conditioning=n_mc_conditioning,
        return_posterior=return_posterior, batch_size=batch_size)
    tau, posterior_phys = tau_result if return_posterior else (tau_result, None)

    calibration = {}
    if sampling_mode == 'auto':
        sampling_mode, calibration = calibrate_sampler(
            model, prior, target, norm_dict, pre_transforms_config, tau,
            calibration_draws=calibration_draws,
            batch_size=batch_size, prior_n_sigma=prior_n_sigma)
        calibration['sampling_mode'] = sampling_mode

    if sampling_mode == 'rejection':
        proposal_phys, diagnostics = _sample_proposal_rejection(
            model, prior, target, norm_dict, pre_transforms_config, tau,
            n_sims=n_sims, draw_batch=draw_batch, batch_size=batch_size,
            oversample_cap=oversample_cap, prior_n_sigma=prior_n_sigma)
    elif sampling_mode == 'flow':
        proposal_phys, diagnostics = _sample_proposal_flow_rejection(
            model, prior, target, norm_dict, pre_transforms_config, tau,
            n_sims=n_sims, draw_batch=draw_batch, batch_size=batch_size,
            oversample_cap=oversample_cap, prior_n_sigma=prior_n_sigma)
    elif sampling_mode == 'sir':
        proposal_phys, diagnostics = _sample_proposal_sir(
            model, prior, target, norm_dict, pre_transforms_config, tau,
            n_sims=n_sims, draw_batch=draw_batch, batch_size=batch_size,
            oversample_cap=oversample_cap, prior_n_sigma=prior_n_sigma)
    else:
        raise ValueError(
            f"sampling_mode={sampling_mode!r} not recognized; "
            "must be 'auto', 'rejection', 'flow' or 'sir'.")

    diagnostics['tau'] = tau
    # Records which sampler ran and what the probe measured, so a round's
    # choice is recoverable from state.json without re-deriving it.
    diagnostics.update(calibration)
    if return_posterior:
        return proposal_phys, diagnostics, posterior_phys
    return proposal_phys, diagnostics

