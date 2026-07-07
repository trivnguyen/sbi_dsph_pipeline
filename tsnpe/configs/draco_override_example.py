"""Draco 1 config: real round-0 model, with pre_transforms_override.

Same as configs/draco.py, except config.pretrained.pre_transforms_override
forces a specific pre_transforms recipe instead of reading one from the
wandb run's logged config. Use this when the checkpoint predates
pre_transforms being logged to wandb, or to deliberately try a different
recipe than the one it was actually trained with.

    python register_run.py    --config configs/draco_override_example.py
    python simulate_round.py  --config configs/draco_override_example.py --config.round=1
    python train_round.py     --config configs/draco_override_example.py --config.round=1
"""

from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder


def get_config() -> ConfigDict:
    config = ConfigDict()

    config.run_dir = '/scratch/tvnguyen/tsnpe_runs/draco_1_override'
    config.seed = 0
    config.round = 1

    config.target = ConfigDict()
    config.target.key = 'draco_1'
    config.target.catalog_path = (
        '/home/tvnguyen/links/my_projects/mock_catalogs/icrs/'
        'draco1_desi_icrs/CoreOM_mock_catalog.csv'
    )
    config.target.catalog_kwargs = ConfigDict()
    config.target.catalog_kwargs.source = 'mock_icrs'
    config.target.catalog_kwargs.mem_prob_min = 0.8
    config.target.catalog_kwargs.vlos_abs_max = 50.0
    config.target.catalog_kwargs.apply_perspective_corr = True

    config.pretrained = ConfigDict()
    config.pretrained.random_init = False
    config.pretrained.wandb_run_path = 'sbi_dsph/8Params_WidePrior/1ijk8flq'
    config.pretrained.wandb_version = 'best'
    config.pretrained.local_checkpoint_dir = ''
    config.pretrained.local_checkpoint_filename = 'model.ckpt'

    # Forces this recipe at registration instead of reading one from the
    # wandb run's config - see register_run.py's register_pretrained().
    config.pretrained.pre_transforms_override = {
        'apply_graph': True,
        'apply_projection': True,
        'apply_selection': True,
        'apply_uncertainty': True,
        'recompute_node_features': True,
        'use_log_features': True,
        'projection_args': {'axis': 2},
        'uncertainty_args': [
            dict(distribution_type='jeffreys_varied', low_range=(0.01, 0.1),
                 width_range=(5.0, 30.0), feature_idx=1),
        ],
        'selection_args': {
            'selection_configs': [
                dict(type='radial', params=dict(dropout_min=0.0, dropout_max=0.5, mode='drop_outer')),
                dict(type='radial', params=dict(dropout_min=0.0, dropout_max=0.5, mode='drop_inner')),
                dict(type='radial', params=dict(dropout_min=0.0, dropout_max=0.5, mode='random')),
            ],
            'probs': [0.6, 0.2, 0.2],
        },
        'graph_name': 'adaptive_knn',
        'graph_args': {'ratio': 0.2, 'loop': True},
    }

    config.proposal = ConfigDict()
    config.proposal.n_sims = 1000
    config.proposal.epsilon = 1e-3
    config.proposal.n_post_samples = 50_000
    config.proposal.num_mc_conditioning = 100
    config.proposal.draw_batch = 10_000
    config.proposal.embed_batch_size = 64
    config.proposal.oversample_cap = 500
    config.proposal.prior_n_sigma = 5.0

    config.simulation = ConfigDict()
    config.simulation.num_stars_mean = 100
    config.simulation.n_jobs = 0
    config.simulation.use_multiprocessing = True
    config.simulation.sample_threads = 1

    config.training = ConfigDict()
    # Set dynamically per round by train_round.py (round_dir, not a fixed
    # shared path). id is likewise set dynamically when resuming a
    # previously-recorded run.
    config.training.workdir = placeholder(str)
    config.training.id = placeholder(str)
    config.training.wandb_project = 'jgnn-tsnpe'
    config.training.entity = 'sbi_dsph'
    config.training.debug = False
    config.training.enable_progress_bar = True
    config.training.accelerator = 'gpu'
    config.training.train_batch_size = 64
    config.training.eval_batch_size = 64
    config.training.train_frac = 0.9
    config.training.num_workers = 0
    config.training.num_epochs = -1
    config.training.num_steps = 20_000
    config.training.patience = 20
    config.training.gradient_clip_val = 0.5

    config.training.optimizer = ConfigDict()
    config.training.optimizer.name = 'AdamW'
    config.training.optimizer.lr = 1e-4
    config.training.optimizer.weight_decay = 0.01

    config.training.scheduler = ConfigDict()
    config.training.scheduler.name = 'ReduceLROnPlateau'
    config.training.scheduler.factor = 0.5
    config.training.scheduler.patience = 10
    config.training.scheduler.interval = 'epoch'

    return config
