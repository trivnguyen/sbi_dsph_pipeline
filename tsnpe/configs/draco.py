"""Draco 1 config: real round-0 model, checkpoint + config from wandb.

config.pretrained.local_checkpoint_dir controls only where the checkpoint
*file* comes from: leave it empty to download it from wandb, or point it
at an already-downloaded copy (e.g. a cached
'artifacts/model-<run_id>:v<N>' directory) to skip the download. Either
way, the architecture and pre_transforms recipe are always read from
wandb (config.pretrained.wandb_run_path) and cached locally at
registration (round_0/model_config.json, round_0/pre_transforms_config.json)
- never hand-copied here, since that risks drifting from what actually
trained the checkpoint. Set config.pretrained.pre_transforms_override to
use a different recipe instead (e.g. a checkpoint trained before
pre_transforms was logged to wandb).

    python register_run.py    --config configs/draco.py
    python simulate_round.py  --config configs/draco.py --config.round=1
    python train_round.py     --config configs/draco.py --config.round=1
"""

from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder


def get_config() -> ConfigDict:
    config = ConfigDict()

    config.run_dir = '/scratch/tvnguyen/trained_models/tsnpe/test'
    config.seed = 0
    config.round = placeholder(int)
    config.overwrite = True

    config.target = ConfigDict()
    config.target.key = 'draco_1'
    config.target.catalog_path = (
        '/home/tvnguyen/links/my_projects/mock_catalogs/icrs/'
        'draco1_desi_icrs/CuspOM_mock_catalog.csv'
    )
    config.target.catalog_kwargs = ConfigDict()
    config.target.catalog_kwargs.source = 'mock_icrs'
    config.target.catalog_kwargs.mem_prob_min = 0.8
    config.target.catalog_kwargs.vlos_abs_max = 50.0
    config.target.catalog_kwargs.apply_perspective_corr = True

    config.pretrained = ConfigDict()
    config.pretrained.random_init = False
    config.pretrained.wandb_run_path = 'sbi_dsph/8p_ZhaoPlumCOM/sfaqzcwx'
    config.pretrained.wandb_version = 'best'
    config.pretrained.local_checkpoint_dir = '/scratch/tvnguyen/trained_models/npe/8p_ZhaoPlumCOM/sfaqzcwx/checkpoints'
    config.pretrained.local_checkpoint_filename = 'last.ckpt'

    config.proposal = ConfigDict()
    config.proposal.n_sims = 10_000
    config.proposal.epsilon = 1e-3
    config.proposal.n_post_samples = 10_000
    config.proposal.n_mc_conditioning = 10_000
    config.proposal.draw_batch = 20_000
    config.proposal.batch_size = 512
    config.proposal.oversample_cap = 500
    config.proposal.prior_n_sigma = 5.0

    config.simulation = ConfigDict()
    config.simulation.n_stars_mean = 100
    config.simulation.n_jobs = 0
    config.simulation.use_multiprocessing = True
    config.simulation.sample_threads = 1

    config.training = ConfigDict()
    # Set dynamically per round by train_round.py (round_dir, not a fixed
    # shared path - each round's checkpoints/wandb files stay
    # self-contained under run_dir/round_<r>/). id is likewise set
    # dynamically when resuming a previously-recorded run.
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

    config.training.enable_visualization_callback = True
    config.training.visualization = ConfigDict()
    config.training.visualization.n_posterior_samples = 500
    config.training.visualization.n_val_samples = 1000
    config.training.visualization.plot_every_n_epochs = 1
    config.training.visualization.plot_tarp = True
    config.training.visualization.plot_median_v_true = True
    config.training.visualization.plot_rank = True

    return config
