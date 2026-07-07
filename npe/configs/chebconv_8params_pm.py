
import numpy as np
from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder

def get_config():
    config = ConfigDict()

    # seeding
    config.seed_data = 23151
    config.seed_training = np.random.randint(0, 1_000_000)

    # data configuration
    config.data_root = '/scratch/tvnguyen/datasets/'
    config.data_name = '8p_ZhaoPlumCOM'
    config.num_datasets = 20
    config.labels = (
        'dm_alpha', 'dm_beta', 'dm_gamma', 'dm_log_rdm', 'dm_log_rho0',
        'df_beta0', 'df_log_ra',
    )
    config.cond_labels = ('stellar_log_rstar_kpc',)
    config.train_frac = 0.9
    config.num_workers = 0

    ## LOGGING AND WANDB CONFIGURATION ###
    config.wandb_project = '8p_ZhaoPlumCOM_5d'
    config.workdir = '/scratch/tvnguyen/trained_models/npe'
    config.entity = "sbi_dsph"
    config.name = None
    config.id = None
    config.tags = ['npe', 'uncertainty', 'chebconv', 'nsf']
    config.checkpoint = None
    config.reset_optimizer = False
    config.debug = False
    config.enable_progress_bar = True
    config.log_model = 'all'  # Log model checkpoints to WandB

    ### MODEL CONFIGURATION ###
    config.model = model = ConfigDict()
    model.input_size = 7
    model.output_size = len(config.labels)

    # Embedding network configuration
    model.embedding = ConfigDict()
    model.embedding.type = 'gnn'
    model.embedding.gnn = ConfigDict()
    model.embedding.gnn.graph_layer = 'ChebConv'
    model.embedding.gnn.graph_layer_params = {'K': 8}
    model.embedding.gnn.hidden_sizes = [128, ] * 5
    model.embedding.gnn.act_name = 'relu'
    model.embedding.gnn.pooling = "mean"
    model.embedding.gnn.layer_norm = True
    model.embedding.gnn.norm_first = False
    model.embedding.mlp = ConfigDict()
    model.embedding.mlp.hidden_sizes = [128, ]
    model.embedding.mlp.output_size = 128
    model.embedding.mlp.act_name = 'relu'
    model.embedding.mlp.dropout = 0.0
    model.embedding.conditional_mlp = ConfigDict()
    model.embedding.conditional_mlp.input_size = len(config.cond_labels)
    model.embedding.conditional_mlp.hidden_sizes = [128, ]
    model.embedding.conditional_mlp.output_size = 128
    model.embedding.conditional_mlp.act_name = 'relu'

    # NPE Normalizing Flows configuration
    model.flows = ConfigDict()
    model.flows.type = 'nsf'
    model.flows.num_transforms = 6
    model.flows.hidden_features = [128, 128]
    model.flows.activation = 'tanh'
    model.flows.num_bins = 8
    model.flows.randperm = True

    # Pre-transformation configuration
    # Note: For NPE, pre_transforms are passed to NPE, not to embedding_nn
    config.pre_transforms = pre_transforms = ConfigDict()
    pre_transforms.apply_graph = True if model.embedding.type == 'gnn' else False
    pre_transforms.apply_projection = True
    pre_transforms.apply_selection = True
    pre_transforms.apply_uncertainty = True
    pre_transforms.use_log_features = True
    pre_transforms.projection_args = {'axis': 2, 'use_proper_motions': True}
    pre_transforms.uncertainty_args = [
        dict(distribution_type='jeffreys_varied', low_range=(0.01, 0.1), width_range=(5.0, 30.0), feature_idx=1),
        dict(distribution_type='jeffreys_varied', low_range=(0.01, 0.1), width_range=(5.0, 30.0), feature_idx=2),
        dict(distribution_type='jeffreys_varied', low_range=(0.01, 0.1), width_range=(5.0, 30.0), feature_idx=3),
    ]
    pre_transforms.selection_args = ConfigDict()
    pre_transforms.selection_args.selection_configs = [
        dict(type='radial', params=dict(dropout_min=0.0, dropout_max=0.5, mode='drop_outer')),
        dict(type='radial', params=dict(dropout_min=0.0, dropout_max=0.5, mode='drop_inner')),
        dict(type='radial', params=dict(dropout_min=0.0, dropout_max=0.5, mode='random')),
    ]
    pre_transforms.selection_args.probs = [0.6, 0.2, 0.2]
    pre_transforms.graph_name = 'adaptive_knn'
    pre_transforms.graph_args = {'ratio': 0.2, 'loop': True}

    ### VISUALIZATION CALLBACK CONFIGURATION ###
    config.enable_visualization_callback = True
    config.visualization = visualization = ConfigDict()
    visualization.n_posterior_samples = 500
    visualization.n_val_samples = 1000
    visualization.plot_every_n_epochs = 1
    visualization.plot_tarp = True
    visualization.plot_median_v_true = True
    visualization.plot_rank = True

    ### OPTIMIZER AND SCHEDULER CONFIGURATION ###
    config.optimizer = optimizer = ConfigDict()
    optimizer.name = "AdamW"
    optimizer.lr = 5e-4
    optimizer.betas = [0.9, 0.999]
    optimizer.weight_decay = 0.01

    config.scheduler = scheduler = ConfigDict()
    # scheduler.name = None
    scheduler.name = "WarmUpCosineAnnealingLR"
    scheduler.decay_steps = int(900_000 * 2 *  0.9 * 100 / 128)
    scheduler.warmup_steps = int(0.05 * scheduler.decay_steps)
    scheduler.eta_min = 1e-6
    scheduler.interval = 'step'
    scheduler.restart = False
    scheduler.T_mult = 1

    ### TRAINING configuration ###
    config.accelerator = 'gpu'
    config.train_batch_size = 128
    config.eval_batch_size = 128
    config.num_epochs = -1
    config.num_steps = scheduler.decay_steps
    config.patience = 100
    config.gradient_clip_val = 0.5
    config.save_top_k = 5

    return config
