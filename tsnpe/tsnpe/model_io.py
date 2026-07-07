"""Construct a fresh NPE model from a stored model_config.

Used to warm-start round r's training from round r-1's checkpoint (see
train_round.py): jgnn.training.fit's `reset_optimizer=True` path already
knows how to load a checkpoint's weights into a freshly-built model and
start a new optimizer, so this module only needs to reconstruct the
architecture — mirroring scripts/train_npe.py's create_embedding_network /
create_model, but from a plain model_config (round_0/model_config.json)
rather than the full training config, since the pipeline never builds a
brand-new embedding network from scratch after round 0.
"""

import ml_collections
from ml_collections import ConfigDict

from jgnn.models import NPE, GNNEmbedding, TransformerEmbedding


def debug_model_config() -> ml_collections.ConfigDict:
    """Small fixed architecture for register_run.py's `random_init` debug path.

    Mirrors npe/configs/chebconv_8params.py, shrunk, plus a conditional_mlp
    for tsnpe's stellar_log_r_star conditioning.
    """
    model = ConfigDict()
    model.input_size = 3   # log10(R_proj), vlos, vlos_err
    model.output_size = 7  # len(tsnpe.prior.PARAM_NAMES)

    model.embedding = ConfigDict()
    model.embedding.type = 'gnn'
    model.embedding.gnn = ConfigDict()
    model.embedding.gnn.graph_layer = 'ChebConv'
    model.embedding.gnn.graph_layer_params = {'K': 4}
    model.embedding.gnn.hidden_sizes = [32, 32]
    model.embedding.gnn.act_name = 'relu'
    model.embedding.gnn.pooling = 'mean'
    model.embedding.gnn.layer_norm = True
    model.embedding.gnn.norm_first = False
    model.embedding.mlp = ConfigDict()
    model.embedding.mlp.hidden_sizes = [32]
    model.embedding.mlp.output_size = 32
    model.embedding.mlp.act_name = 'relu'
    model.embedding.mlp.dropout = 0.0
    model.embedding.conditional_mlp = ConfigDict()
    model.embedding.conditional_mlp.input_size = 1
    model.embedding.conditional_mlp.hidden_sizes = [32]
    model.embedding.conditional_mlp.output_size = 32
    model.embedding.conditional_mlp.act_name = 'relu'

    model.flows = ConfigDict()
    model.flows.type = 'nsf'
    model.flows.num_transforms = 2
    model.flows.hidden_features = [32, 32]
    model.flows.activation = 'tanh'
    model.flows.num_bins = 4
    model.flows.randperm = True

    return model


def debug_pre_transforms_config() -> dict:
    """Fixed pre_transforms recipe for register_run.py's `random_init` debug path.

    Mirrors npe/configs/chebconv_8params.py: fixed axis=2 projection, no
    proper motions, selection + single-feature uncertainty augmentation.
    """
    return {
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


def build_embedding(model_config: ml_collections.ConfigDict):
    """Build a fresh (randomly initialized) embedding network.

    Args:
        model_config: Architecture config with a `model.embedding` section
            (see configs/debug.py / scripts/train_npe.py's config.model).

    Returns:
        A freshly constructed GNNEmbedding or TransformerEmbedding.

    Raises:
        ValueError: If `model_config.embedding.type` is not recognized.
    """
    embed_cfg = model_config.embedding
    model_type = embed_cfg.get('type', 'gnn')
    if model_type == 'gnn':
        return GNNEmbedding(
            input_size=model_config.input_size,
            gnn_args=embed_cfg.gnn,
            mlp_args=embed_cfg.mlp,
            loss_type=embed_cfg.get('loss_type', 'mse'),
            loss_args=embed_cfg.get('loss_args', None),
            conditional_mlp_args=embed_cfg.get('conditional_mlp', None),
            # NPE handles optimizer, scheduler, and pre_transforms
            optimizer_args=None,
            scheduler_args=None,
            pre_transforms=None,
        )
    elif model_type == 'transformer':
        return TransformerEmbedding(
            input_size=model_config.input_size,
            transformer_args=embed_cfg.transformer,
            loss_type=embed_cfg.get('loss_type', 'mse'),
            loss_args=embed_cfg.get('loss_args', None),
            mlp_args=embed_cfg.get('mlp', None),
            optimizer_args=None,
            scheduler_args=None,
            pre_transforms=None,
        )
    raise ValueError(f"Unsupported embedding model type: {model_type}")


def build_npe(
    model_config: ml_collections.ConfigDict,
    pre_transforms,
    norm_dict: dict,
    optimizer_args: ml_collections.ConfigDict = None,
    scheduler_args: ml_collections.ConfigDict = None,
) -> NPE:
    """Build a fresh NPE model, ready to have a checkpoint's weights loaded in.

    Args:
        model_config: Architecture config (round_0/model_config.json).
        pre_transforms: Pre-transformation pipeline passed to NPE.
        norm_dict: Fixed normalization dict for the whole run.
        optimizer_args: Optimizer config for this round's training.
        scheduler_args: Scheduler config for this round's training.

    Returns:
        A freshly constructed (randomly initialized) NPE model with the
        stored architecture. Call `load_state_dict` (or use
        `jgnn.training.fit(..., reset_optimizer=True)`) to warm-start it
        from a previous round's checkpoint.
    """
    embedding_nn = build_embedding(model_config)
    return NPE(
        input_size=model_config.input_size,
        output_size=model_config.output_size,
        flows_args=model_config.flows,
        embedding_nn=embedding_nn,
        optimizer_args=optimizer_args,
        scheduler_args=scheduler_args,
        norm_dict=norm_dict,
        pre_transforms=pre_transforms,
        init_flows_from_embedding=False,
    )
