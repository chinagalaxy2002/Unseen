"""nncore configuration for the independent sparse DQ-CGP-v3 variant."""

# Reuse the baseline registry names (ConvPyramid, AdaPooling, BundleLoss).
# The DQ-CGP model itself is isolated in this directory; registering copied
# block classes a second time would conflict with nncore's global registry.
_base_ = ["models.flash_vtg_gmr.blocks"]

model = dict(
    strides=(1, 2, 4, 8),
    buffer_size=1024,
    max_num_moment=50,
    pyramid_cfg=dict(type="ConvPyramid"),
    pooling_cfg=dict(type="AdaPooling"),
    class_head_cfg=dict(type="ConvHead", kernal_size=3),
    coord_head_cfg=dict(type="ConvHead", kernal_size=3),
    loss_cfg=dict(
        type="BundleLoss",
        sample_radius=1.5,
        loss_cls=dict(type="FocalLoss"),
        loss_reg=dict(type="L1Loss"),
        loss_sal=dict(type="SampledNCELoss"),
    ),
    # DQ-CGP-v3. These values are read only by this experiment directory.
    # Main routing is sparse level-wise top-k; point routing is a bounded
    # probability-space mixture, not a logit-space perturbation.
    use_dq_cgp=True,
    dq_cgp_num_basis=16,
    dq_cgp_prompt_length=6,
    dq_cgp_router_hidden_dim=256,
    dq_cgp_point_router_hidden_dim=128,
    dq_cgp_frf_hidden_dim=512,
    dq_cgp_temperature=1.0,
    dq_cgp_point_mixture_ratio=0.10,
    dq_cgp_routing_topk=4,
    dq_cgp_local_prototype_radius=2,
    # Semantic interaction replaces the direct level-embedding shortcut;
    # logits start near uniform and remain bounded before sparse top-k routing.
    dq_cgp_use_level_embedding_in_router=False,
    dq_cgp_router_logit_scale=2.0,
    dq_cgp_router_output_init_std=0.001,
    dq_cgp_beta=0.05,
    dq_cgp_locality_strength=0.5,
    dq_cgp_refine_exist=False,
    dq_cgp_binding_loss_coef=0.05,
    dq_cgp_route_loss_coef=0.01,
    # Four bases are active by construction. Keep their entropy near 0.9*log(4)
    # while the global load-balancing term keeps all 16 bases available.
    dq_cgp_route_entropy_target_ratio=0.90,
    dq_cgp_min_level_usage_entropy_ratio=0.50,
    dq_cgp_route_entropy_loss_coef=1.00,
    dq_cgp_level_balance_loss_coef=0.50,
    # Match point-route divergence to temporal-binding divergence.  This is a
    # weak evidence-alignment term, not unconditional adjacent smoothing.
    dq_cgp_relation_loss_coef=0.02,
    dq_cgp_relation_huber_delta=0.10,
    # The relation objective supersedes unconditional adjacent smoothing.
    dq_cgp_smooth_loss_coef=0.0,
    # Train from scratch: all parameters (backbone, existence head, DQ-CGP) are trainable.
    train_exist_head=True,
    freeze_flashvtg_baseline=False,
    gmr_decision_threshold=0.5,
    # The shared trainer calls scheduler.step(training_loss), so this variant
    # must use a metric-aware scheduler instead of StepLR.
    lr_plateau_factor=0.5,
    lr_plateau_patience=10,
    lr_plateau_threshold=0.002,
    lr_min=3e-6,
)
