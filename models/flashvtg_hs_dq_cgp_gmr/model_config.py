"""nncore model configuration for the independent FlashVTG HS-DQ-CGP variant."""

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
    # HS-DQ-CGP. These values are read only by this experiment directory.
    # Main routing is level-wise; the intentionally smaller point router only
    # supplies a lambda*tanh bounded correction.
    use_dq_cgp=True,
    dq_cgp_num_basis=16,
    dq_cgp_prompt_length=6,
    dq_cgp_router_hidden_dim=256,
    dq_cgp_point_router_hidden_dim=128,
    dq_cgp_frf_hidden_dim=512,
    dq_cgp_temperature=1.0,
    dq_cgp_point_residual_scale=0.10,
    # Anti-collapse routing: semantic interaction replaces the direct level
    # embedding shortcut, logits start near uniform and remain bounded.
    dq_cgp_use_level_embedding_in_router=False,
    dq_cgp_router_logit_scale=2.0,
    dq_cgp_router_output_init_std=0.001,
    dq_cgp_beta=0.05,
    dq_cgp_locality_strength=0.5,
    dq_cgp_refine_exist=False,
    dq_cgp_binding_loss_coef=0.05,
    dq_cgp_route_loss_coef=0.005,
    # Target roughly four effective bases per individual route while requiring
    # broad utilization globally and within each level across the batch.
    dq_cgp_route_entropy_target_ratio=0.50,
    dq_cgp_min_level_usage_entropy_ratio=0.50,
    dq_cgp_route_entropy_loss_coef=0.10,
    dq_cgp_level_balance_loss_coef=0.50,
    # Keep zero for the first HS-DQ-CGP run. Enable 0.005--0.01 only if the
    # recorded adjacent JS remains unexpectedly high.
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
