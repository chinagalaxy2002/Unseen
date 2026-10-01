"""nncore model configuration for the independent FlashVTG-DQ-CGP variant."""

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
    # Candidate-wise DQ-CGP.  These values are read by model.py without
    # changing the baseline FlashVTG argparse implementation.
    use_dq_cgp=True,
    dq_cgp_num_basis=16,
    dq_cgp_prompt_length=6,
    dq_cgp_router_hidden_dim=256,
    dq_cgp_frf_hidden_dim=512,
    dq_cgp_temperature=1.0,
    dq_cgp_beta=0.05,
    dq_cgp_locality_strength=0.5,
    dq_cgp_refine_exist=False,
    dq_cgp_binding_loss_coef=0.05,
    dq_cgp_route_loss_coef=0.005,
    dq_cgp_gate_loss_coef=0.05,
    # Train from scratch: all parameters (backbone, existence head, DQ-CGP) are trainable.
    train_exist_head=True,
    freeze_flashvtg_baseline=False,
    gmr_decision_threshold=0.5,
)
