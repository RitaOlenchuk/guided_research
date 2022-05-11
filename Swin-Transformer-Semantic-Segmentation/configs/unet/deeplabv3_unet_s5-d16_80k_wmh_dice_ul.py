_base_ = [
    '../_base_/models/deeplabv3_unet_s5-d16.py', '../_base_/datasets/wmh_png_ul.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(test_cfg=dict(mode='whole'),
    decode_head=dict(
        loss_decode=dict(
            type='DiceLoss', use_sigmoid=False, loss_weight=1.0, ignore_index=-1)
    ),
    auxiliary_head=dict(
        loss_decode=dict(
            type='DiceLoss', use_sigmoid=False, loss_weight=0.4, ignore_index=-1)
    ))

evaluation = dict(metric='mDice')
# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
data=dict(samples_per_gpu=16, workers_per_gpu=16)
