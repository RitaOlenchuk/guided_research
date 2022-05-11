_base_ = [
    '../_base_/models/pspnet_unet_s5-d16.py', '../_base_/datasets/wmh_png.py',
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
