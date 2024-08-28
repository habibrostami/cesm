from packages.parameters.Datasets import (cm_cc, cm_mlo, dm_cc, dm_mlo, cm, dm, mlo,
                                          cm_cc_seg, cm_mlo_seg, dm_cc_seg, dm_mlo_seg,
                                          cm_seg, dm_seg, full, cm_cc_pre_seg, cm_mlo_pre_seg, dm_cc_pre_seg,
                                          dm_mlo_pre_seg, full_pre_segment)

splits = (
    {
        'name': 'joint-unet-pre-segment-cm-cc[b64]',
        'dataset': cm_cc_pre_seg,
        'epochs': 50,
        'train': False,
        'type': 'joint-unet-pre-segment',
        'batch_size': 64,
        'bilinear': True
    },
{
        'name': 'joint-unet-pre-segment-cm-mlo[b64]',
        'dataset': cm_mlo_pre_seg,
        'epochs': 50,
        'train': False,
        'type': 'joint-unet-pre-segment',
        'batch_size': 64,
        'bilinear': True
    },
{
        'name': 'joint-unet-pre-segment-dm-cc[b64]',
        'dataset': dm_cc_pre_seg,
        'epochs': 50,
        'train': False,
        'type': 'joint-unet-pre-segment',
        'batch_size': 64,
        'bilinear': True
    },
{
        'name': 'joint-unet-pre-segment-dm-mlo[b64]',
        'dataset': dm_mlo_pre_seg,
        'epochs': 50,
        'train': False,
        'type': 'joint-unet-pre-segment',
        'batch_size': 64,
        'bilinear': True
    },
    {
        'name': 'joint-pre-segment-ensemble',
        'dataset': full_pre_segment,
        'epochs': 50,
        'train': False,
        'type': 'joint-pre-segment-ensemble',
        'batch_size': 64,
        'cm_cc_path': './save:manual:joint-unet-pre-segment-cm-cc[b64]/best_val/accuracy/whole.pth',
        'cm_mlo_path': './save:manual:joint-unet-pre-segment-cm-mlo[b64]/best_val/accuracy/whole.pth',
        'dm_cc_path': './save:manual:joint-unet-pre-segment-dm-cc[b64]/best_val/accuracy/whole.pth',
        'dm_mlo_path': './save:manual:joint-unet-pre-segment-dm-mlo[b64]/best_val/accuracy/whole.pth'
    },
    {
        'name': 'joint-unet-cm-cc',
        'dataset': cm_cc,
        'epochs': 10,
        'train': False,
        'type': 'joint-unet',
        'batch_size': 8,
        'unet_path': './save:manual:unet[cm_cc]/best_val/dice/whole.pth',
        'bilinear': True
    },
    {
        'name': 'joint-unet-cm-mlo',
        'dataset': cm_mlo,
        'epochs': 50,
        'train': False,
        'type': 'joint-unet',
        'batch_size': 8,
        'unet_path': './save:manual:unet[cm_mlo]_500/best_val/dice/whole.pth',
        'bilinear': True
    },
    {
        'name': 'joint-unet-dm-cc',
        'dataset': dm_cc,
        'epochs': 20,
        'train': False,
        'type': 'joint-unet',
        'batch_size': 50,
        'unet_path': './save:manual:unet[dm_cc]_500/best_val/dice/whole.pth',
        'bilinear': True
    },
    {
        'name': 'joint-unet-dm-mlo',
        'dataset': dm_mlo,
        'epochs': 50,
        'train': False,
        'type': 'joint-unet',
        'batch_size': 8,
        'unet_path': './save:manual:unet[dm_mlo]_500/best_val/dice/whole.pth',
        'bilinear': True
    },
    {
        'name': 'joint-unet-ensemble',
        'dataset': full,
        'epochs': 50,
        'train': False,
        'type': 'joint-unet-ensemble',
        'batch_size': 8,
        'cm_cc_path': './save:manual:joint-unet-cm-cc/best_val/accuracy/whole.pth',
        'cm_mlo_path': './save:manual:joint-unet-cm-mlo/best_val/accuracy/whole.pth',
        'dm_cc_path': './save:manual:joint-unet-dm-cc/best_val/accuracy/whole.pth',
        'dm_mlo_path': './save:manual:joint-unet-dm-mlo/best_val/accuracy/whole.pth'
    },
    {
        'name': 'seg_class_cm_cc',
        'dataset': cm_cc,
        'epochs': 20,
        'train': False,
        'type': 'segment_classifier',
        'batch_size': 64,
        'resnet_size': 18,
        'segment_path': './save:manual:unet[cm_cc]_500/best_val/dice/whole.pth',
        'bilinear': True
    },
    {
        'name': 'seg_class_cm_mlo',
        'dataset': cm_mlo,
        'epochs': 30,
        'train': False,
        'type': 'segment_classifier',
        'batch_size': 64,
        'resnet_size': 18,
        'segment_path': './save:manual:unet[cm_mlo]_orgdata/best_val/dice/whole.pth',
            # './save:manual:unet[cm_mlo]_500/best_val/dice/whole.pth',
        'bilinear': True
    },
    {
        'name': 'seg_class_dm_cc',
        'dataset': dm_cc,
        'epochs': 50,
        'train': False,
        'type': 'segment_classifier',
        'batch_size': 16,
        'resnet_size': 18,
        'segment_path': './save:manual:unet[dm_cc]_500/best_val/dice/whole.pth',
        'bilinear': True
    },
    {
        'name': 'seg_class_dm_mlo',
        'dataset': dm_mlo,
        'epochs': 50,
        'train': False,
        'type': 'segment_classifier',
        'batch_size': 16,
        'resnet_size': 18,
        'segment_path': './save:manual:unet[dm_mlo]_500/best_val/dice/whole.pth',
        'bilinear': True
    },
    {
        'name': 'Joint[cm_cc]',
        'dataset': cm_cc,
        'epochs': 100,
        'resnet_size': 18,
        'train': False,
        'type': 'simple-run',
        'batch_size': 32
    },
    {
        'name': 'res18[cm_mlo]',
        'dataset': cm_mlo,
        'epochs': 100,
        'resnet_size': 18,
        'train': False,
        'type': 'simple-run',
        'batch_size': 32
    },
    {
        'name': 'res18[dm_cc]',
        'dataset': dm_cc,
        'epochs': 100,
        'resnet_size': 18,
        'train': False,
        'type': 'simple-run',
        'batch_size': 32
    },
    {
        'name': 'res18[dm_mlo]',
        'dataset': dm_mlo,
        'epochs': 100,
        'resnet_size': 18,
        'train': False,
        'type': 'simple-run',
        'batch_size': 32
    },
    {
        'name': 'unet[cm_cc]_(new_mask)',
        'dataset': cm_cc_seg,
        'epochs': 500,
        'resnet_size': 18,
        'train': False,
        'type': 'u-net',
        'batch_size': 64,
        'bilinear': True
    },
    {
        'name': 'unet[cm_mlo]_(new_mask)',
        'dataset': cm_mlo_seg,
        'epochs': 500,
        'resnet_size': 18,
        'train': False,
        'type': 'u-net',
        'batch_size': 64,
        'bilinear': True
    },
    {
        'name': 'unet[dm_cc]_(new_mask)',
        'dataset': dm_cc_seg,
        'epochs': 500,
        'resnet_size': 18,
        'train': False,
        'type': 'u-net',
        'batch_size': 64,
        'bilinear': True
    },
    {
        'name': 'unet[dm_mlo]_(new_mask)',
        'dataset': dm_mlo_seg,
        'epochs': 500,
        'resnet_size': 18,
        'train': False,
        'type': 'u-net',
        'batch_size': 64,
        'bilinear': True
    },
    {
        'name': 'unet[cm]',
        'dataset': cm_seg,
        'epochs': 100,
        'resnet_size': 18,
        'train': False,
        'type': 'u-net',
        'batch_size': 64,
        'bilinear': True
    },
    {
        'name': 'unet[dm]',
        'dataset': dm_seg,
        'epochs': 500,
        'resnet_size': 18,
        'train': False,
        'type': 'u-net',
        'batch_size': 64,
        'bilinear': True
    },

    {
        'name': 'swin-unet[cm_cc]',
        'dataset': cm_cc_seg,
        'epochs': 500,
        'resnet_size': 18,
        'train': False,
        'type': 'swin-unet',
        'model_path': '/mnt/2T/BreastCancerAll/SwinUNet/out/epoch_149.pth',
        'batch_size': 64,
        'bilinear': True
    },
    #/mnt/2T/BreastCancerAll/SwinUNet/pretrained_ckpt/swin_tiny_patch4_window7_224.pth
    {
        'name': 'swin-unet[cm_mlo]',
        'dataset': cm_mlo_seg,
        'epochs': 500,
        'resnet_size': 18,
        'train': False,
        'type': 'swin-unet',
        'batch_size': 64,
        'bilinear': True
    },
    {
        'name': 'swin-unet[dm_cc]',
        'dataset': dm_cc_seg,
        'epochs': 500,
        'resnet_size': 18,
        'train': False,
        'type': 'swin-unet',
        'batch_size': 64,
        'bilinear': True
    },
    {
        'name': 'swin-unet[dm_mlo]',
        'dataset': dm_mlo_seg,
        'epochs': 500,
        'resnet_size': 18,
        'train': False,
        'type': 'swin-unet',
        'batch_size': 64,
        'bilinear': True
    },
    {
        'name': 'RefineNet(cm_cc)',
        'dataset': cm_cc_seg,
        'epochs': 100,
        'resnet_size': 18,
        'train': False,
        'type': 'u-net',
        'batch_size': 64,
        'bilinear': True
    },
    {
        'name': 'Joint-ROI[cm_cc]-batch16',
        'dataset': cm_cc,
        'epochs': 100,
        'resnet_size': 18,
        'train': False,
        'type': 'simple-run',
        'batch_size': 16
    },
    {
        'name': 'Joint-ROI[cm_mlo]-batch16',
        'dataset': cm_mlo,
        'epochs': 100,
        'resnet_size': 18,
        'train': False,
        'type': 'simple-run',
        'batch_size': 16
    },
    {
        'name': 'Joint-ROI[dm_cc]-batch16',
        'dataset': dm_cc,
        'epochs': 100,
        'resnet_size': 18,
        'train': False,
        'type': 'simple-run',
        'batch_size': 16
    },
    {
        'name': 'Joint-ROI[dm_mlo]-batch16',
        'dataset': dm_mlo,
        'epochs': 100,
        'resnet_size': 18,
        'train': False,
        'type': 'simple-run',
        'batch_size': 16
    },
    {
        'name': 'joint-ROI-ensemble',
        'dataset': full,
        'epochs': 50,
        'train': False,
        'type': 'joint-unet-ensemble',
        'batch_size': 8,
        'cm_cc_path': './save:manual:Joint-ROI[cm_cc]-batch16:18/best_val/accuracy/whole.pth',
        'cm_mlo_path': './save:manual:Joint-ROI[cm_mlo]-batch16:18/best_val/accuracy/whole.pth',
        'dm_cc_path': './save:manual:Joint-ROI[dm_cc]-batch16:18/best_val/accuracy/whole.pth',
        'dm_mlo_path': './save:manual:Joint-ROI[dm_mlo]-batch16:18/best_val/accuracy/whole.pth'
    },
    {
        'name': 'Joint-2attentionmask[cm_cc]',
        'dataset': cm_cc,
        'epochs': 50,
        'resnet_size': 18,
        'train': False,
        'type': 'simple-run',
        'batch_size': 64
    },
    {
        'name': 'Joint-2attentionmask[cm_mlo]',
        'dataset': cm_mlo,
        'epochs': 50,
        'resnet_size': 18,
        'train': False,
        'type': 'simple-run',
        'batch_size': 64
    },
    {
        'name': 'Joint-2attentionmask[dm_cc]',
        'dataset': dm_cc,
        'epochs': 50,
        'resnet_size': 18,
        'train': False,
        'type': 'simple-run',
        'batch_size': 64
    },
    {
        'name': 'Joint-2attentionmask[dm_mlo]',
        'dataset': dm_mlo,
        'epochs': 50,
        'resnet_size': 18,
        'train': False,
        'type': 'simple-run',
        'batch_size': 64
    },
    {
        'name': 'joint-attentionmask-ensemble',
        'dataset': full,
        'epochs': 50,
        'train': False,
        'type': 'joint-unet-ensemble',
        'batch_size': 64,
        'cm_cc_path': './save:manual:Joint-2attentionmask[cm_cc]:18/best_val/accuracy/whole.pth',
        'cm_mlo_path': './save:manual:Joint-2attentionmask[cm_mlo]:18/best_val/accuracy/whole.pth',
        'dm_cc_path': './save:manual:Joint-2attentionmask[dm_cc]:18/best_val/accuracy/whole.pth',
        'dm_mlo_path': './save:manual:Joint-2attentionmask[dm_mlo]:18/best_val/accuracy/whole.pth'
    },
    {
        'name': 'ROI-AutoUnet[cm_cc]',
        'dataset': cm_cc,
        'epochs': 50,
        'resnet_size': 18,
        'train': False,
        'type': 'simple-run',
        'batch_size': 64
    },
    {
        'name': 'ROI-AutoUnet[cm_mlo]_aug',
        'dataset': cm_mlo,
        'epochs': 50,
        'resnet_size': 18,
        'train': False,
        'type': 'simple-run',
        'batch_size': 64
    },
    {
        'name': 'ROI-AutoUnet[dm_cc]_aug',
        'dataset': dm_cc,
        'epochs': 50,
        'resnet_size': 18,
        'train': False,
        'type': 'simple-run',
        'batch_size': 64
    },
    {
        'name': 'ROI-AutoUnet[dm_mlo]_aug',
        'dataset': dm_mlo,
        'epochs':50,
        'resnet_size': 18,
        'train': False,
        'type': 'simple-run',
        'batch_size': 64
    },
    {
        'name': 'ROI-unet-ensemble',
        'dataset': full,
        'epochs': 50,
        'train': False,
        'type': 'joint-unet-ensemble',
        'batch_size': 64,
        'cm_cc_path': './save:manual:ROI-AutoUnet[cm_cc]:18/best_val/accuracy/whole.pth',
        'cm_mlo_path': './save:manual:ROI-AutoUnet[cm_mlo]:18/best_val/accuracy/whole.pth',
        'dm_cc_path': './save:manual:ROI-AutoUnet[dm_cc]:18/best_val/accuracy/whole.pth',
        'dm_mlo_path': './save:manual:ROI-AutoUnet[dm_mlo]:18/best_val/accuracy/whole.pth'
    },
    {
        'name': 'AOL-AutoUnet[cm_cc]',
        'dataset': cm_cc,
        'epochs': 50,
        'resnet_size': 18,
        'train': False,
        'type': 'simple-run',
        'batch_size': 64
    },
    {
        'name': 'AOL-AutoUnet[cm_mlo]',
        'dataset': cm_mlo,
        'epochs': 50,
        'resnet_size': 18,
        'train': False,
        'type': 'simple-run',
        'batch_size': 64
    },
    {
        'name': 'AOL-AutoUnet[dm_cc]',
        'dataset': dm_cc,
        'epochs': 50,
        'resnet_size': 18,
        'train': False,
        'type': 'simple-run',
        'batch_size': 64
    },
    {
        'name': 'AOL-AutoUnet[dm_mlo]',
        'dataset': dm_mlo,
        'epochs': 50,
        'resnet_size': 18,
        'train': False,
        'type': 'simple-run',
        'batch_size': 64
    },
    {
        'name': 'AOL-unet-ensemble2',
        'dataset': full,
        'epochs': 50,
        'train': False,
        'type': 'joint-unet-ensemble',
        'batch_size': 64,
        'cm_cc_path': './save:manual:AOL-AutoUnet[cm_cc]:18/best_train/accuracy/whole.pth',
        'cm_mlo_path': './save:manual:AOL-AutoUnet[cm_mlo]:18/best_train/accuracy/whole.pth',
        'dm_cc_path': './save:manual:AOL-AutoUnet[dm_cc]:18/best_train/accuracy/whole.pth',
        'dm_mlo_path': './save:manual:AOL-AutoUnet[dm_mlo]:18/best_train/accuracy/whole.pth'
    },
    {
        'name': 'kfold-2attentionmask[cm_cc]',
        'dataset': cm_cc,
        'epochs': 15,
        'resnet_size': 18,
        'train': False,
        'type': 'kfold-simple-run',
        'batch_size': 64
    },
    {
        'name': 'kfold-2attentionmask[cm_mlo]',
        'dataset': cm_mlo,
        'epochs': 15,
        'resnet_size': 18,
        'train': False,
        'type': 'kfold-simple-run',
        'batch_size': 64
    },
    {
        'name': 'kfold-2attentionmask[dm_cc]',
        'dataset': dm_cc,
        'epochs': 15,
        'resnet_size': 18,
        'train': False,
        'type': 'kfold-simple-run',
        'batch_size': 64
    },
    {
        'name': 'kfold-2attentionmask[dm_mlo]',
        'dataset': dm_mlo,
        'epochs': 15,
        'resnet_size': 18,
        'train': False,
        'type': 'kfold-simple-run',
        'batch_size': 64
    },
    {
        'name': 'Kfold-Semiauto-AOL-Ensemble',
        'dataset': full,
        'epochs': 15,
        'train': False,
        'type': 'kfold-joint-unet-ensemble',
        'batch_size': 128,
        'cm_cc_path': './save:KFold:kfold-2attentionmask[cm_cc]/fold_4/best_val/accuracy/whole.pth',
        'cm_mlo_path': './save:KFold:kfold-2attentionmask[cm_mlo]/fold_4/best_val/accuracy/whole.pth',
        'dm_cc_path': './save:KFold:kfold-2attentionmask[dm_cc]/fold_4/best_val/accuracy/whole.pth',
        'dm_mlo_path': './save:KFold:kfold-2attentionmask[dm_mlo]/fold_4/best_val/accuracy/whole.pth'
    },
    {
        'name': 'kfold-Auto_AOL[cm_cc]',
        'dataset': cm_cc,
        'epochs': 15,
        'resnet_size': 18,
        'train': False,
        'type': 'kfold-simple-run',
        'batch_size': 64
    },
    {
        'name': 'kfold-Auto_AOL[cm_mlo]',
        'dataset': cm_mlo,
        'epochs': 15,
        'resnet_size': 18,
        'train': False,
        'type': 'kfold-simple-run',
        'batch_size': 64
    },
    {
        'name': 'kfold-Auto_AOL[dm_cc]',
        'dataset': dm_cc,
        'epochs': 15,
        'resnet_size': 18,
        'train': False,
        'type': 'kfold-simple-run',
        'batch_size': 64
    },
    {
        'name': 'kfold-Auto_AOL[dm_mlo]',
        'dataset': dm_mlo,
        'epochs': 15,
        'resnet_size': 18,
        'train': False,
        'type': 'kfold-simple-run',
        'batch_size': 64
    },
    {
        'name': 'Kfold-Auto_AOL-Ensemble',
        'dataset': full,
        'epochs': 15,
        'train': False,
        'type': 'kfold-joint-unet-ensemble',
        'batch_size': 64,
        'cm_cc_path': './save:KFold:kfold-Auto_AOL[cm_cc]/fold_4/best_val/accuracy/whole.pth',
        'cm_mlo_path': './save:KFold:kfold-Auto_AOL[cm_mlo]/fold_4/best_val/accuracy/whole.pth',
        'dm_cc_path': './save:KFold:kfold-Auto_AOL[dm_cc]/fold_4/best_val/accuracy/whole.pth',
        'dm_mlo_path': './save:KFold:kfold-Auto_AOL[dm_mlo]/fold_4/best_val/accuracy/whole.pth'
    },
    {
        'name': 'kfold-SemiAuto_ROI[cm_cc]',
        'dataset': cm_cc,
        'epochs': 15,
        'resnet_size': 18,
        'train': False,
        'type': 'kfold-simple-run',
        'batch_size': 64
    },
    {
        'name': 'kfold-SemiAuto_ROI[cm_mlo]',
        'dataset': cm_mlo,
        'epochs': 15,
        'resnet_size': 18,
        'train': False,
        'type': 'kfold-simple-run',
        'batch_size': 64
    },
    {
        'name': 'kfold-SemiAuto_ROI[dm_cc]',
        'dataset': dm_cc,
        'epochs': 15,
        'resnet_size': 18,
        'train': False,
        'type': 'kfold-simple-run',
        'batch_size': 64
    },
    {
        'name': 'kfold-SemiAuto_ROI[dm_mlo]',
        'dataset': dm_mlo,
        'epochs': 15,
        'resnet_size': 18,
        'train': False,
        'type': 'kfold-simple-run',
        'batch_size': 64
    },
    {
        'name': 'Kfold-SemiAuto_ROI-Ensemble',
        'dataset': full,
        'epochs': 15,
        'train': False,
        'type': 'kfold-joint-unet-ensemble',
        'batch_size': 64,
        'cm_cc_path': './save:KFold:kfold-SemiAuto_ROI[cm_cc]/fold_4/best_val/accuracy/whole.pth',
        'cm_mlo_path': './save:KFold:kfold-SemiAuto_ROI[cm_mlo]/fold_4/best_val/accuracy/whole.pth',
        'dm_cc_path': './save:KFold:kfold-SemiAuto_ROI[dm_cc]/fold_4/best_val/accuracy/whole.pth',
        'dm_mlo_path': './save:KFold:kfold-SemiAuto_ROI[dm_mlo]/fold_4/best_val/accuracy/whole.pth'
    },
    {
        'name': 'kfold-Auto_ROI[cm_cc]',
        'dataset': cm_cc,
        'epochs': 15,
        'resnet_size': 18,
        'train': False,
        'type': 'kfold-simple-run',
        'batch_size': 64
    },
    {
        'name': 'kfold-Auto_ROI[cm_mlo]',
        'dataset': cm_mlo,
        'epochs': 15,
        'resnet_size': 18,
        'train': False,
        'type': 'kfold-simple-run',
        'batch_size': 64
    },
    {
        'name': 'kfold-Auto_ROI[dm_cc]',
        'dataset': dm_cc,
        'epochs': 15,
        'resnet_size': 18,
        'train': False,
        'type': 'kfold-simple-run',
        'batch_size': 64
    },
    {
        'name': 'kfold-Auto_ROI[dm_mlo]',
        'dataset': dm_mlo,
        'epochs': 15,
        'resnet_size': 18,
        'train': False,
        'type': 'kfold-simple-run',
        'batch_size': 64
    },
    {
        'name': 'Kfold-Auto_ROI-Ensemble',
        'dataset': full,
        'epochs': 15,
        'train': False,
        'type': 'kfold-joint-unet-ensemble',
        'batch_size': 64,
        'cm_cc_path': './save:KFold:kfold-Auto_ROI[cm_cc]/fold_4/best_val/accuracy/whole.pth',
        'cm_mlo_path': './save:KFold:kfold-Auto_ROI[cm_mlo]/fold_4/best_val/accuracy/whole.pth',
        'dm_cc_path': './save:KFold:kfold-Auto_ROI[dm_cc]/fold_4/best_val/accuracy/whole.pth',
        'dm_mlo_path': './save:KFold:kfold-Auto_ROI[dm_mlo]/fold_4/best_val/accuracy/whole.pth'
    },
    {
        'name': 'KFOLD-joint-unet[cm_cc]',
        'dataset': cm_cc,
        'epochs': 20,
        'train': False,
        'type': 'kfold-joint-unet',
        'batch_size': 32,
        'unet_path': './save:manual:unet[cm_cc]/best_val/dice/whole.pth',
        'bilinear': True
    },
    {
        'name': 'KFOLD-joint-unet[cm_mlo]',
        'dataset': cm_mlo,
        'epochs': 20,
        'train': False,
        'type': 'kfold-joint-unet',
        'batch_size': 32,
        'unet_path': './save:manual:unet[cm_mlo]_500/best_val/dice/whole.pth',
        'bilinear': True
    },
    {
        'name': 'KFOLD-joint-unet[dm_cc]',
        'dataset': dm_cc,
        'epochs': 20,
        'train': False,
        'type': 'kfold-joint-unet',
        'batch_size': 32,
        'unet_path': './save:manual:unet[dm_cc]_500/best_val/dice/whole.pth',
        'bilinear': True
    },
    {
        'name': 'KFOLD-joint-unet[dm_mlo]',
        'dataset': dm_mlo,
        'epochs': 20,
        'train': False,
        'type': 'kfold-joint-unet',
        'batch_size': 32,
        'unet_path': './save:manual:unet[dm_mlo]_500/best_val/dice/whole.pth',
        'bilinear': True
    },
    {
        'name': 'KFOLD-joint-unet-ensemble',
        'dataset': full,
        'epochs': 20,
        'train': False,
        'type': 'kfold-joint-unet-ensemble',
        'batch_size': 32,
        'cm_cc_path': './save:KFold:KFOLD-joint-unet[cm_cc]/fold_4/best_val/accuracy/whole.pth',
        'cm_mlo_path': './save:KFold:KFOLD-joint-unet[cm_mlo]/fold_4/best_val/accuracy/whole.pth',
        'dm_cc_path': './save:KFold:KFOLD-joint-unet[dm_cc]/fold_4/best_val/accuracy/whole.pth',
        'dm_mlo_path': './save:KFold:KFOLD-joint-unet[dm_mlo]/fold_4/best_val/accuracy/whole.pth'
    },
{
        'name': 'KFOLD-joint-unet-pre-segment[cm_cc]',
        'dataset': cm_cc_pre_seg,
        'epochs': 20,
        'train': False,
        'type': 'kfold-joint-unet-pre-segment',
        'batch_size': 32,
        'bilinear': True
    },
{
        'name': 'KFOLD-joint-unet-pre-segment[cm_mlo]',
        'dataset': cm_mlo_pre_seg,
        'epochs': 20,
        'train': False,
        'type': 'kfold-joint-unet-pre-segment',
        'batch_size': 32,
        'bilinear': True
    },
{
        'name': 'KFOLD-joint-unet-pre-segment[dm_cc]',
        'dataset': dm_cc_pre_seg,
        'epochs': 20,
        'train': False,
        'type': 'kfold-joint-unet-pre-segment',
        'batch_size': 32,
        'bilinear': True
    },
{
        'name': 'KFOLD-joint-unet-pre-segment[dm_mlo]',
        'dataset': dm_mlo_pre_seg,
        'epochs': 20,
        'train': False,
        'type': 'kfold-joint-unet-pre-segment',
        'batch_size': 32,
        'bilinear': True
    },
    {
        'name': 'KFOLD-joint-pre-segment-ensemble',
        'dataset': full_pre_segment,
        'epochs': 20,
        'train': False,
        'type': 'kfold-joint-pre-segment-ensemble',
        'batch_size': 32,
        'cm_cc_path': './save:KFold:KFOLD-joint-unet-pre-segment[cm_cc]/fold_4/best_val/accuracy/whole.pth',
        'cm_mlo_path': './save:KFold:KFOLD-joint-unet-pre-segment[cm_mlo]/fold_4/best_val/accuracy/whole.pth',
        'dm_cc_path': './save:KFold:KFOLD-joint-unet-pre-segment[dm_cc]/fold_4/best_val/accuracy/whole.pth',
        'dm_mlo_path': './save:KFold:KFOLD-joint-unet-pre-segment[dm_mlo]/fold_4/best_val/accuracy/whole.pth'
    }
)
