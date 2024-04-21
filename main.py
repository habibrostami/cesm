import torch

from packages.models.Resnet import get_model
from packages.models.SegmentClassifier import SegmentClassifier
from packages.models.SwinUNet import SwinUnet as SwinUNet
from packages.models.UNet import UNet
from packages.models.RefinNet import rf101
from packages.models.JointUNet import JointUNet
from packages.models.JointUNetEnsemble import JointUNetEnsemble
from packages.models.Joint import JointModel
from packages.models.joint_SemiAuto import JointSemiAuto
from packages.models.JointPreSegment import JointPreSegment
from packages.models.JointPreSegmentEnsemble import JointPreSegmentEnsemble


from packages.parameters.Device import device
from packages.parameters.Splits import splits
from packages.runs.SimpleRun import SimpleRun

torch.random.manual_seed(42)

print('Device used:', device)


# ---------------------------------- train 4 residual networks for backbones - and other resnet models requested
for split in splits:
    name = split['name']
    if not split['train']:
        print(f'{name.upper()} skipped...')
        continue
    datasets = split['dataset']
    epochs = split['epochs']
    resnet_size = split.get('resnet_size', 0)
    batch_size = split['batch_size']
    run_type = split['type']

    print('Running for', name.upper())
    print('Run type:', *run_type.split('-'))

    loss_fn = torch.nn.CrossEntropyLoss()

    if run_type == 'joint-unet':
        bilinear = split.get('bilinear', False)
        unet_path = split['unet_path']
        unet_model = UNet(3, 2, bilinear).to(device)
        unet_model.load_all(unet_path)
        model = JointUNet(unet_model, 2)
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.001)
        runner = SimpleRun(datasets['train'], datasets['validation'], datasets['test'], model)
        runner.train(device, loss_fn, optimizer, f'./save:manual:{name}',
                     verbose=False, batch_size=batch_size, epochs=epochs)
        runner.evaluate(device, loss_fn, f'save:manual:{name}')
    if run_type == 'joint-unet-pre-segment':
        model = JointPreSegment(2)
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.001)
        runner = SimpleRun(datasets['train'], datasets['validation'], datasets['test'], model)
        runner.train(device, loss_fn, optimizer, f'./save:manual:{name}',
                     verbose=False, batch_size=batch_size, epochs=epochs)
        runner.evaluate(device, loss_fn, f'save:manual:{name}')
    if run_type == 'joint-unet-ensemble':
        bilinear = split.get('bilinear', False)
        cm_cc_path = split['cm_cc_path']
        cm_mlo_path = split['cm_mlo_path']
        dm_cc_path = split['dm_cc_path']
        dm_mlo_path = split['dm_mlo_path']

        print('Creating Model...')
        print('1. Loading Backbones...')
        print('\r[1/4] CM CC', end='')
        # cm_cc = JointUNet()
        cm_cc = JointModel()
        cm_cc.load_all(cm_cc_path)
        print('\r[2/4] CM MLO', end='')
        # cm_mlo = JointUNet()
        cm_mlo = JointModel()
        cm_mlo.load_all(cm_mlo_path)
        print('\r[3/4] DM CC', end='')
        # dm_cc = JointUNet()
        dm_cc = JointModel()
        dm_cc.load_all(dm_cc_path)
        print('\r[4/4] DM MLO')
        # dm_mlo = JointUNet()
        dm_mlo = JointModel()
        dm_mlo.load_all(dm_mlo_path)

        print('2. Creating Ensemble Model...')
        model = JointUNetEnsemble(cm_cc, cm_mlo, dm_cc, dm_mlo, 2)
        print('Model Created')
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.001)
        runner = SimpleRun(datasets['train'], datasets['validation'], datasets['test'], model)
        runner.train(device, loss_fn, optimizer, f'./save:manual:{name}',
                     verbose=False, batch_size=batch_size, epochs=epochs)
        runner.evaluate(device, loss_fn, f'save:manual:{name}')
    if run_type == 'joint-pre-segment-ensemble':
        cm_cc_path = split['cm_cc_path']
        cm_mlo_path = split['cm_mlo_path']
        dm_cc_path = split['dm_cc_path']
        dm_mlo_path = split['dm_mlo_path']

        print('Creating Model...')
        print('1. Loading Backbones...')
        print('\r[1/4] CM CC', end='')
        cm_cc = JointPreSegment()
        cm_cc.load_all(cm_cc_path)
        print('\r[2/4] CM MLO', end='')
        cm_mlo = JointPreSegment()
        cm_mlo.load_all(cm_mlo_path)
        print('\r[3/4] DM CC', end='')
        dm_cc = JointPreSegment()
        dm_cc.load_all(dm_cc_path)
        print('\r[4/4] DM MLO')
        dm_mlo = JointPreSegment()
        dm_mlo.load_all(dm_mlo_path)

        print('2. Creating Ensemble Model...')
        model = JointPreSegmentEnsemble(cm_cc, cm_mlo, dm_cc, dm_mlo, 2)
        print('Model Created')
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.001)
        runner = SimpleRun(datasets['train'], datasets['validation'], datasets['test'], model)
        runner.train(device, loss_fn, optimizer, f'./save:manual:{name}',
                     verbose=False, batch_size=batch_size, epochs=epochs)
        runner.evaluate(device, loss_fn, f'save:manual:{name}')
    if run_type == 'swin-unet':
        model_path = split['model_path']
        model = SwinUNet()
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.001)
        model = model.to(device)
        model.load_all(model_path)
        runner = SimpleRun(datasets['train'], datasets['validation'], datasets['test'], model, is_segmentation=True)
        # runner.train(device, loss_fn, optimizer, f'./save:manual:{name}',
        #              verbose=False, batch_size=batch_size, epochs=epochs)
        # runner.evaluate(device, loss_fn, f'save:manual:{name}', model_path)
        runner.evaluate(device, loss_fn, f'save:manual:{name}',
                        f'/mnt/2T/BreastCancerAll/SwinUNet/out/epoch_149.pth')
    if run_type == 'segment_classifier':
        segment_path = split.get('segment_path', None)
        bilinear = split.get('bilinear', False)
        segment_model = None
        if segment_path:
            segment_model = UNet(3, 2, bilinear)
            segment_model.load_all(segment_path)
        model = SegmentClassifier(segment_model)
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.001)

        runner = SimpleRun(datasets['train'], datasets['validation'], datasets['test'], model)

        runner.train(device, loss_fn, optimizer, f'./save:manual:{name}',
                     verbose=False, batch_size=batch_size, epochs=epochs)
        runner.evaluate(device, loss_fn, f'save:manual:{name}',
                        f'save:manual:{name}/best_val/accuracy/whole.pth')
    if run_type == 'u-net':
        bilinear = split.get('bilinear', False)
        model = UNet(3, 2, bilinear).to(device)
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.001)

        runner = SimpleRun(datasets['train'], datasets['validation'], datasets['test'], model, is_segmentation=True)

        runner.train(device, loss_fn, optimizer, f'./save:manual:{name}',
                     verbose=False, batch_size=batch_size, epochs=epochs)
        runner.evaluate(device, loss_fn, f'save:manual:{name}',
                        f'save:manual:{name}/best_val/dice/whole.pth')
    if run_type == 'refine-net':
        model = rf101(num_classes=2)
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.001)
        runner = SimpleRun(datasets['train'], datasets['validation'], datasets['test'], model, is_segmentation=True)

        runner.train(device, loss_fn, optimizer, f'./save:manual:{name}',
                     verbose=False, batch_size=batch_size, epochs=epochs)
        runner.evaluate(device, loss_fn, f'save:manual:{name}',
                        f'save:manual:{name}/best_val/dice/whole.pth')

    if run_type == 'simple-run':
        # model = get_model(resnet_size, 2).to(device)
        model = JointModel(num_labels=2)
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.001)

        runner = SimpleRun(datasets['train'], datasets['validation'], datasets['test'], model)

        runner.train(device, loss_fn, optimizer, f'./save:manual:{name}:{resnet_size}',
                     verbose=False, batch_size=batch_size, epochs=epochs)
        runner.evaluate(device, loss_fn, f'save:manual:{name}:{resnet_size}',
                        f'save:manual:{name}:{resnet_size}/best_val/accuracy/whole.pth')

    # if run_type == 'Joint-SemiAuto':
    #     # model = get_model(resnet_size, 2).to(device)
    #     model = JointSemiAuto(num_labels=2)
    #     optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.001)
    #
    #     runner = SimpleRun(datasets['train'], datasets['validation'], datasets['test'], model)
    #
    #     runner.train(device, loss_fn, optimizer, f'./save:manual:{name}:{resnet_size}',
    #                  verbose=False, batch_size=batch_size, epochs=epochs)
    #     runner.evaluate(device, loss_fn, f'save:manual:{name}:{resnet_size}',
    #                     f'save:manual:{name}:{resnet_size}/best_val/accuracy/whole.pth')
