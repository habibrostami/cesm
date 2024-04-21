from packages.Dataset import (OneImage, TwoImages, FourImages, SegmentDataset, SegmentTwoImage, PreSegmentDataset,
                              FourImagesPreSegment)
from packages.parameters.Device import device
from packages.parameters.Transform import transform

cm_cc = {
    'train': OneImage('./train_data.csv', 'cm', 'cc', transform, device),
    'validation': OneImage('./validation_data.csv', 'cm', 'cc', transform, device),
    'test': OneImage('./test_data.csv', 'cm', 'cc', transform, device)
}
cm_mlo = {
    'train': OneImage('./train_data.csv', 'cm', 'mlo', transform, device),
    'validation': OneImage('./validation_data.csv', 'cm', 'mlo', transform, device),
    'test': OneImage('./test_data.csv', 'cm', 'mlo', transform, device)
}
dm_cc = {
    'train': OneImage('./train_data.csv', 'dm', 'cc', transform, device),
    'validation': OneImage('./validation_data.csv', 'dm', 'cc', transform, device),
    'test': OneImage('./test_data.csv', 'dm', 'cc', transform, device)
}
dm_mlo = {
    'train': OneImage('./train_augment.csv', 'dm', 'mlo', transform, device),
    'validation': OneImage('./validation_data.csv', 'dm', 'mlo', transform, device),
    'test': OneImage('./test_data.csv', 'dm', 'mlo', transform, device)
}
cm = {
    'train': OneImage('./train_data.csv', 'cm', 'both', transform, device),
    'validation': OneImage('./validation_data.csv', 'cm', 'both', transform, device),
    'test': OneImage('./test_data.csv', 'cm', 'both', transform, device)
}
dm = {
    'train': OneImage('./train_data.csv', 'dm', 'both', transform, device),
    'validation': OneImage('./validation_data.csv', 'dm', 'both', transform, device),
    'test': OneImage('./test_data.csv', 'dm', 'both', transform, device)
}
cc = {
    'train': TwoImages('./train_data.csv', 'cc', transform=transform, device=device),
    'validation': TwoImages('./validation_data.csv', 'cc', transform=transform, device=device),
    'test': TwoImages('./test_data.csv', 'cc', transform=transform, device=device)
}
mlo = {
    'train': TwoImages('./train_data.csv', 'mlo', transform=transform, device=device),
    'validation': TwoImages('./validation_data.csv', 'mlo', transform=transform, device=device),
    'test': TwoImages('./test_data.csv', 'both', transform=transform, device=device)
}

cm_cc_seg = {
    'train': SegmentDataset('./train_segment.csv', 'cm', 'cc', './data/masks',
                            transform, device),
    'validation': SegmentDataset('./validation_segment.csv', 'cm', 'cc',
                                 './data/masks', transform, device),
    'test': SegmentDataset('./test_segment.csv', 'cm', 'cc', './data/masks',
                           transform, device)
}

cm_mlo_seg = {
    'train': SegmentDataset('./train_segment.csv', 'cm', 'mlo', './data/masks',
                            transform, device),
    'validation': SegmentDataset('./validation_segment.csv', 'cm', 'mlo', './data/masks',
                                 transform, device),
    'test': SegmentDataset('./test_segment.csv', 'cm', 'mlo', './data/masks',
                           transform, device)
}

dm_cc_seg = {
    'train': SegmentDataset('./train_segment.csv', 'dm', 'cc', './data/masks',
                            transform, device),
    'validation': SegmentDataset('./validation_segment.csv', 'dm', 'cc', './data/masks',
                                 transform, device),
    'test': SegmentDataset('./test_segment.csv', 'dm', 'cc', './data/masks',
                           transform, device)
}

dm_mlo_seg = {
    'train': SegmentDataset('./train_segment.csv', 'dm', 'mlo', './data/masks',
                            transform, device),
    'validation': SegmentDataset('./validation_segment.csv', 'dm', 'mlo', './data/masks',
                                 transform, device),
    'test': SegmentDataset('./test_segment.csv', 'dm', 'mlo', './data/masks',
                           transform, device)
}
cm_cc_pre_seg = {
    'train': PreSegmentDataset('./train_segment.csv', 'cm', 'cc', './data/masks',
                               transform, device),
    'validation': PreSegmentDataset('./validation_segment.csv', 'cm', 'cc',
                                    './data/masks', transform, device),
    'test': PreSegmentDataset('./test_segment.csv', 'cm', 'cc', './data/masks',
                              transform, device)
}

cm_mlo_pre_seg = {
    'train': PreSegmentDataset('./train_segment.csv', 'cm', 'mlo', './data/masks',
                               transform, device),
    'validation': PreSegmentDataset('./validation_segment.csv', 'cm', 'mlo', './data/masks',
                                    transform, device),
    'test': PreSegmentDataset('./test_segment.csv', 'cm', 'mlo', './data/masks',
                              transform, device)
}

dm_cc_pre_seg = {
    'train': PreSegmentDataset('./train_segment.csv', 'dm', 'cc', './data/masks',
                               transform, device),
    'validation': PreSegmentDataset('./validation_segment.csv', 'dm', 'cc', './data/masks',
                                    transform, device),
    'test': PreSegmentDataset('./test_segment.csv', 'dm', 'cc', './data/masks',
                              transform, device)
}

dm_mlo_pre_seg = {
    'train': PreSegmentDataset('./train_segment.csv', 'dm', 'mlo', './data/masks',
                               transform, device),
    'validation': PreSegmentDataset('./validation_segment.csv', 'dm', 'mlo', './data/masks',
                                    transform, device),
    'test': PreSegmentDataset('./test_segment.csv', 'dm', 'mlo', './data/masks',
                              transform, device)
}

dm_seg = {
    'train': SegmentTwoImage('./train_segment.csv', 'dm', './data/masks',
                             transform, device),
    'validation': SegmentTwoImage('./validation_segment.csv', 'dm', './data/masks',
                                  transform, device),
    'test': SegmentTwoImage('./test_segment.csv', 'dm', './data/masks',
                            transform, device)
}
cm_seg = {
    'train': SegmentTwoImage('./train_segment.csv', 'cm', './data/masks',
                             transform, device),
    'validation': SegmentTwoImage('./validation_segment.csv', 'cm', './data/masks',
                                  transform, device),
    'test': SegmentTwoImage('./test_segment.csv', 'cm', './data/masks',
                            transform, device)
}

full = {
    'train': FourImages('./train_data.csv', transform, device),
    'validation': FourImages('./validation_data.csv', transform, device),
    'test': FourImages('./test_data.csv', transform, device)
}

full_pre_segment = {
    'train': FourImagesPreSegment('./train_segment.csv', './data/masks/', transform, device),
    'validation': FourImagesPreSegment('./validation_segment.csv', './data/masks/', transform, device),
    'test': FourImagesPreSegment('./test_segment.csv', './data/masks/', transform, device)
}
