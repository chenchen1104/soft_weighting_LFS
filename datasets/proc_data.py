from .dataset import *
from .transform import *
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_train_dataset(hparams):
    # noisifier
    noise_l = hparams.noise
    noise_h = hparams.noise_high
    if noise_h > noise_l:
        noisifier = AddRandomNoise(std_range=[noise_l, noise_h])
        print('[INFO] Using random noise level.')
    else:
        noisifier = AddNoise(std=hparams.noise)

    # Scaling augmentation
    if hparams.aug_scale:
        print('[INFO] Scaling augmentation ENABLED.')
        scaler = RandomScale([0.8, 1.2], attr=['pos',
                                               'clean'])  # anisotropic scaling doesn't change the direction of normal vectors.
    else:
        print('[INFO] Scaling augmentation DISABLED.')
        scaler = IdentityTransform()

    t = transforms.Compose([
        noisifier,
    ])
    print('[INFO] Using multiple datasets for training.')
    dataset = H5Dataset(hparams.train_dataset, 'data', batch_size=hparams.batch_size, transform=t)

    return dataset


def get_valid_dataset(hparams):
    noisifier = AddNoiseForEval([0.01, 0.03, 0.08])
    t = transforms.Compose([
        noisifier,
    ])
    dataset_path = hparams.valid_dataset
    return DataLoader(
        H5Dataset(dataset_path, 'data', batch_size=hparams.batch_size, transform=t),
        batch_size=hparams.batch_size, shuffle=False
    )
