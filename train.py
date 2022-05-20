import os
import argparse
import torch, gc
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from datasets import *
from models.denoise import *
from models.losses import *
from models.utils import AverageMeter
from lfs_core.loss_func_search import LossFuncSearch
from utils.misc import *

# Arguments
parser = argparse.ArgumentParser()
# Dataset and pre-processing
parser.add_argument('--noise', default=0.02, type=float)
parser.add_argument('--noise_high', default=0.06, type=float, help='-1 for fixed noise level.')
parser.add_argument('--aug_scale', action='store_true', help='Enable scaling augmentation.')
# parser.add_argument('--train_dataset',
#                     default='./data/patches_20k_1024.h5;./data/patches_30k_1024.h5;./data/patches_50k_1024.h5;./data/patches_80k_1024.h5',
#                     type=str)
# parser.add_argument('--valid_dataset', default='./data/patches_10k_1024.h5', type=str)
parser.add_argument('--train_dataset', default='./data/trainingset_data_patches.h5', type=str)
parser.add_argument('--valid_dataset', default='./data/validationset_data_patches.h5', type=str)
parser.add_argument('--subset_size', default=7000, type=int, help='-1 for unlimited.')
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--aug_rotate', type=eval, default=True, choices=[True, False])

# Network
parser.add_argument('--net', type=str, default='DenoiseNet')
parser.add_argument('--gpool_mlp', action='store_true',
                    help='Use MLP instead of single linear layer in the GPool module.')
parser.add_argument('--knn', type=str, default='8,16,24')
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--static_graph', action='store_true',
                    help='Use static graph convolution instead of dynamic graph (DGCNN).')
parser.add_argument('--random_mesh', action='store_true',
                    help='Use random mesh instead of regular mesh in the folding layer.')
parser.add_argument('--random_pool', action='store_true',
                    help='Use random pooling layer instead of differentiable pooling layer.')
parser.add_argument('--no_prefilter', action='store_true', help='Disable prefiltering.')
# Loss
parser.add_argument('--loss_rec', default='emd', type=str, help='Reconstruction loss.')
parser.add_argument('--loss_ds', default='cd', type=str, help='Downsample adjustment loss.')

## Optimizer and scheduler
parser.add_argument('--learning_rate', default=5e-4, type=float)
parser.add_argument('--sched_patience', default=10, type=int)
parser.add_argument('--sched_factor', default=0.5, type=float)
parser.add_argument('--min_lr', default=1e-5, type=float)
parser.add_argument('--max_grad_norm', type=float, default=float("inf"))
## Training
parser.add_argument('--do_search', type=int, default=1, help='if 1, do loss search otherwise perform random softmax')
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--log_root', type=str, default='./logs')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_iters', type=int, default=80000)
parser.add_argument('--val_freq', type=int, default=2000)
parser.add_argument('--best_emd', type=int, default=1000)
parser.add_argument('--best_cd', type=int, default=1000)
parser.add_argument('--best_mse', type=int, default=10000)
args = parser.parse_args()
seed_all(args.seed)

# Logging
if args.logging:
    log_dir = get_new_log_dir(args.log_root)
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
    log_hyperparams(writer, log_dir, args)
else:
    logger = get_logger('train', None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()
logger.info(args)

# Datasets and loaders
logger.info('Loading datasets')
train_dset = get_train_dataset(args)
val_dset = get_valid_dataset(args)
train_iter = get_data_iterator(DataLoader(train_dset, batch_size=args.batch_size, shuffle=True))

# Model
logger.info('Building model...')
model = PointCloudDenoising(args).to(args.device)
logger.info(repr(model))

# 损失函数 create LFS
lfs = LossFuncSearch(True if args.do_search == 1 else False)
lfs.set_model(model)

# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.sched_patience,
                                                       factor=args.sched_factor, min_lr=args.min_lr)


# Train, validate and test
def train(it):
    # Load data
    batch = next(train_iter)
    input = batch['pos'].to(args.device)  # torch.Size([4, 1024, 3])
    denoised = model(input).to(args.device)
    noiseless = batch['clean'].to(args.device)

    # Reset grad and model state
    optimizer.zero_grad()
    model.train()

    # Forward
    loss = lfs.get_loss(denoised, noiseless, input)

    # Backward and optimize
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    # Logging
    logger.info('[Train] Iter %04d | Loss %.6f | Grad %.6f' % (
        it, loss.item(), orig_grad_norm.item(),
    ))
    writer.add_scalar('loss_train/loss', loss, it)
    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], it)
    writer.add_scalar('grad_norm', orig_grad_norm, it)
    writer.flush()


def validate(it):
    model.eval()
    CD = ChamferLoss()
    EMD = EMDLoss()
    MSE = MSELoss()
    chamfer, emd, mse = 0.0, 0.0, 0.0
    with torch.no_grad():
        for key in ['noisy_0.01', 'noisy_0.03', 'noisy_0.08']:
            emdloss = AverageMeter()
            cdloss = AverageMeter()
            mseloss = AverageMeter()
            for batch_idx, (batch) in enumerate(val_dset):
                input = batch[key].to(args.device)  # torch.Size([batchsize, 1024, 3])
                denoised = model.forward(input).to(args.device)
                noiseless = batch['clean'].to(args.device)

                cdloss.update(CD(preds=denoised, gts=noiseless).reshape(1), len(input))
                emdloss.update(EMD(preds=denoised, gts=noiseless).reshape(1), len(input))
                mseloss.update(MSE(preds=denoised, gts=noiseless).reshape(1), len(input))

            chamfer += (cdloss.avg)
            emd += (emdloss.avg)
            mse += (mseloss.avg)
            writer.add_scalar('loss_val/' + key + '_cd_loss', cdloss.avg, it)
            writer.add_scalar('loss_val/' + key + '_emd_loss', emdloss.avg, it)
            writer.add_scalar('loss_val/' + key + '_MSE', mseloss.avg, it)
            writer.flush()
    avg_emd, avg_cd, avg_mse = emd / 3, chamfer / 3, mse / 3
    if avg_emd < args.best_emd:
        args.best_emd = avg_emd
    if avg_cd < args.best_cd:
        args.best_cd = avg_cd
    if avg_mse < args.best_mse:
        args.best_mse = avg_mse
    logger.info(
        '[Val] Iter %04d | best CD %.6f best EMD %.6f best MSE %.6f' % (it, args.best_cd, args.best_emd, args.best_mse))
    scheduler.step(avg_cd)
    model.train()
    return chamfer


if __name__ == '__main__':
    # Main loop
    logger.info('Start training...')
    for it in range(1, args.max_iters + 1):
        train(it)
        gc.collect()
        torch.cuda.empty_cache()
        if it % args.val_freq == 0 or it == args.max_iters:
            lfs.set_loss_parameters(it)
            logger.info('[LFS] Iter %04d | a_cd %.6f a_emd %.6f a_repulsion %.6f' % (it, lfs.a[0], lfs.a[1], lfs.a[2]))
            logger.info(
                '[LFS] Iter %04d | mu_cd %.6f mu_emd %.6f mu_repulsion %.6f' % (it, lfs.mu[0], lfs.mu[1], lfs.mu[2]))
            writer.add_scalar('a/cd', lfs.a[0], it)
            writer.add_scalar('a/emd', lfs.a[1], it)
            writer.add_scalar('a/repulsion', lfs.a[2], it)
            writer.add_scalar('mu/cd', lfs.mu[0], it)
            writer.add_scalar('mu/emd', lfs.mu[1], it)
            writer.add_scalar('mu/repulsion', lfs.mu[2], it)
            cd_loss = validate(it)
            gc.collect()
            torch.cuda.empty_cache()
            opt_states = {
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            if args.do_search == 1:
                lfs.update_lfs(cd_loss)
            ckpt_mgr.save(model, args, cd_loss, opt_states, step=it)
            # ckpt_mgr.save(model, args, 0, opt_states, step=it)
