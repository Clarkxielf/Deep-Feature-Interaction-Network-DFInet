"""Train RegistrationNet"""


import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from collections import defaultdict
from typing import Dict, List
from tensorboardX import SummaryWriter
from tqdm import tqdm
import time

from DFInet_2.src.arguments import RegistrationNet_arguments
from DFInet_2.src.common.colors import BLUE, ORANGE
from DFInet_2.src.common.misc import prepare_logger
from DFInet_2.src.common.torch import dict_all_to_device, CheckPointManager, TorchDebugger
from DFInet_2.src.common import se3
from DFInet_2.src.data_loader.datasets import get_train_datasets
from DFInet_2.src.eval import print_metrics
from DFInet_2.src.models.rpmnet import get_model


# Set up arguments and logging
parser = RegistrationNet_arguments()
_args = parser.parse_args()
_logger, _log_path = prepare_logger(_args)
if _args.gpu >= 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(_args.gpu)
    _device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
else:
    _device = torch.device('cpu')


def main():
    train_set, val_set = get_train_datasets(_args)
    run(train_set, val_set)


def compute_losses(data: Dict, pred_transforms: List, endpoints: Dict, loss_type: str = 'mae', reduction: str = 'mean') -> Dict:
    """Compute losses

    Args:
        data: Current mini-batch data
        pred_transforms: Predicted transform, to compute main registration loss
        loss_type: Registration loss type, either 'mae' (Mean absolute error) or 'mse'
        reduction: Either 'mean' or 'none'. Use 'none' to accumulate losses outside
                   (useful for accumulating losses for entire validation dataset)

    Returns:
        losses: Dict containing various fields. Total loss to be optimized is in losses['total']

    """

    losses = {}
    num_iter = len(pred_transforms)

    # Compute losses
    if reduction.lower() != 'none':
        gt_src_transformed = se3.transform(data['transform_gt'], data['points_src'][..., :2])

        if loss_type == 'mse':
        # MSE loss to the groundtruth (does not take into account possible symmetries)
            criterion = nn.MSELoss(reduction=reduction)
            for i in range(num_iter):
                pred_src_transformed = se3.transform(pred_transforms[i], data['points_src'][..., :2])
                losses['mse_{}'.format(i)] = criterion(pred_src_transformed, gt_src_transformed)

        elif loss_type == 'mae':
            # MSE loss to the groundtruth (does not take into account possible symmetries)
            criterion = nn.L1Loss(reduction=reduction)
            for i in range(num_iter):
                pred_src_transformed = se3.transform(pred_transforms[i], data['points_src'][..., :2])
                losses['mae_{}'.format(i)] = criterion(pred_src_transformed, gt_src_transformed)

        elif loss_type == 'rmse':
            criterion = nn.MSELoss(reduction=reduction)
            for i in range(num_iter):
                pred_src_transformed = se3.transform(pred_transforms[i], data['points_src'][..., :2])
                losses['rmse_{}'.format(i)] = torch.sqrt(criterion(pred_src_transformed, gt_src_transformed))

        # Penalize outliers
        for i in range(num_iter):
            ref_outliers_strength = (1.0 - torch.sum(endpoints['perm_matrices'][i], dim=1)) * _args.wt_inliers
            src_outliers_strength = (1.0 - torch.sum(endpoints['perm_matrices'][i], dim=2)) * _args.wt_inliers
            losses['outlier_{}'.format(i)] = torch.mean(ref_outliers_strength) + torch.mean(src_outliers_strength)


        discount_factor = 0.5  # Early iterations will be discounted
        total_losses = []
        for k in losses:
            discount = discount_factor ** (num_iter - int(k[k.rfind('_')+1:]) - 1)
            total_losses.append(losses[k] * discount)
        losses['total'] = torch.sum(torch.stack(total_losses), dim=0)

    # Evaluation Metrics
    else:
        gt_src_transformed = se3.transform(data['transform_gt'], data['points_src'][..., :2])
        gt_Rot_mat = data['transform_gt'][:, :, :2]
        gt_Translation = data['transform_gt'][:, :, 2]

        criterion = nn.MSELoss(reduction=reduction)
        for i in range(num_iter):
            pred_src_transformed = se3.transform(pred_transforms[i], data['points_src'][..., :2])
            pred_Rot_mat = pred_transforms[i][:, :, :2]
            pred_Translation = pred_transforms[i][:, :, 2]

            losses['mse_{}'.format(i)] = torch.mean(criterion(pred_src_transformed, gt_src_transformed),
                                                dim=[-1, -2])
            losses['r_mse_{}'.format(i)] = torch.mean(criterion(pred_Rot_mat, gt_Rot_mat),
                                                dim=[-1, -2])
            losses['t_mse_{}'.format(i)] = torch.mean(criterion(pred_Translation, gt_Translation),
                                                dim=[-1])

        criterion = nn.L1Loss(reduction=reduction)
        for i in range(num_iter):
            pred_src_transformed = se3.transform(pred_transforms[i], data['points_src'][..., :2])
            pred_Rot_mat = pred_transforms[i][:, :, :2]
            pred_Translation = pred_transforms[i][:, :, 2]

            losses['mae_{}'.format(i)] = torch.mean(criterion(pred_src_transformed, gt_src_transformed),
                                                    dim=[-1, -2])
            losses['r_mae_{}'.format(i)] = torch.mean(criterion(pred_Rot_mat, gt_Rot_mat),
                                                    dim=[-1, -2])
            losses['t_mae_{}'.format(i)] = torch.mean(criterion(pred_Translation, gt_Translation),
                                                    dim=[-1])

    return losses


def save_summaries(writer: SummaryWriter, data: Dict, predicted: List, endpoints: Dict = None, losses: Dict = None, step: int = 0):
    """Save tensorboard summaries"""

    subset = [0, 1]

    with torch.no_grad():
        # Save clouds

        points_src = data['points_src'][subset, ..., :2]
        # points_ref = data['points_ref'][subset, ..., :2]

        gt_src_transformed = se3.transform(data['transform_gt'][subset, ...], points_src)

        colors = torch.from_numpy(
            np.concatenate([np.tile(ORANGE, (*points_src.shape[:2], 1)),
                            np.tile(BLUE, (*gt_src_transformed.shape[:2], 1))], axis=1))

        # iters_to_save = [0, len(predicted)-1] if len(predicted) > 1 else [0]
        iters_to_save = list(range(len(predicted)))


        # Save point cloud at iter0, iter1 and after last iter
        concat_cloud_input = torch.cat((points_src, gt_src_transformed), dim=1)
        concat_cloud_input = torch.cat((concat_cloud_input, torch.zeros(*concat_cloud_input.shape[:2], 1).cuda()), dim=2)

        writer.add_mesh('iter_0', vertices=concat_cloud_input, colors=colors, global_step=step)

        # gt_src_transformed = points_ref
        for i_iter in iters_to_save:
            src_transformed_first = se3.transform(predicted[i_iter][subset, ...], points_src)

            concat_cloud_first = torch.cat((src_transformed_first, gt_src_transformed), dim=1)
            concat_cloud_first = torch.cat((concat_cloud_first, torch.zeros(*concat_cloud_first.shape[:2], 1).cuda()), dim=2)

            writer.add_mesh('iter_{}'.format(i_iter+1), vertices=concat_cloud_first, colors=colors, global_step=step)

        # if endpoints is not None:
        #     if 'perm_matrices' in endpoints:
        #         for i_iter in range(len(endpoints['perm_matrices'])):
        #             src_weights = torch.sum(endpoints['perm_matrices'][i_iter], dim=2)
        #             ref_weights = torch.sum(endpoints['perm_matrices'][i_iter], dim=1)
        #             writer.add_histogram('src_weights_{}'.format(i_iter), src_weights, global_step=step)
        #             writer.add_histogram('ref_weights_{}'.format(i_iter), ref_weights, global_step=step)

        # Write losses
        if losses is not None:
            for l in losses:
                writer.add_scalar('losses/{}'.format(l), losses[l], step)

        writer.flush()


def validate(data_loader, model: torch.nn.Module, summary_writer: SummaryWriter, step: int):
    """Perform a single validation run, and saves results into tensorboard summaries"""

    _logger.info('Starting validation run...')
    start_time = time.time()
    with torch.no_grad():
        all_val_losses = defaultdict(list)
        for val_data in data_loader:
            dict_all_to_device(val_data, _device)
            pred_test_transforms, endpoints = model(val_data, _args.num_reg_iter)
            val_losses = compute_losses(val_data, pred_test_transforms, endpoints, loss_type=_args.loss_type, reduction='none')

            for k in val_losses:
                all_val_losses[k].append(val_losses[k])

        all_val_losses = {k: torch.cat(all_val_losses[k]) for k in all_val_losses}
        mean_val_losses = {k: torch.mean(all_val_losses[k]) for k in all_val_losses}

    total_time = time.time()-start_time
    total_time = total_time/len(data_loader.dataset)
    _logger.info('Total inference time: {}s'.format(total_time))

    # Rerun on random and worst data instances and save to summary
    rand_idx = random.randint(0, all_val_losses['{}_{}'.format(_args.loss_type, _args.num_reg_iter - 1)].shape[0] - 1)
    worst_idx = torch.argmax(all_val_losses['{}_{}'.format(_args.loss_type, _args.num_reg_iter - 1)]).cpu().item()
    indices_to_rerun = [rand_idx, worst_idx]
    data_to_rerun = defaultdict(list)
    for i in indices_to_rerun:
        cur = data_loader.dataset[i]
        for k in cur:
            data_to_rerun[k].append(cur[k])
    for k in data_to_rerun:
        data_to_rerun[k] = torch.from_numpy(np.stack(data_to_rerun[k], axis=0))
    dict_all_to_device(data_to_rerun, _device)
    pred_transforms, endpoints = model(data_to_rerun, _args.num_reg_iter)


    mse_losses_by_iteration = torch.stack([mean_val_losses['mse_{}'.format(k)] for k in range(_args.num_reg_iter)]).cpu().numpy()
    mae_losses_by_iteration = torch.stack([mean_val_losses['mae_{}'.format(k)] for k in range(_args.num_reg_iter)]).cpu().numpy()
    rmse_losses_by_iteration = torch.stack([torch.sqrt(mean_val_losses['mse_{}'.format(k)]) for k in range(_args.num_reg_iter)]).cpu().numpy()

    r_mse_losses_by_iteration = torch.stack([mean_val_losses['r_mse_{}'.format(k)] for k in range(_args.num_reg_iter)]).cpu().numpy()
    r_mae_losses_by_iteration = torch.stack([mean_val_losses['r_mae_{}'.format(k)] for k in range(_args.num_reg_iter)]).cpu().numpy()
    r_rmse_losses_by_iteration = torch.stack([torch.sqrt(mean_val_losses['r_mse_{}'.format(k)]) for k in range(_args.num_reg_iter)]).cpu().numpy()

    t_mse_losses_by_iteration = torch.stack([mean_val_losses['t_mse_{}'.format(k)] for k in range(_args.num_reg_iter)]).cpu().numpy()
    t_mae_losses_by_iteration = torch.stack([mean_val_losses['t_mae_{}'.format(k)] for k in range(_args.num_reg_iter)]).cpu().numpy()
    t_rmse_losses_by_iteration = torch.stack([torch.sqrt(mean_val_losses['t_mse_{}'.format(k)]) for k in range(_args.num_reg_iter)]).cpu().numpy()

    print_metrics(_logger,
                  mse_losses_by_iteration, mae_losses_by_iteration, rmse_losses_by_iteration,
                  r_mse_losses_by_iteration, r_mae_losses_by_iteration, r_rmse_losses_by_iteration,
                  t_mse_losses_by_iteration, t_mae_losses_by_iteration, t_rmse_losses_by_iteration,
                  'Validation results')


    save_summaries(summary_writer, data=data_to_rerun, predicted=pred_transforms, endpoints=endpoints, losses=mean_val_losses, step=step)

    # score = -mean_val_losses['{}_{}'.format(_args.loss_type, _args.num_reg_iter - 1)]
    score = -min([mean_val_losses['t_mae_{}'.format(i)] for i in range(_args.num_reg_iter)])
    return score


def run(train_set, val_set):
    """Main train/val loop"""

    _logger.debug('Trainer (PID=%d), %s', os.getpid(), _args)

    model = get_model(_args)
    model.to(_device)
    global_step = 0

    # dataloaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=_args.train_batch_size, shuffle=True, num_workers=_args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=_args.val_batch_size, shuffle=False, num_workers=_args.num_workers)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=_args.lr)

    # Summary writer and Checkpoint manager
    train_writer = SummaryWriter(os.path.join(_log_path, 'train'), flush_secs=10)
    val_writer = SummaryWriter(os.path.join(_log_path, 'val'), flush_secs=10)
    saver = CheckPointManager(os.path.join(_log_path, 'ckpt', 'model'), keep_checkpoint_every_n_hours=0.5)
    if _args.resume is not None:
        global_step = saver.load(_args.resume, model, optimizer)

    # trainings
    torch.autograd.set_detect_anomaly(_args.debug)
    model.train()

    steps_per_epoch = len(train_loader)
    if _args.summary_every < 0:
        _args.summary_every = abs(_args.summary_every) * steps_per_epoch
    if _args.validate_every < 0:
        _args.validate_every = abs(_args.validate_every) * steps_per_epoch

    for epoch in range(0, _args.epochs):
        _logger.info('Begin epoch {} (steps {} - {})'.format(epoch, global_step, global_step + len(train_loader)))
        tbar = tqdm(total=len(train_loader), ncols=100)
        for train_data in train_loader:

            global_step += 1
            #
            # optimizer.zero_grad()
            #
            # # Forward through neural network
            # dict_all_to_device(train_data, _device)
            # pred_transforms, endpoints = model(train_data, _args.num_train_reg_iter)  # Use less iter during training
            #
            # # Compute loss, and optimize
            # train_losses = compute_losses(train_data, pred_transforms, endpoints, loss_type=_args.loss_type, reduction='mean')
            # if _args.debug:
            #     with TorchDebugger():
            #         train_losses['total'].backward()
            # else:
            #     train_losses['total'].backward()
            # optimizer.step()
            #
            # tbar.set_description('Loss:{:.3g}'.format(train_losses['total']))
            # tbar.update(1)
            #
            # if global_step % _args.summary_every == 0:  # Save tensorboard logs
            #
            #     save_summaries(train_writer, data=train_data, predicted=pred_transforms, endpoints=endpoints, losses=train_losses, step=global_step)

            if global_step % _args.validate_every == 0:  # Validation loop. Also saves checkpoints
                model.eval()
                val_score = validate(val_loader, model, val_writer, global_step)
                saver.save(model, optimizer, step=global_step, score=val_score)
                model.train()

        tbar.close()

    _logger.info('Ending training. Number of steps = {}.'.format(global_step))


if __name__ == '__main__':
    main()
