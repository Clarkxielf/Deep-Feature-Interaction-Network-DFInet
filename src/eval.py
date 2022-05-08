"""Evaluate RegistrationNet

    --resume [path-to-model.pth]

"""


import os
import time
import numpy as np
import torch
import scipy.io as sio
import glob
import matplotlib.pyplot as plt
from typing import List
from tqdm import tqdm

from DFInet_2.src.models import rpmnet
from DFInet_2.src.arguments_eval import RegistrationNet_arguments
from DFInet_2.src.common.misc import prepare_logger
from DFInet_2.src.common.torch import dict_all_to_device, CheckPointManager, to_numpy
from DFInet_2.src.common import se3
from DFInet_2.src.data_loader.datasets import get_test_datasets

color = ['red', 'orange', 'blue', 'green', 'fuchsia', 'black', 'yellow']

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def print_metrics(logger,
                  mse_losses_by_iteration: List = None, mae_losses_by_iteration: List = None, rmse_losses_by_iteration: List = None,
                  r_mse_losses_by_iteration: List = None, r_mae_losses_by_iteration: List = None, r_rmse_losses_by_iteration: List = None,
                  t_mse_losses_by_iteration: List = None, t_mae_losses_by_iteration: List = None, t_rmse_losses_by_iteration: List = None,
                  title: str = 'Metrics'):
    """Prints out formated metrics to logger"""

    logger.info(title + ':')
    logger.info('=' * (len(title) + 1))

    if mse_losses_by_iteration is not None:
        losses_all_str = ' | '.join(['{:.6f}'.format(c) for c in mse_losses_by_iteration])
        logger.info('mse_losses_by_iteration: {}'.format(losses_all_str))
    if rmse_losses_by_iteration is not None:
        losses_all_str = ' | '.join(['{:.6f}'.format(c) for c in rmse_losses_by_iteration])
        logger.info('rmse_losses_by_iteration: {}'.format(losses_all_str))
    if mae_losses_by_iteration is not None:
        losses_all_str = ' | '.join(['{:.6f}'.format(c) for c in mae_losses_by_iteration])
        logger.info('mae_losses_by_iteration: {}'.format(losses_all_str))
    if r_mse_losses_by_iteration is not None:
        losses_all_str = ' | '.join(['{:.6f}'.format(c) for c in r_mse_losses_by_iteration])
        logger.info('r_mse_losses_by_iteration: {}'.format(losses_all_str))
    if r_rmse_losses_by_iteration is not None:
        losses_all_str = ' | '.join(['{:.6f}'.format(c) for c in r_rmse_losses_by_iteration])
        logger.info('r_rmse_losses_by_iteration: {}'.format(losses_all_str))
    if r_mae_losses_by_iteration is not None:
        losses_all_str = ' | '.join(['{:.6f}'.format(c) for c in r_mae_losses_by_iteration])
        logger.info('r_mae_losses_by_iteration: {}'.format(losses_all_str))
    if t_mse_losses_by_iteration is not None:
        losses_all_str = ' | '.join(['{:.6f}'.format(c) for c in t_mse_losses_by_iteration])
        logger.info('t_mse_losses_by_iteration: {}'.format(losses_all_str))
    if t_rmse_losses_by_iteration is not None:
        losses_all_str = ' | '.join(['{:.6f}'.format(c) for c in t_rmse_losses_by_iteration])
        logger.info('t_rmse_losses_by_iteration: {}'.format(losses_all_str))
    if t_mae_losses_by_iteration is not None:
        losses_all_str = ' | '.join(['{:.6f}'.format(c) for c in t_mae_losses_by_iteration])
        logger.info('t_mae_losses_by_iteration: {}'.format(losses_all_str))

def inference(data_loader, model: torch.nn.Module):
    """Runs inference over entire dataset

    Args:
        data_loader (torch.utils.data.DataLoader): Dataset loader
        model (model.nn.Module): Network model to evaluate

    Returns:
        pred_transforms_all: predicted transforms (B, 2, 3) where B is total number of instances

    """

    _logger.info('Starting inference...')
    model.eval()

    all_src = []
    all_ref = []
    pred_transforms_all = []
    total_time = 0.0
    with torch.no_grad():
        for val_data in tqdm(data_loader):

            i = 0
            dict_all_to_device(val_data, _device)
            time_before = time.time()
            pred_transforms, endpoints = model(val_data, _args.num_reg_iter)
            total_time += time.time() - time_before


            points_src = val_data['points_src']
            points_ref = val_data['points_ref']


            index = -1
            for pred_transform in pred_transforms[index]:

                src_transformed = to_numpy(se3.transform(pred_transform, points_src[i]))

                # plt.scatter(src_transformed[:, 0], src_transformed[:, 1], s=0.1, facecolors='none', edgecolors='red')
                # plt.scatter(to_numpy(points_ref[i])[:, 0], to_numpy(points_ref[i])[:, 1], s=0.1, facecolors='none', edgecolors='blue')
                # plt.show()
                plt.scatter(src_transformed[:, 0], src_transformed[:, 1], s=10, color=color[0])
                plt.scatter(to_numpy(points_ref[i])[:, 0], to_numpy(points_ref[i])[:, 1], s=10, color=color[1])
                plt.yticks([])
                plt.xticks([])
                plt.subplots_adjust(top=0.98, bottom=0.02, left=0.02, right=0.98, hspace=0, wspace=0)
                plt.show()

                all_src.append(src_transformed)
                all_ref.append(to_numpy(points_ref[i]))

                i = i+1

            all_Src = np.stack(all_src, 0)  # B*N*2
            all_Ref = np.stack(all_ref, 0)  # B*N*2

            sio.savemat('../result/DFInet_blade1_S1_inference_data_test0.mat',
                        {'Src': all_Src,
                         'Ref': all_Ref})

            pred_transforms_all.append(pred_transforms[index])

    _logger.info('Total inference time: {}s'.format(total_time))

    return torch.cat(pred_transforms_all, 0)


def get_model():
    _logger.info('Computing transforms using {}'.format(_args.method))

    assert _args.resume is not None
    model = rpmnet.get_model(_args)
    model.to(_device)
    saver = CheckPointManager(os.path.join(_log_path, 'ckpt', 'models'))
    saver.load(_args.resume, model)

    return model


def main():
    # Load data_loader
    test_dataset = get_test_datasets(_args)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=_args.val_batch_size, shuffle=False)

    model = get_model()
    pred_transforms = inference(test_loader, model)  # Feedforward transforms

    all_Segmented_idx = sio.loadmat('../data/blade1_S2_inference_data_Segmented_idx.mat')['all_Segmented_idx'][0]
    sio.savemat('../result/DFInet_blade1_S1_all_Segmented_idx.mat',
                {'all_Segmented_idx': all_Segmented_idx})
    i = 2
    j = 0
    for inspection_data_path in glob.glob(os.path.join('../data', 'blade1_S*_preprocess.mat')):
        inspection_data = sio.loadmat(inspection_data_path)['inspection_data'].astype(np.float32)[..., :2]

        inspection_segmented_data_i = inspection_data[all_Segmented_idx[i-2]:all_Segmented_idx[i-1], :2]
        pred_Inspection_segmented_data_i = inspection_segmented_data_i

        while all_Segmented_idx[i]>all_Segmented_idx[i-1]:

            inspection_segmented_data_j = inspection_data[all_Segmented_idx[i-1]:all_Segmented_idx[i], :2]

            pred_Inspection_segmented_data_i = to_numpy(
                se3.transform(pred_transforms[j], torch.tensor(pred_Inspection_segmented_data_i).cuda()))

            # plt.scatter(pred_Inspection_segmented_data_i[:, 0],
            #             pred_Inspection_segmented_data_i[:, 1],
            #             s=10, color='red')
            # plt.scatter(inspection_segmented_data_j[:, 0],
            #             inspection_segmented_data_j[:, 1],
            #             s=10, color='orange')
            #
            # plt.yticks([])
            # plt.xticks([])
            # plt.show()

            pred_Inspection_segmented_data_i = np.concatenate([pred_Inspection_segmented_data_i, inspection_segmented_data_j], axis=0)

            for colorful in range(i):
                plt.scatter(pred_Inspection_segmented_data_i[all_Segmented_idx[colorful]:all_Segmented_idx[colorful+1], 0],
                            pred_Inspection_segmented_data_i[all_Segmented_idx[colorful]:all_Segmented_idx[colorful+1], 1],
                            s=10, color=color[colorful])
            plt.yticks([])
            plt.xticks([])
            # plt.axis('off')
            plt.subplots_adjust(top=0.98, bottom=0.02, left=0.02, right=0.98, hspace=0, wspace=0)
            plt.show()

            i = i+1
            j = j+1


        i = i+2


        plt.scatter(pred_Inspection_segmented_data_i[..., 0],
                    pred_Inspection_segmented_data_i[..., 1],
                    s=10, color='red')

        plt.yticks([])
        plt.xticks([])
        plt.subplots_adjust(top=0.98, bottom=0.02, left=0.02, right=0.98, hspace=0, wspace=0)
        plt.show()

        pred_Inspection_segmented_data_i = np.concatenate([pred_Inspection_segmented_data_i, np.ones((pred_Inspection_segmented_data_i.shape[0], 1))*77.55], axis=-1)
        Inspection_data = []
        i = 1
        num = 10
        while num*i<pred_Inspection_segmented_data_i.shape[0]:
            Inspection_data.append(pred_Inspection_segmented_data_i[num*i])
            i += 1
        Inspection_data = np.stack(Inspection_data, axis=0)

        sio.savemat('../result/DFInet_blade1_S1_preprocess.mat',
                    {'inspection_data': Inspection_data})




    _logger.info('Finished')


if __name__ == '__main__':
    # Arguments and logging
    parser = RegistrationNet_arguments()
    _args = parser.parse_args()
    _logger, _log_path = prepare_logger(_args, log_path=_args.eval_save_path)
    if _args.gpu >= 0 and _args.method == 'RegistrationNet':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(_args.gpu)
        _device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    else:
        _device = torch.device('cpu')

    main()
