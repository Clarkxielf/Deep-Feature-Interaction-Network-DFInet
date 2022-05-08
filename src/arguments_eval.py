"""Common arguments for train and evaluation for RegistrationNet"""


import argparse


def RegistrationNet_arguments():
    """Arguments used for both training and testing"""
    parser = argparse.ArgumentParser(add_help=False)

    # Logging
    parser.add_argument('--logdir', default='../logs', type=str,
                        help='Directory to store logs, summaries, checkpoints.')
    parser.add_argument('--debug', action='store_true', help='If set, will enable autograd anomaly detection')
    # settings for input data_loader
    parser.add_argument('-i', '--dataset_path',
                        default='../data',
                        type=str, metavar='PATH',
                        help='path to the processed data. Default: ../data')
    parser.add_argument('--data_type', default='mat',
                        metavar='DATASET', help='data type (default: mat)')
    parser.add_argument('--dataset', default='blade1_S2_inference_data_{}*.mat',
                        choices=['0.1_blade1_64data_2d_{}*.mat', 'blade1_S1_inference_data_{}*.mat'],
                        help='0.1_blade1_64data_2d_{}*.mat used for training.'
                             'blade1_S1_inference_data_{}*.mat\' used for both inference and evaluation')
    # parser.add_argument('--resume', default=None, type=str, metavar='PATH',
    #                     help='Pretrained network to load from. Optional for train, required for inference.')
    parser.add_argument('--resume', default='../logs/211216_164437/ckpt', type=str, metavar='PATH',
                        help='Pretrained network to load from. Optional for train, required for inference.')
    parser.add_argument('--num_points', default=64, type=int,
                        metavar='N', help='points in point-cloud (default: 128)')
    # Model
    parser.add_argument('--method', type=str, default='RegistrationNet', choices=['RegistrationNet'],
                        help='Model to use. Note: Only RegistrationNet is supported for training.'
                             '\'gt\' denotes groundtruth transforms')
    # DGCNN settings
    parser.add_argument('--num_neighbors', type=int, default=32, metavar='N', help='Max num of neighbors to use')
    parser.add_argument('--feat_dim', type=int, default=96,
                        help='Feature dimension (to compute distances on). Other numbers will be scaled accordingly')
    # RegistrationNet settings
    parser.add_argument('--num_reg_iter', type=int, default=5,
                        help='Number of iterations used for registration (only during inference)')
    parser.add_argument('--loss_type', type=str, choices=['mse', 'mae', 'rmse'], default='mae',
                        help='Loss to be optimized')
    # Training parameters
    parser.add_argument('--train_batch_size', default=8, type=int, metavar='N',
                        help='training mini-batch size (default: 8)')
    parser.add_argument('-b', '--val_batch_size', default=16, type=int, metavar='N',
                        help='mini-batch size during validation or testing (default: 16)')
    parser.add_argument('--gpu', default=0, type=int, metavar='DEVICE',
                        help='GPU to use, ignored if no GPU is present. Set to negative to use cpu')
    """Used only for training"""
    # Training parameters
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate during training')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--summary_every', default=200, type=int, metavar='N',
                        help='Frequency of saving summary (number of steps if positive, number of epochs if negative)')
    parser.add_argument('--validate_every', default=-1, type=int, metavar='N',
                        help='Frequency of evaluation (number of steps if positive, number of epochs if negative). Also saves checkpoints at the same interval')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Number of workers for data_loader loader (default: 0).')
    parser.add_argument('--num_train_reg_iter', type=int, default=2,
                        help='Number of iterations used for registration (only during training)')
    """Used during evaluation"""
    # Save out evaluation data_loader for further analysis
    parser.add_argument('--eval_save_path', type=str, default='../eval_results',
                        help='Output data_loader to save evaluation results')
    parser.add_argument('--wt_inliers', type=float, default=1e-2, help='Weight to encourage inliers')
    parser.add_argument('--no_slack', action='store_true', help='If set, will not have a slack column.')
    parser.add_argument('--num_sk_iter', type=int, default=5,
                        help='Number of inner iterations used in sinkhorn normalization')


    return parser
