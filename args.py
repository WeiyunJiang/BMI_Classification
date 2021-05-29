import argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def breast_arg():

    parser = argparse.ArgumentParser('Train a model')
    parser.add_argument('--logging_root', type=str, default='./logs_final', help='root for logging')
    parser.add_argument('--exp_name', type=str, default='effnet', help='experiment name')
    parser.add_argument('--model', type=str, default='effnet', help='{vgg, alexnet, effnet, senet}')
    parser.add_argument('--epochs', type=float, default=20, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--bt', type=int, default=2, help='learning rate')
    parser.add_argument("--cat_feat", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate cat_feat mode.")
    parser.add_argument("--data_aug", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate augmentation mode.")
    parser.add_argument('--epochs_til_checkpoint', type=float, default=1, help='epochs_til_checkpoint')
    
    args = parser.parse_args()
    return args
