import argparse

def breast_arg():

    parser = argparse.ArgumentParser('Train a model')
    parser.add_argument('--logging_root', type=str, default='./logs_516', help='root for logging')
    parser.add_argument('--exp_name', type=str, default='vgg', help='experiment name')
    parser.add_argument('--model', type=str, default='vgg', help='{vgg, alexnet, effnet}')
    parser.add_argument('--epochs', type=float, default=20, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--bt', type=int, default=128, help='learning rate')
    parser.add_argument('--epochs_til_checkpoint', type=float, default=1, help='epochs_til_checkpoint')
    args = parser.parse_args()
    return args
