import argparse
from cs231n.classifiers.cnn import MultiLayerConvNet, MultiLayerConvNetCupy
from cs231n.data_utils import get_CIFAR10_data
from cs231n.solver import Solver, CupySolver
import os
import time
import numpy as np
import cupy as cp
data = get_CIFAR10_data()

def parse_args():
    parser = argparse.ArgumentParser(description="Train a MultiLayerConvNet on CIFAR-10")
    parser.add_argument('--conv', type=int, nargs='+', default=[16, 32], help='List of filters for conv layers')
    parser.add_argument('--fc', type=int, nargs='+', default=[300, 100], help='List of hidden dims for FC layers')
    parser.add_argument('--reg', type=float, default=0.001, help='L2 regularization strength')
    parser.add_argument('--epochs', '-n',type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--batch_size','-b', type=int, default=50, help='Batch size')
    parser.add_argument('--optim', type=str, default='adam', help='Optimizer (sgd, adam, etc)')
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate')

    parser.add_argument('--lr_decay', type=float, default=1.0, help='Learning rate decay')
    
    parser.add_argument('--filter_size', type=int, default=5, help='Filter size for conv layers')
    
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--print_every', type=int, default=50, help='Print every n iterations')
    parser.add_argument('--output_dir', '-o', type=str, default='output', help='Output directory')
    parser.add_argument('--checkpoint', '-c',type=str, default='checkpoint', help='Checkpoint file name')
    parser.add_argument('--device', '-d', type=str, default='cpu', help='Device to use (cpu, gpu)', choices=['cpu', 'gpu'])
    parser.add_argument('--seed', type=int, default=42, help='Seed for the random number generator')

    return parser.parse_args()

def main():
    args = parse_args()

    np.random.seed(args.seed)
    cp.random.seed(args.seed)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    if args.device == 'gpu':
        model = MultiLayerConvNetCupy(
            weight_scale=0.001,
            nums_filters=args.conv,
            hidden_dims=args.fc,
            reg=args.reg,
            filter_size=args.filter_size
        )
    else:
        model = MultiLayerConvNet(
            weight_scale=0.001,
            nums_filters=args.conv,
            hidden_dims=args.fc,
            reg=args.reg,
            filter_size=args.filter_size
        )
        
    print(f"-----Info-----")
    print(f"Convolutional Layers: {args.conv}")
    print(f"Filter size: {args.filter_size}")
    print(f"Fully connected Layers: {args.fc}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Regularization strength: {args.reg}")
    print(f"Output directory: {output_dir}")
    if args.device == 'gpu':
        solver = CupySolver(
            model,
            data,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            update_rule=args.optim,
            optim_config={'learning_rate': args.lr},
            lr_decay=args.lr_decay,
            verbose=args.verbose,
            print_every=args.print_every,
            checkpoint_name=os.path.join(output_dir, args.checkpoint)
        )
    else:
        solver = Solver(
        model,
        data,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        update_rule=args.optim,
        optim_config={'learning_rate': args.lr},
        lr_decay=args.lr_decay,
        verbose=args.verbose,
        print_every=args.print_every,
        checkpoint_name=os.path.join(output_dir, args.checkpoint)
    )
    solver.train()

    print("Full data training accuracy:", solver.check_accuracy(data['X_train'], data['y_train']))
    print("Full data validation accuracy:", solver.check_accuracy(data['X_val'], data['y_val']))
    print("Full data test accuracy:", solver.check_accuracy(data['X_test'], data['y_test']))


if __name__ == "__main__":
    main()
