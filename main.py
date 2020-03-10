import torch
from torch.utils.data import DataLoader
from dataset.dataset import Landsat8Dataset, Landsat8DatasetHDF5
from dataset.dataset import LocalRandomSampler
from dataset.customTransform import NormalizeL8, NormalizeS2
from torchvision import transforms

from DBPN.solver import DBPNTrainer
from DRCN.solver import DRCNTrainer
from EDSR.solver import EDSRTrainer
from FSRCNN.solver import FSRCNNTrainer
from SRCNN.solver import SRCNNTrainer
from SRGAN.solver import SRGANTrainer
from VDSR.solver import VDSRTrainer

from SubPixelCNN.solver import SubPixelTrainer
from TransConvCNN.solver import TransConvTrainer

from SubPixelMaxPoolCNN.solver import SubPixelMaxPoolTrainer
from TransConvMaxPoolCNN.solver import TransConvMaxPoolTrainer

from UpsampleCNN.solver import UpsampleTrainer

import argparse
from IPython import embed

parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
# hyper-parameters
parser.add_argument('--batchSize', type=int, default=4, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=4, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.01')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')

# model configuration
parser.add_argument('--upscale_factor', '-uf',  type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--model', '-m', type=str, default='sub', help='choose which model is going to use')

args = parser.parse_args()

def main():
    train_csv = "../dataset/l8s2-train.csv"
    val_csv = "../dataset/l8s2-val.csv"
    test_csv = "../dataset/l8s2-test.csv"

    #====================================================================================================
    # Dataloader with HDF5
    #====================================================================================================
    input_transform = transforms.Compose([
                            transforms.ToTensor()
                        ])

    target_transform = transforms.Compose([
                            transforms.Lambda(lambda x: [x[i].astype('float32') for i in range(13)]),
                            transforms.Lambda(lambda x: [transforms.ToTensor()(x[i]) for i in range(13)])
                        ])

    train_set = Landsat8DatasetHDF5(train_csv,
        input_transform = input_transform,
        target_transform=target_transform)
    # train_data_loader = DataLoader(dataset=train_set, batch_size=args.batchSize, sampler = LocalRandomSampler(train_set))
    train_data_loader = DataLoader(dataset=train_set, batch_size=args.batchSize, shuffle=True)

    val_set = Landsat8DatasetHDF5(val_csv,
        input_transform = input_transform,
        target_transform=target_transform)
    val_data_loader = DataLoader(dataset=val_set, batch_size=args.testBatchSize, shuffle=False)

    test_set = Landsat8DatasetHDF5(test_csv,
        input_transform = input_transform,
        target_transform=target_transform)
    test_data_loader = DataLoader(dataset=test_set, batch_size=args.testBatchSize, shuffle=False)
    #====================================================================================================

    if args.model == 'sub':
        model = SubPixelTrainer(args, train_data_loader, val_data_loader)
    elif args.model == 'trans':
        model = TransConvTrainer(args, train_data_loader, val_data_loader)

    elif args.model == 'submax':
        model = SubPixelMaxPoolTrainer(args, train_data_loader, val_data_loader)
    elif args.model == 'transmax':
        model = TransConvMaxPoolTrainer(args, train_data_loader, val_data_loader)

    else:
        raise Exception("the model does not exist")

    model.run()

if __name__ == '__main__':
    main()
