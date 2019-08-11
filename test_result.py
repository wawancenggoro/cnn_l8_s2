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
    # Dataloader with open from directory
    #====================================================================================================
    # train_csv = "../dataset/l8s2-train.csv"
    # val_csv = "../dataset/l8s2-val.csv"
    # test_csv = "../dataset/l8s2-test.csv"

    # input_transform = transforms.Compose([
    #                         transforms.ToTensor(),
    #                         NormalizeL8(
    #                                 (489.7118, 591.63416, 826.2221, 948.7332, 1858.4872, 1864.6527, 1355.4669), 
    #                                 (338.75378, 403.48727, 572.8161, 784.2508, 1208.3722, 1436.1204, 1138.7588)
    #                         )
    #                     ])

    # target_transform = transforms.Compose([
    #                         transforms.Lambda(lambda x: [x[i].astype('float32') for i in range(13)]),
    #                         transforms.Lambda(lambda x: [transforms.ToTensor()(x[i]) for i in range(13)]),
    #                         NormalizeS2(
    #                                 (1440.2627, 1258.3445, 1214.9252, 1325.0135, 1486.8649, 1866.3961, 2085.1528, 2070.0884, 2272.1758, 931.276, 21.306807, 2370.4104, 1701.286), 
    #                                 (366.68463, 378.73654, 512.0519, 771.2212, 791.2124, 874.36127, 989.072, 1001.9915, 1093.7765, 552.87885, 28.292986, 1379.6288, 1097.3044)
    #                         )
    #                     ])

    # train_set = Landsat8Dataset(train_csv,
    #     input_transform = input_transform,
    #     target_transform=target_transform)
    # train_data_loader = DataLoader(dataset=train_set, batch_size=args.batchSize, shuffle=True)

    # val_set = Landsat8Dataset(val_csv,
    #     input_transform = input_transform,
    #     target_transform=target_transform)
    # val_data_loader = DataLoader(dataset=val_set, batch_size=args.testBatchSize, shuffle=False)

    # test_set = Landsat8Dataset(test_csv,
    #     input_transform = input_transform,
    #     target_transform=target_transform)
    # test_data_loader = DataLoader(dataset=test_set, batch_size=args.testBatchSize, shuffle=False)
    #====================================================================================================




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

    # from SRCNN.model import Net
    # model = Net(num_channels=7, base_filter=64).to(torch.device('cuda'))
    
    # data,target=train_set.__getitem__(0)
    # out = model(data.unsqueeze(0).to(torch.device('cuda')))

    if args.model == 'sub':
        model = SubPixelTrainer(args, train_data_loader, test_data_loader, weights='/mnt/SSD/Projects/dikti2019PakSani/l8-s2-cnn/save/SubPixelCNN/model_path.pth')
    elif args.model == 'trans':
        model = TransConvTrainer(args, train_data_loader, test_data_loader, weights='/mnt/SSD/Projects/dikti2019PakSani/l8-s2-cnn/save/TransConvCNN/model_path.pth')

    elif args.model == 'submax':
        model = SubPixelMaxPoolTrainer(args, train_data_loader, test_data_loader, weights='/mnt/SSD/Projects/dikti2019PakSani/l8-s2-cnn/save/SubPixelMaxPoolCNN/model_path.pth')
    elif args.model == 'transmax':
        model = TransConvMaxPoolTrainer(args, train_data_loader, test_data_loader, weights='/mnt/SSD/Projects/dikti2019PakSani/l8-s2-cnn/save/TransConvMaxPoolCNN/model_path.pth')

    elif args.model == 'ups':
        model = UpsampleTrainer(args, train_data_loader, test_data_loader)

    elif args.model == 'srcnn':
        model = SRCNNTrainer(args, train_data_loader, test_data_loader)
    elif args.model == 'vdsr':
        model = VDSRTrainer(args, train_data_loader, test_data_loader)
    elif args.model == 'edsr':
        model = EDSRTrainer(args, train_data_loader, test_data_loader)
    elif args.model == 'fsrcnn':
        model = FSRCNNTrainer(args, train_data_loader, test_data_loader)
    elif args.model == 'drcn':
        model = DRCNTrainer(args, train_data_loader, test_data_loader)
    elif args.model == 'srgan':
        model = SRGANTrainer(args, train_data_loader, test_data_loader)
    elif args.model == 'dbpn':
        model = DBPNTrainer(args, train_data_loader, test_data_loader)
    else:
        raise Exception("the model does not exist")

    model.test()

if __name__ == '__main__':
    main()