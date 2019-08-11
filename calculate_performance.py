import torch
from torch.utils.data import DataLoader
from dataset.dataset import Landsat8Dataset, Landsat8DatasetHDF5
from dataset.dataset import LocalRandomSampler
from dataset.customTransform import DenormalizeS2
from torchvision import transforms

import argparse
from IPython import embed

import gc
import gdal
from gdalconst import GA_ReadOnly
from osgeo import osr

import numpy as np
from PIL import Image
from math import log10

parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
# hyper-parameters
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.01')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')

# model configuration
parser.add_argument('--upscale_factor', '-uf',  type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--model', '-m', type=str, default='sub', help='choose which model is going to use')

args = parser.parse_args()

def main():
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

    test_set = Landsat8DatasetHDF5(test_csv,
        input_transform = input_transform,
        target_transform=target_transform)
    test_data_loader = DataLoader(dataset=test_set, batch_size=args.testBatchSize, shuffle=False)
    #====================================================================================================

    # L8
    # means = [489.7118, 591.63416, 826.2221, 948.7332, 1858.4872, 1864.6527, 1355.4669]
    # sds = [338.75378, 403.48727, 572.8161, 784.2508, 1208.3722, 1436.1204, 1138.7588]

    # S2
    means = [1440.2627, 1258.3445, 1214.9252, 1325.0135, 1486.8649, 1866.3961, 2085.1528, 2070.0884, 2272.1758, 931.276, 21.306807, 2370.4104, 1701.286]
    sds = [366.68463, 378.73654, 512.0519, 771.2212, 791.2124, 874.36127, 989.072, 1001.9915, 1093.7765, 552.87885, 28.292986, 1379.6288, 1097.3044]

    modelname = 'SubPixelCNN'
    modelname = 'SubPixelMaxPoolCNN'
    # modelname = 'TransConvCNN'
    # modelname = 'TransConvMaxPoolCNN'
    model = torch.load('save/'+modelname+'/model_path.pth')

    # model = torch.load('save/SubPixelCNN/model_path.pth')
    # model = torch.load('save/SubPixelMaxPoolCNN/model_path.pth')
    # model = torch.load('save/TransConvCNN/model_path.pth')
    # model = torch.load('save/TransConvMaxPoolCNN/model_path.pth')

    model.eval()

    pred_min = np.zeros((90,13))
    pred_max = np.zeros((90,13))
    pred_mean = np.zeros((90,13))
    pred_std = np.zeros((90,13))

    gt_min = np.zeros((90,13))
    gt_max = np.zeros((90,13))
    gt_mean = np.zeros((90,13))
    gt_std = np.zeros((90,13))

    rmse = np.zeros((90,13))
    mae = np.zeros((90,13))
    psnr = np.zeros((90,13))
    ssim = np.zeros((90,13))

    iter_loader = iter(test_data_loader)
    for i in range(90):
        print(i)
        input, target = next(iter_loader)
        # import pdb; pdb.set_trace()
        out = model(input.cuda())
        denorm = DenormalizeS2(means, sds)

        out_denorm = denorm(out)
        out_denorm[0] = out_denorm[0].reshape(500,500) 
        out_denorm[9] = out_denorm[9].reshape(500,500)
        out_denorm[10] = out_denorm[10].reshape(500,500)
        out_denorm[4] = out_denorm[4].reshape(1500,1500) 
        out_denorm[5] = out_denorm[5].reshape(1500,1500) 
        out_denorm[6] = out_denorm[6].reshape(1500,1500) 
        out_denorm[8] = out_denorm[8].reshape(1500,1500)
        out_denorm[11] = out_denorm[11].reshape(1500,1500)
        out_denorm[12] = out_denorm[12].reshape(1500,1500)
        out_denorm[1] = out_denorm[1].reshape(3000,3000) 
        out_denorm[2] = out_denorm[2].reshape(3000,3000) 
        out_denorm[3] = out_denorm[3].reshape(3000,3000) 
        out_denorm[7] = out_denorm[7].reshape(3000,3000) 

        target_denorm = denorm(target)
        target_denorm[0] = target_denorm[0].reshape(500,500) 
        target_denorm[9] = target_denorm[9].reshape(500,500)
        target_denorm[10] = target_denorm[10].reshape(500,500)
        target_denorm[4] = target_denorm[4].reshape(1500,1500) 
        target_denorm[5] = target_denorm[5].reshape(1500,1500) 
        target_denorm[6] = target_denorm[6].reshape(1500,1500) 
        target_denorm[8] = target_denorm[8].reshape(1500,1500)
        target_denorm[11] = target_denorm[11].reshape(1500,1500)
        target_denorm[12] = target_denorm[12].reshape(1500,1500)
        target_denorm[1] = target_denorm[1].reshape(3000,3000) 
        target_denorm[2] = target_denorm[2].reshape(3000,3000) 
        target_denorm[3] = target_denorm[3].reshape(3000,3000) 
        target_denorm[7] = target_denorm[7].reshape(3000,3000) 

        for j in range(13):
            pred = out_denorm[j]
            gt = target_denorm[j]
            gt = gt.round()

            pred_min[i,j] = pred.min()
            pred_max[i,j] = pred.max()
            pred_mean[i,j] = pred.mean()
            pred_std[i,j] = pred.std()

            gt_min[i,j] = gt.min()
            gt_max[i,j] = gt.max()
            gt_mean[i,j] = gt.mean()
            gt_std[i,j] = gt.std()

            pred_gt_cov = np.mean((pred-pred_mean[i,j])*(gt-gt_mean[i,j]))

            mae[i,j] = np.abs(pred-gt).mean()
            mse = np.square(pred-gt).mean()
            rmse[i,j] = np.sqrt(mse)

            L = 10000.            
            psnr[i,j] = 10 * log10(L**2 / mse)
            
            k1 = 0.01
            k2 = 0.03
            c1 = (k1*L)**2
            c2 = (k2*L)**2
            ssim[i,j] = ((2*pred_mean[i,j]*gt_mean[i,j] + c1) * (pred_gt_cov + c2)) / ((pred_mean[i,j]**2 + gt_mean[i,j]**2 + c1) * (pred_std[i,j]**2 + gt_std[i,j]**2 + c2))

        del out

    np.savetxt("test_results/pred_min.csv", pred_min, delimiter=",")
    np.savetxt("test_results/pred_max.csv", pred_max, delimiter=",")
    np.savetxt("test_results/pred_mean.csv", pred_mean, delimiter=",")
    np.savetxt("test_results/pred_std.csv", pred_std, delimiter=",")

    np.savetxt("test_results/gt_min.csv", gt_min, delimiter=",")
    np.savetxt("test_results/gt_max.csv", gt_max, delimiter=",")
    np.savetxt("test_results/gt_mean.csv", gt_mean, delimiter=",")
    np.savetxt("test_results/gt_std.csv", gt_std, delimiter=",")

    np.savetxt("test_results/mae.csv", mae, delimiter=",")
    np.savetxt("test_results/rmse.csv", rmse, delimiter=",")
    np.savetxt("test_results/psnr.csv", psnr, delimiter=",")
    np.savetxt("test_results/ssim.csv", ssim, delimiter=",")


        # import IPython; IPython.embed()


if __name__ == '__main__':
    main()