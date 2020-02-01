from __future__ import print_function

from math import log10
import csv

import torch
import torch.backends.cudnn as cudnn

from SubPixelMaxPoolCNN.model import Net
from progress_bar import progress_bar
import time
from progress_bar import format_time


class SubPixelMaxPoolTrainer(object):
    def __init__(self, config, training_loader, val_loader, weights=None):
        super(SubPixelMaxPoolTrainer, self).__init__()
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.CUDA else 'cpu')
        self.model = None
        self.lr = config.lr
        self.nEpochs = config.nEpochs
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.seed = config.seed
        self.upscale_factor = config.upscale_factor
        self.training_loader = training_loader
        self.val_loader = val_loader
        self.num_targets = 13
        self.band_names = ['01','02','03','04','05','06','07','08','8A','09', '10','11','12']
        self.gpus = [0]
        self.weights = weights

    def build_model(self):
        self.model = Net(upscale_factor=self.upscale_factor)
        
        if self.weights is not None:
            self.model = torch.load(self.weights)
        elif len(self.gpus)>1:
            print("Use Multiple GPUs")
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpus).to(self.device)
        else:
            self.model=self.model.to(self.device)

        self.criterion = torch.nn.MSELoss()
        torch.manual_seed(self.seed)

        if self.CUDA:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 75, 100], gamma=0.5)  # lr decay

    def save(self):
        model_out_path = "model_path.pth"
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def train(self):
        self.model.train()
        train_loss = 0

        for batch_num, (data, target) in enumerate(self.training_loader):
            data = data.to(self.device)
            target = [target[i].to(self.device) for i in range(self.num_targets)]
            
            self.optimizer.zero_grad()
            out = self.model(data)

            loss = self.criterion(out[0], target[0])
            for i in range(self.num_targets-1):
                loss += self.criterion(out[i+1], target[i+1]) 

            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            progress_bar(batch_num, len(self.training_loader), 'Loss: {:.4f}'.format((train_loss / (batch_num + 1))))

        avg_train_loss = train_loss / len(self.training_loader)
        print("    Average Loss: {:.4f}".format(avg_train_loss))
        return avg_train_loss

    def val(self):
        self.model.eval()
        avg_mse = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
        avg_psnr = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
        val_loss = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.val_loader):
                data = data.to(self.device)
                target = [target[i].to(self.device) for i in range(self.num_targets)]

                prediction = self.model(data)

                mse = [self.criterion(prediction[i], target[i]) for i in range(self.num_targets)]
                loss = mse[0]
                for i in range(self.num_targets-1):
                    loss += mse[i+1]
                
                val_loss += loss.item()

                avg_mse = [avg_mse[i] + mse[i].item() for i in range(self.num_targets)]
                psnr = [10 * log10(1 / mse[i].item()) for i in range(self.num_targets)]
                avg_psnr = [avg_psnr[i] + psnr[i] for i in range(self.num_targets)]

                # progress_bar(batch_num, len(self.val_loader), 'PSNR: %.4f' % (avg_psnr / (batch_num + 1)))
                progress_bar(batch_num, len(self.val_loader), 'Val Loss: {:.4f}'.format(val_loss / (batch_num+1)))


        avg_val_loss = val_loss / len(self.val_loader)
        avg_mse = [avg_mse[i] / len(self.val_loader) for i in range(self.num_targets)]
        avg_psnr = [avg_psnr[i] / len(self.val_loader) for i in range(self.num_targets)]

        print("    Average Val Loss: {:.4f}".format(avg_val_loss))
        print("    MSE: [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(
            avg_mse[0], 
            avg_mse[1], 
            avg_mse[2], 
            avg_mse[3], 
            avg_mse[4], 
            avg_mse[5], 
            avg_mse[6], 
            avg_mse[7], 
            avg_mse[8], 
            avg_mse[9], 
            avg_mse[10], 
            avg_mse[11], 
            avg_mse[12]))

        print("    PSNR: [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(
            avg_psnr[0], 
            avg_psnr[1], 
            avg_psnr[2], 
            avg_psnr[3], 
            avg_psnr[4], 
            avg_psnr[5], 
            avg_psnr[6], 
            avg_psnr[7], 
            avg_psnr[8], 
            avg_psnr[9], 
            avg_psnr[10], 
            avg_psnr[11], 
            avg_psnr[12]))      

        return avg_val_loss, avg_mse, avg_psnr

    def test(self):
        self.build_model()
        self.model.eval()
        avg_mse = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
        avg_psnr = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
        val_loss = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.val_loader):
                data = data.to(self.device)
                target = [target[i].to(self.device) for i in range(self.num_targets)]

                prediction = self.model(data)

                mse = [self.criterion(prediction[i], target[i]) for i in range(self.num_targets)]
                loss = mse[0]
                for i in range(self.num_targets-1):
                    loss += mse[i+1] 
                
                val_loss += loss.item()

                avg_mse = [avg_mse[i] + mse[i].item() for i in range(self.num_targets)]
                psnr = [10 * log10(1 / mse[i].item()) for i in range(self.num_targets)]
                avg_psnr = [avg_psnr[i] + psnr[i] for i in range(self.num_targets)]

                # progress_bar(batch_num, len(self.val_loader), 'PSNR: %.4f' % (avg_psnr / (batch_num + 1)))
                progress_bar(batch_num, len(self.val_loader), 'Val Loss: {:.4f}'.format(val_loss / (batch_num+1)))


        avg_val_loss = val_loss / len(self.val_loader)
        avg_mse = [avg_mse[i] / len(self.val_loader) for i in range(self.num_targets)]
        avg_psnr = [avg_psnr[i] / len(self.val_loader) for i in range(self.num_targets)]

        print("    Average Val Loss: {:.4f}".format(avg_val_loss))
        print("    MSE: [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(
            avg_mse[0], 
            avg_mse[1], 
            avg_mse[2], 
            avg_mse[3], 
            avg_mse[4], 
            avg_mse[5], 
            avg_mse[6], 
            avg_mse[7], 
            avg_mse[8], 
            avg_mse[9], 
            avg_mse[10], 
            avg_mse[11], 
            avg_mse[12]))

        print("    PSNR: [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(
            avg_psnr[0], 
            avg_psnr[1], 
            avg_psnr[2], 
            avg_psnr[3], 
            avg_psnr[4], 
            avg_psnr[5], 
            avg_psnr[6], 
            avg_psnr[7], 
            avg_psnr[8], 
            avg_psnr[9], 
            avg_psnr[10], 
            avg_psnr[11], 
            avg_psnr[12]))      

        with open(r'mse_test.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12'])
            writer.writerow(avg_mse)

        with open(r'psnr_test.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12'])
            writer.writerow(avg_psnr)

        return avg_val_loss, avg_mse, avg_psnr

    def run(self):
        self.build_model()
        best_val_loss = float('inf')

        with open(r'losses.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['train_loss', 'val_loss'])

        with open(r'mse.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12'])

        with open(r'psnr.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12'])

        for epoch in range(1, self.nEpochs + 1):
            print("\n===> Epoch {} starts:".format(epoch))
            train_loss = self.train()

            print("===> Validating:")
            val_loss, avg_mse, avg_psnr = self.val()
            if val_loss < best_val_loss:
                self.save()
                best_val_loss = val_loss

            with open(r'losses.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow([train_loss, val_loss])

            with open(r'mse.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(avg_mse)

            with open(r'psnr.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(avg_psnr)

            self.scheduler.step(epoch)
