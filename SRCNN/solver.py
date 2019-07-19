from __future__ import print_function

from math import log10

import torch
import torch.backends.cudnn as cudnn

from SRCNN.model import Net
from progress_bar import progress_bar


class SRCNNTrainer(object):
    def __init__(self, config, training_loader, testing_loader):
        super(SRCNNTrainer, self).__init__()
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
        self.testing_loader = testing_loader
        self.num_targets = 13
        self.band_names = ['01','02','03','04','05','06','07','08','8A','09', '10','11','12']

    def build_model(self):
        self.model = Net(num_channels=7, base_filter=64).to(self.device)
        self.model.weight_init(mean=0.0, std=0.01)
        self.criterion = torch.nn.MSELoss()
        torch.manual_seed(self.seed)

        if self.CUDA:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 75, 100], gamma=0.5)

    def save_model(self):
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

            for i in range(self.num_targets):
                if i==0:
                    loss = self.criterion(out[i], target[i])
                else:
                    loss += self.criterion(out[i], target[i]) 

            train_loss += loss.item()
            loss.backward(retain_graph=True)
            self.optimizer.step()
            progress_bar(batch_num, len(self.training_loader), 'Loss: %.4f' % (train_loss / (batch_num + 1)))

        print("    Average Loss: {:.4f}".format(train_loss / len(self.training_loader)))

    def test(self):
        self.model.eval()
        avg_psnr = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.testing_loader):
                data = data.to(self.device)
                target = [target[i].to(self.device) for i in range(self.num_targets)]

                prediction = self.model(data)
                mse = [self.criterion(prediction[i], target[i]) for i in range(self.num_targets)]
                psnr = [10 * log10(1 / mse[i].item()) for i in range(self.num_targets)]
                avg_psnr = [avg_psnr[i] + psnr[i] for i in range(self.num_targets)]
                # progress_bar(batch_num, len(self.testing_loader), 'PSNR: %.4f' % (avg_psnr / (batch_num + 1)))
                progress_bar(batch_num, len(self.testing_loader), '')

        for i in range(self.num_targets):
            print("    Average PSNR Band {:s}: {:.4f} dB".format(self.band_names[i], avg_psnr[i] / len(self.testing_loader)))

    def run(self):
        self.build_model()
        for epoch in range(1, self.nEpochs + 1):
            print("\n===> Epoch {} starts:".format(epoch))
            self.train()
            self.test()
            self.scheduler.step(epoch)
            if epoch == self.nEpochs:
                self.save_model()
