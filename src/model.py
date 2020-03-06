import pytorch_lightning as pl

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .networks import LeNet
from .metrics import top_k_accuracy

class Model(pl.LightningModule):
    def __init__(self, hparams):
        super(Model, self).__init__()

        del hparams.config
        self.hparams = hparams

        if   hparams.dataset.lower() == 'mnist':
            in_ch, out_ch, fc_shape = 1, 10, (7, 7)
        else:
            raise NotImplementedError(f'Unknown dataset "{self.hparams.dataset}"')

        if   hparams.network.lower() == 'lenet':
            self.network = LeNet(in_ch, out_ch, hparams.activation, fc_shape)
            self.loss = nn.NLLLoss()
        else:
            raise NotImplementedError(f'Unknown network "{hparams.network}"')

    def configure_optimizers(self):
        lr = self.hparams.learning_rate

        if   self.hparams.optimizer.lower() == 'sgd':
            return optim.SGD(self.network.parameters(), lr,
                             momentum=0, dampening=0, weight_decay=0, nesterov=False)
        elif self.hparams.optimizer.lower() == 'adam':
            return optim.Adam(self.network.parameters(), lr)
        else:
            raise NotImplementedError(f'Unknown optimizer "{self.hparams.optimizer}"')

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        loss = self.loss(y_hat, y)
        top1 = top_k_accuracy(y_hat, y, 1)
        top3 = top_k_accuracy(y_hat, y, 3)
        top5 = top_k_accuracy(y_hat, y, 5)

        return {
            'loss': loss,
            'log': {
                'train/loss': loss,
                'train/top1_acc': top1,
                'train/top3_acc': top3,
                'train/top5_acc': top5
            }
        }

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)['log']

    def validation_end(self, outputs):
        val_dict = {f'{key.replace("train", "valid", 1)}': 0 for key in outputs[0].keys()}

        for output in outputs:
            for key, value in output.items():
                val_dict[f'{key.replace("train", "valid", 1)}'] += value.float().mean()

        val_dict = {key: value / len(outputs) for key, value in val_dict.items()}

        return {
            'log' : val_dict
        }

    @pl.data_loader
    def train_dataloader(self):
        if   self.hparams.dataset.lower() == 'mnist':
            transform = transforms.ToTensor()
            dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        else:
            raise NotImplementedError(f'Unknown dataset "{self.hparams.dataset}"')

        batch_size = self.hparams.batch_size

        return DataLoader(dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4,
                          drop_last=True)

    @pl.data_loader
    def val_dataloader(self):
        if   self.hparams.dataset.lower() == 'mnist':
            transform = transforms.ToTensor()
            dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        else:
            raise NotImplementedError(f'Unknown dataset "{self.hparams.dataset}"')

        batch_size = self.hparams.batch_size

        return DataLoader(dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4,
                          drop_last=False)
