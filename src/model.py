import pytorch_lightning as pl

class Model(pl.LightningModule):

    def __init__(self, hparams):
        super(Model, self).__init__()

        self.hparams = hparams

        if hparams.network.lower() == 'lenet':
            pass
