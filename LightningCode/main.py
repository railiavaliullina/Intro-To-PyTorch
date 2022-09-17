import time
import warnings

from LightningCode.config import cfg
from LightningCode.model import LitLogisticRegression, LitMLP1HL, LitMLP2HL
from LightningCode.data import get_data

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    dl_train, dl_test = get_data()

    if cfg.model is 'LR':
        model = LitLogisticRegression(cfg.input_dim, cfg.output_dim).to(cfg.device)
    elif cfg.model is 'MLP1HL':
        model = LitMLP1HL(cfg.input_dim, cfg.output_dim).to(cfg.device)
    elif cfg.model is 'MLP2HL':
        model = LitMLP2HL(cfg.input_dim, cfg.output_dim).to(cfg.device)
    else:
        raise Exception

    print('\nModel parameters: ')
    print(model)

    nb_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters number: {nb_trainable_params}')

    tb_logger = pl_loggers.TensorBoardLogger('logs/')
    trainer = pl.Trainer(max_epochs=cfg.epochs, log_every_n_steps=1, logger=tb_logger)#, auto_lr_find=True,
                         # auto_scale_batch_size=True)  #, profiler=True
    start_time = time.time()
    trainer.fit(model, dl_train, dl_test)
    print(f'Total training time: {round((time.time() - start_time) / 60, 3)} min')
