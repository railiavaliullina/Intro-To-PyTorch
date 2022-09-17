import torch
import time
import warnings
import matplotlib.pyplot as plt

from FastaiCode.config import cfg
from FastaiCode.model import LogisticRegression, MLP1HL, MLP2HL
from FastaiCode.data import get_data

from fastai.learner import Learner
from fastai.metrics import accuracy
from fastai.callback.tensorboard import TensorBoardCallback
from fastai.optimizer import SGD
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    dls = get_data()

    if cfg.model is 'LR':
        model = LogisticRegression(cfg.input_dim, cfg.output_dim)
    elif cfg.model is 'MLP1HL':
        model = MLP1HL(cfg.input_dim, cfg.output_dim)
    elif cfg.model is 'MLP2HL':
        model = MLP2HL(cfg.input_dim, cfg.output_dim)
    else:
        raise Exception

    print('\nModel parameters: ')
    print(model)
    nb_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters number: {nb_trainable_params}')

    cbs = TensorBoardCallback(log_dir='logs', trace_model=True, log_preds=False)
    learner = Learner(dls=dls, model=model, loss_func=torch.nn.CrossEntropyLoss(), metrics=accuracy, cbs=cbs, opt_func=SGD)
    suggested_lr = learner.lr_find()[0]

    plt.show()

    start_time = time.time()
    learner.fit_one_cycle(cfg.epochs, suggested_lr)
    learner.save(file=f'checkpoint_{cfg.epochs}')

    print(f'Total training time: {round((time.time() - start_time) / 60, 3)} min')

    learner.recorder.plot_loss()
    plt.show()

    learner.recorder.plot_loss()
