import torch
import time
import mlflow
import os
import warnings

from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.metrics import Accuracy
from IgniteCode.utils import CustomLossValidation

from IgniteCode.config import cfg
from IgniteCode.model import LogisticRegression, MLP1HL, MLP2HL
from IgniteCode.data import get_data
warnings.filterwarnings('ignore')


def get_criterion():
    criterion_ = torch.nn.CrossEntropyLoss()
    return criterion_


def get_optimizer(model_):
    optimizer_ = torch.optim.SGD(model_.parameters(), lr=cfg.lr)
    return optimizer_


def make_step(engine, batch):
    x, y = batch[0], batch[1]
    model.train()
    y_pred = model(x)
    cross_entropy_loss = criterion(y_pred, y)
    l2_reg = torch.tensor(0.0, requires_grad=True)
    for p in model.parameters():
        l2_reg = l2_reg + cfg.l2_norm_lambda * p.norm(2)
    loss = cross_entropy_loss + l2_reg
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), cross_entropy_loss.item(), l2_reg.item()


def train():
    trainer = Engine(make_step)
    evaluator_train = create_supervised_evaluator(model, metrics={"accuracy": Accuracy()})
    evaluator_test = create_supervised_evaluator(model, metrics={"accuracy": Accuracy(),
                                                                 "losses": CustomLossValidation(model,
                                                                                                cfg.l2_norm_lambda)})

    @trainer.on(Events.EPOCH_COMPLETED(every=1))
    def evaluate(engine):
        e = engine.state.epoch
        state_train = evaluator_train.run(dl_train)
        print(f'Accuracy on train data: {state_train.metrics["accuracy"]}')
        mlflow.log_metric('train/accuracy', state_train.metrics['accuracy'], e)
        state_test = evaluator_test.run(dl_test)
        test_acc, (loss, cross_entropy_loss, reg_loss) = state_test.metrics["accuracy"], state_test.metrics["losses"]
        print(f'Accuracy on test data: {test_acc} | loss: {loss} | cross_entropy_loss: {cross_entropy_loss} | reg_loss: {reg_loss}')
        mlflow.log_metric('test/accuracy', test_acc, e)
        mlflow.log_metric('test/cross_entropy_loss', cross_entropy_loss, e)
        mlflow.log_metric('test/loss', loss, e)
        mlflow.log_metric('test/reg_loss', reg_loss, e)
        torch.save(model.state_dict(), os.path.join(cfg.checkpoints_dir, f'checkpoint_{e}.pth'))

    @trainer.on(Events.ITERATION_COMPLETED(every=100))
    def log_training(engine):
        loss, cross_entropy_loss, reg_loss = engine.state.output
        e = engine.state.epoch
        n = engine.state.max_epochs
        i = engine.state.iteration
        mlflow.log_metric('train/cross_entropy_loss', cross_entropy_loss, i)
        mlflow.log_metric('train/reg_loss', reg_loss, i)
        mlflow.log_metric('train/total_loss', loss, i)
        print(f'Epoch {e}/{n} | iteration: {i} | loss: {loss} | cross_entropy_loss: {cross_entropy_loss} | reg_loss: {reg_loss}')
    return trainer


if __name__ == '__main__':
    dl_train, dl_test = get_data()

    if cfg.model is 'LR':
        model = LogisticRegression(cfg.input_dim, cfg.output_dim).to(cfg.device)
    elif cfg.model is 'MLP1HL':
        model = MLP1HL(cfg.input_dim, cfg.output_dim).to(cfg.device)
    elif cfg.model is 'MLP2HL':
        model = MLP2HL(cfg.input_dim, cfg.output_dim).to(cfg.device)
    else:
        raise Exception

    print('\nModel parameters: ')
    print(model)

    nb_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters number: {nb_trainable_params}')

    criterion = get_criterion()
    optimizer = get_optimizer(model)
    trainer = train()

    start_time = time.time()
    trainer.run(dl_train, max_epochs=cfg.epochs)

    print(f'Total training time: {round((time.time() - start_time) / 60, 3)} min')
