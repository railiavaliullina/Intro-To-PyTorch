import os
import tarfile
import torch
import time

from CatalystCode.config import cfg
from CatalystCode.model import LogisticRegression, MLP1HL, MLP2HL
from CatalystCode.data import get_data
from trains import Task
from CatalystCode.custom_callbacks import TbLogger

from catalyst import dl, metrics


class CustomRunner(dl.Runner):

    def predict_batch(self, batch):
        return self.model(batch[0].to(self.device).view(batch[0].size(0), -1))

    def _handle_batch(self, batch):
        x, y = batch
        y_hat = self.model(x.view(x.size(0), -1))

        cross_entropy_loss = self.criterion(y_hat, y)
        l2_reg = torch.tensor(0.0, requires_grad=True)
        for p in self.model.parameters():
            l2_reg = l2_reg + cfg.l2_norm_lambda * p.norm(2)
        loss = cross_entropy_loss + l2_reg

        accuracy01, accuracy03 = metrics.accuracy(y_hat, y, topk=(1, 3))

        self.batch_metrics.update(
            {"cross_entropy_loss": cross_entropy_loss, "accuracy01": accuracy01, "accuracy03": accuracy03,
             "reg_loss": l2_reg, "loss": loss})

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()


if __name__ == '__main__':
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

    loaders = get_data()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    task = Task.init(project_name=f'Catalyst exp', task_name=f'{cfg.model} test task')
    # task.output_uri = 'https://drive.google.com/drive/folders/1-LeDC01JWAaF38k1T59KhI95W1VN90eb?usp=sharing'
    # task.output_uri = 'file:///C:/Users/Admin/PycharmProjects/ML&CV/intro-to-pytorch/CatalystCode/allegro_tr'
    parameters = task.connect(cfg)
    task.add_tags(f'{cfg.model}')

    runner = CustomRunner()

    start_time = time.time()
    runner.train(
        model=model.cuda(),
        criterion=criterion,
        optimizer=optimizer,
        # scheduler=scheduler,
        loaders=loaders,
        logdir="./logs",
        num_epochs=cfg.epochs,
        verbose=True,
        load_best_on_end=True
        # callbacks=[TbLogger]
    )
    print(f'training time: {round((time.time() - start_time) / 60, 3)} min')

    for prediction in runner.predict_loader(loader=loaders["valid"]):
        assert prediction.detach().cpu().numpy().shape[-1] == 10

    traced_model = runner.trace(loader=loaders["valid"])

    print(f'Total training time: {round((time.time() - start_time) / 60, 3)} min')

    # for kaggle
    # files_to_save = os.listdir('/kaggle/working/')
    # print(files_to_save)
    # tar = tarfile.open(f'logs.tar.gz', 'w:gz')
    # for item in files_to_save:
    #     tar.add(item)
    # tar.close()
