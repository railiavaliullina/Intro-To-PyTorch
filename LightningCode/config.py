from easydict import EasyDict

cfg = EasyDict()

cfg.batch_size = 32
cfg.epochs = 50
cfg.input_dim = 784  # 1024
cfg.output_dim = 10
cfg.lr = 0.01
cfg.num_workers = 4
cfg.sz_crop = 28
cfg.l2_norm_lambda = 1e-5

cfg.device = 'cpu'  # ''cuda:0'
cfg.model = 'MLP2HL'  # ['LR', 'MLP1HL', 'MLP2HL']

