from easydict import EasyDict

cfg = EasyDict()

cfg.batch_size = 16#32
cfg.epochs = 50
cfg.input_dim = 784  # 1024
cfg.output_dim = 10
cfg.lr = 0.01
cfg.num_workers = 4
cfg.sz_crop = 28
cfg.l2_norm_lambda = 1e-5
cfg.checkpoints_dir = 'saved_models'

cfg.model = 'LR'  # ['LR', 'MLP1HL', 'MLP2HL']
cfg.log_metrics = False
cfg.use_profiler = False
cfg.use_const_multiplication = True  # is used in LinearRegression model
# const for custom class
cfg.const = 0.9
