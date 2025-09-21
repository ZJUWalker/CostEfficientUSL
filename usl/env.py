import random
import numpy as np
import torch

# 所有关于路径的拼接都必须要使用os.path.join()，不允许使用+，防止路径拼接错误
nltk_path = '~/nltk_data'  # 用于计算meteor指标数据集
dataset_cache_dir = f'data/datasets/'
model_path = f'data/models/'
model_save_path = f'data/ft_models'
naive_train_model_save_path = f'{model_save_path}/naive'
sl_model_save_path = f'{model_save_path}/sl'
log_dir = f'log'

train_log_dir = f'{log_dir}/train'


def set_random_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
