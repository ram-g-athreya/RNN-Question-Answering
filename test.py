

import torch, os
data_dir = 'data/lc-quad'
train_dir = os.path.join(data_dir, 'train/')
train_file = os.path.join(data_dir, 'pth/lc_quad_train.pth')

dataset = torch.load(train_file)
print(dataset)