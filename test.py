import torch


train_file = 'data/lc-quad/pth/lc_quad_train.pth'

train_dataset = torch.load(train_file)

tree = train_dataset.trees[0]

for idx in range(tree.num_children):
    print(tree.children[idx])

