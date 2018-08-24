import torch
import csv

with open("analysis/t.csv", "w") as csv_file:
    writer = csv.writer(csv_file)
    x = torch.Tensor([1, 2, 3])
    y = x.numpy()

    a = torch.Tensor([4, 5, 6])
    b = a.numpy()

    writer.writerows(zip(y, b))