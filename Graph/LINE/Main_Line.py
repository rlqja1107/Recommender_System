from LINE_CLASS import Line
import os
import sys
sys.path.append(os.getcwd())
config = {
    'epoch': 200,
    'data_path': '../data/ind.citeseer',
    'n_dim': 128,
    'order': 2,
    'batch_size': 15,
    'neg_sample_size': 5,
    'lr': 0.025
}
line = Line(config).cuda()
print("Training Start")
Line.run(line)


