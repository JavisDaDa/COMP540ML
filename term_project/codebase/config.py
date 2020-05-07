import os
import pandas as pd

class_list_path = os.path.join("class_list.txt")
train_label_dir = os.path.join("data", "train_labels.csv")
valid_label_dir = os.path.join("val_labels.csv")
test_label_dir = os.path.join("sample_submission.csv")
log_dir = './drive/My Drive/COMP540/runs/resnext101'
train_df = pd.read_csv(train_label_dir)
valid_df = pd.read_csv(valid_label_dir)

RESUME = False
MAX_EPOCH = 30          # training epoch
BATCH_SIZE = 32         # batch size
LR = 0.001              # learning rate
log_interval = 10
#check points
val_interval = 1        # checkpoint time
N_CLASSES = 251         # classes
TRAIN_SAMPLE = len(train_df)    # training sample
VALID_SAMPLE = len(valid_df)    # valid sample
checkpoint_interval = 1
momentum = 0.9
freeze_rate = 0.01
step_size = 5
gamma = 0.1
weight_decay = 1e-4
checkpoint_path = f'./drive/My Drive/COMP540/checkpoints/resnext101V2/'
path_checkpoint = './drive/My Drive/COMP540/checkpoints/resnext101V2/checkpoint_22_epoch.pkl'
model_path = "./drive/My Drive/COMP540/modelsaves/resnext101/resnext101V5_state_dict.pkl"
train_data_df = train_df.sample(TRAIN_SAMPLE)
valid_data_df = valid_df.sample(VALID_SAMPLE)
test_data_df = pd.read_csv(test_label_dir)

data_dir = os.path.join("data")
train_dir = os.path.join(data_dir, "train_set")
valid_dir = os.path.join(data_dir, "val_set")
test_dir = os.path.join(data_dir, "test_set")

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

name = 'resnet152'

if __name__ == '__main__':
    e = 6
    print(f'./drive/My Drive/COMP540/checkpoints/resnext101V2/' + f'checkpoint_{e}_epoch.pkl')