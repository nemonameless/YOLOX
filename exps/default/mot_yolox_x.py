# for MOT ByteTrack, but diff in mosaic process

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 1
        self.data_num_workers = 4 #0

        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.train_ann = "train.json"
        self.val_ann = "val_half.json"
        self.input_size = (800, 1440)
        self.test_size = (800, 1440)
        self.random_size = (18, 32)

        self.max_epoch = 80
        self.print_interval = 20
        self.eval_interval = 1 #5
        self.test_conf = 0.1
        self.nmsthre = 0.7
        self.no_aug_epochs = 10

        self.basic_lr_per_img = 0.001 / 64.0
        self.warmup_epochs = 1

        self.mosaic_prob = 0.0 # 1.0
