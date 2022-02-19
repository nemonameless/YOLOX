#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.multiscale_range = 0 #

        self.max_epoch = 36
        self.warmup_epochs = 2
        self.no_aug_epochs = 6
        self.print_interval = 100
        self.eval_interval = 1
