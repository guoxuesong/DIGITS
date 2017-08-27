#!/usr/bin/env python
# coding:utf-8
# vi:tabstop=4:shiftwidth=4:expandtab:sts=4

import deepstacks
import deepstacks.framework.using_lmdb
from deepstacks.modelzoo import *
from deepstacks.macros import *
from deepstacks.framework.macros import *
network=(
        (0,slice(16,256-16),0,0,0,0,{'axis':2}),
        (0,slice(16,256-16),0,0,0,0,{'axis':3}),
        (googlenet,1000),
        (ln,'loss1/prob'),
        (classify,'target',0.3),
        (ln,'loss2/prob'),
        (classify,'target',0.3),
        (ln,'prob'),
        (classify,'target'),
        (ln,'prob'),
        )
