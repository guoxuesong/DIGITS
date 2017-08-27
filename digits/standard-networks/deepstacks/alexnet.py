#!/usr/bin/env python
# coding:utf-8
# vi:tabstop=4:shiftwidth=4:expandtab:sts=4

import deepstacks
import deepstacks.framework.using_lmdb
from deepstacks.modelzoo import *
from deepstacks.macros import *
from deepstacks.framework.macros import *
def alexnet(n):
    return (
            (0,96,11,4,0,0,{}),
            (0,0,3,2,0,0,{'maxpool'}),
            (0,256,5,1,0,0,{}),
            (0,0,3,2,0,0,{'maxpool'}),
            (0,384,3,1,0,0,{}),
            (0,384,3,1,0,0,{}),
            (0,256,3,1,0,0,{}),
            (0,0,3,2,0,0,{'maxpool'}),
            (0,4096,0,0,0,0,{'dense'}),
            (0,0,0,0,0,0,{'dropout':0.5}),
            (0,4096,0,0,0,0,{'dense'}),
            (0,0,0,0,0,0,{'dropout':0.5}),
            (0,n,0,0,0,0,{
                'dense':True,
                'nonlinearity':lasagne.nonlinearities.softmax,
                }),
            )
network=(
        (0,slice(16,256-16),0,0,0,0,{'axis':2}),
        (0,slice(16,256-16),0,0,0,0,{'axis':3}),
        (alexnet,1000),
        (classify,'target'),
        (ln,'prob'),
        )
