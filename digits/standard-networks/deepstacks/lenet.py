#!/usr/bin/env python
# coding:utf-8
# vi:tabstop=4:shiftwidth=4:expandtab:sts=4

import deepstacks
import deepstacks.framework.using_lmdb
from deepstacks.modelzoo import *
from deepstacks.macros import *
from deepstacks.framework.macros import *
def lenet(n):
    return (
            (0,20,5,1,0,0,{}),
            (0,0,2,2,0,0,{'maxpool'}),
            (0,50,5,1,0,0,{}),
            (0,0,2,2,0,0,{'maxpool'}),
            (0,0,0,0,0,0,{'dropout':0.5}),
            (0,500,0,0,0,0,{'dense'}),
            (0,0,0,0,0,0,{'dropout':0.5}),
            (0,n,0,0,0,0,{
                'dense':True,
                'nonlinearity':lasagne.nonlinearities.softmax,
                }),
            )
network=(
        (0,0,0,0,0,0,{'nonlinearity':lambda x:x*0.0125}),
        (lenet,10),
        (classify,'target'),
        (ln,'prob'),
        )
