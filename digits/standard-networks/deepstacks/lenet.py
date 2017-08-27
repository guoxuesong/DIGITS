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

# Example for using raw lasagne:
#
#import deepstacks
#import lasagne
#from deepstacks.framework.main import *
#import deepstacks.framework.using_lmdb
#from deepstacks.macros import *
#from deepstacks.framework.macros import *
#def build_network(inputs):
#    network = inputs['image']
#    if 'mean' in inputs:
#        network=lasagne.layers.ElemwiseMergeLayer((network,inputs['mean']),T.sub)
#    network = lasagne.layers.Conv2DLayer(
#            network, num_filters=20, filter_size=(5, 5),
#            nonlinearity=lasagne.nonlinearities.rectify,
#            W=lasagne.init.GlorotUniform())
#    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
#    network = lasagne.layers.Conv2DLayer(
#            network, num_filters=50, filter_size=(5, 5),
#            nonlinearity=lasagne.nonlinearities.rectify,
#            W=lasagne.init.GlorotUniform())
#    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
#    network = lasagne.layers.DenseLayer(
#            lasagne.layers.dropout(network, p=.5),
#            num_units=500,
#            nonlinearity=lasagne.nonlinearities.rectify)
#    network = lasagne.layers.DenseLayer(
#            lasagne.layers.dropout(network, p=.5),
#            num_units=10,
#            nonlinearity=lasagne.nonlinearities.softmax)
#
#    res, stacks, paramlayers, errors, watchpoints=deepstacks.lasagne.build_network(network,(
#        (classify,'target'),
#        ),inputs)
#    return res,stacks,paramlayers+[network],errors,watchpoints
#
#
