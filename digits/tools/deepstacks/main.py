#!/usr/bin/env python
# coding:utf-8
# vi:tabstop=4:shiftwidth=4:expandtab:sts=4

import deepstacks
from deepstacks.framework.main import *
import deepstacks.framework.using_lmdb
import deepstacks.framework.inference_logging
from deepstacks.macros import *
from deepstacks.framework.macros import *
from deepstacks.lasagne import curr_layer,curr_stacks,curr_flags,curr_model,curr_batchsize

if __name__ == '__main__':
    main()
