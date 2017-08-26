import deepstacks
from deepstacks.modelzoo import *
from deepstacks.macros import *
from deepstacks.framework.macros import *
network=(
        (googlenet,1000),
        (ln,'loss1/prob'),
        (classify,'y',0.3),
        (ln,'loss2/prob'),
        (classify,'y',0.3),
        (ln,'prob'),
        (classify,'y'),
        (ln,'prob'),
        )
