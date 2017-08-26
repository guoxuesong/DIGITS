import os
import re
import subprocess
import tempfile
import sys

from .framework import Framework
import digits
from digits.model.tasks import DeepstacksTrainTask
from digits.utils import subclass, override, constants

@subclass
class DeepstacksFramework(Framework):
    NAME = 'Deepstacks'
    CLASS = 'deepstacks'
    CAN_SHUFFLE_DATA = False
    SUPPORTS_PYTHON_LAYERS_FILE = False
    SUPPORTS_TIMELINE_TRACING = False
    SUPPORTED_SOLVER_TYPES = ['SGD', 'MOMENTUM', 'ADAM']
    SUPPORTED_DATA_TRANSFORMATION_TYPES = ['MEAN_SUBTRACTION']
    def __init__(self):
        super(DeepstacksFramework, self).__init__()
        # id must be unique
        self.framework_id = self.CLASS
    @override
    def create_train_task(self, **kwargs):
        """
        create train task
        """
        return DeepstacksTrainTask(framework_id=self.framework_id, **kwargs)
    @override
    def get_standard_network_desc(self, network):
        """
        return description of standard network
        """
        networks_dir = os.path.join(os.path.dirname(digits.__file__), 'standard-networks', self.CLASS)

        for filename in os.listdir(networks_dir):
            path = os.path.join(networks_dir, filename)
            if os.path.isfile(path):
                match = None
                match = re.match(r'%s.py$' % network, filename)
                if match:
                    with open(path) as infile:
                        return infile.read()
        # return None if not found
        return None
    @override
    def get_network_from_desc(self, network_desc):
        """
        return network object from a string representation
        """
        # return the same string
        return network_desc
    @override
    def validate_network(self, data):
        """
        validate a network
        """
        return True
