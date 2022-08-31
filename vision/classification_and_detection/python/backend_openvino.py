"""
openvino backend (https://docs.openvino.ai/latest/index.html)
"""

# pylint: disable=unused-argument,missing-docstring,useless-super-delegation

from openvino.runtime import Core, get_version

import backend

import numpy as np

class BackendOpenVINO(backend.Backend):
    def __init__(self):
        super(BackendOpenVINO, self).__init__()
        self.core = Core()
        
    def version(self):
        return get_version() 

    def name(self):
        """Name of the runtime."""
        return "openvino"

    def image_format(self):
        """image_format. """
        return "NHWC"

    def load(self, model_path, inputs=None, outputs=None):
        """Load model and find input/outputs from the model file."""
        model = self.core.read_model(model_path)
        self.compiled_model = self.core.compile_model(model, "CPU")
        
        # get input and output names
        if not inputs:
            self.inputs = model.inputs
        else:
            self.inputs = inputs
        if not outputs:
            self.outputs = model.outputs 
        else:
            self.outputs = outputs

        self.dynamic_shape = False 
        self.model_batch_size = 0
        input_shape = model.input(0).partial_shape
        if input_shape.is_dynamic :
            self.dynamic_shape = True
            print("WARNING: Dynamic Batch may cause performance drop. Please consider to fix batch size by Model Optimizer.")
        else:
            self.model_batch_size = len(input_shape[0])
        return self

    def predict(self, feed):
        """Run the prediction."""

        batch_shape = feed[self.inputs[0]].shape

        if batch_shape[0] == self.model_batch_size or self.dynamic_shape:
            input_tensor = feed
        elif batch_shape[0] < self.model_batch_size:
            pad = self.model_batch_size - batch_shape[0]
            input_tensor = {self.inputs[0]: np.pad(feed[self.inputs[0]], [ (0,pad), (0,0), (0,0), (0,0)])}
        else:
            input_tensor = feed
            raise ValueError("Input batch size ({}) is larger than model batch size ({}).".format(batch_shape[0], self.model_batch_size))

        results = self.compiled_model.infer_new_request(input_tensor)
        predictions = [ next(iter(results.values()))[:batch_shape[0]] ]
        return predictions
