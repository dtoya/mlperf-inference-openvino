# coding=utf-8
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import array
import json
import os
import sys
sys.path.insert(0, os.getcwd())

import mlperf_loadgen as lg
import numpy as np
from openvino.runtime import Core
from transformers import BertConfig, BertForQuestionAnswering
from squad_QSL import get_squad_QSL

class BERT_OpenVINO_SUT():
    def __init__(self, args):
        self.profile = args.profile
        self.core = Core()

        print("Loading ONNX model...")
        self.quantized = args.quantized
        if self.quantized:
            model_path = "build/data/bert_tf_v1_1_large_fp32_384_v2/bert_large_v1_1_fake_quant.onnx"
        else:
            model_path = "build/data/bert_tf_v1_1_large_fp32_384_v2/model.onnx"
        model = self.core.read_model(model_path)

        config = {}
        if  'MLPERF_OPENVINO_CONFIG' in os.environ:
            file = os.environ['MLPERF_OPENVINO_CONFIG'] 
            if os.path.exists(file):
                with open(file) as f:
                    config.update(json.load(f))
            else:
                print(f"WARNING: Not found OpenVINO config file {file}")
        self.set_config(config)

        device = 'CPU'
        if 'MLPERF_OPENVINO_DEVICE' in os.environ:
            device = os.environ['MLPERF_OPENVINO_DEVICE']

        self.compiled_model = self.core.compile_model(model, device)
        self.query_devices(device)

        print("Constructing SUT...")
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries, self.process_latencies)
        print("Finished constructing SUT.")

        self.qsl = get_squad_QSL(args.max_examples)

    def issue_queries(self, query_samples):
        for i in range(len(query_samples)):
            eval_features = self.qsl.get_features(query_samples[i].index)
            if self.quantized:
                fd = {
                    "input_ids": np.array(eval_features.input_ids).astype(np.int64)[np.newaxis, :],
                    "attention_mask": np.array(eval_features.input_mask).astype(np.int64)[np.newaxis, :],
                    "token_type_ids": np.array(eval_features.segment_ids).astype(np.int64)[np.newaxis, :]
                }
            else:
                fd = {
                    "input_ids": np.array(eval_features.input_ids).astype(np.int64)[np.newaxis, :],
                    "input_mask": np.array(eval_features.input_mask).astype(np.int64)[np.newaxis, :],
                    "segment_ids": np.array(eval_features.segment_ids).astype(np.int64)[np.newaxis, :]
                }
            scores = self.compiled_model.infer_new_request(fd)
            #scores = self.sess.run([o.name for o in self.sess.get_outputs()], fd)
            output = np.stack(scores, axis=-1)[0]

            response_array = array.array("B", output.tobytes())
            bi = response_array.buffer_info()
            response = lg.QuerySampleResponse(query_samples[i].id, bi[0], bi[1])
            lg.QuerySamplesComplete([response])

    def flush_queries(self):
        pass

    def process_latencies(self, latencies_ns):
        pass

    def __del__(self):
        print("Finished destroying SUT.")

def get_openvino_sut(args):
    return BERT_OpenVINO_SUT(args)
