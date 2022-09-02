# How to run MLPerf Inference Benchmarks with OpenVINO Backend for Image Classification and Object Detection Tasks 


## Installation 
```
$ git clone git@github.com:dtoya/mlperf-inference-openvino.git
$ cd mlperf-inference-openvino
$ git checkout r2.0_openvino
$ git submodule update --init --recursive
$ sudo apt install build-essential python3-dev

$ cd vision/classification_and_detection/
$ python3 -m venv venv
$ . venv/bin/activate
(venv) $ pip install -U pip
(venv) $ pip install openvino
(venv) $ pip install openvino-dev[tensorflow2]
(venv) $ cd ../../loadgen; CFLAGS="-std=c++14" python setup.py develop; cd ../vision/classification_and_detection
(venv) $ python setup.py develop
```

## Setup model and dataset
```
(venv) $ export MODEL_DIR=./models
(venv) $ mkdir models
(venv) $ wget https://zenodo.org/record/2535873/files/resnet50_v1.pb -O models/resnet50_v1.pb
(venv) $ ./tools/resnet50-to-openvino.sh
(venv) $ export DATA_DIR=./dataset/imagenet2012
(venv) $ # Create dataset directory and extract imagenet dataset under the directory.
```

## Example
```
(venv) $ ./run_local.sh openvino resnet50 cpu --scenario Offline
```
