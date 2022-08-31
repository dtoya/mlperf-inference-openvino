#!/bin/bash
time ./run_local.sh openvino resnet50 cpu --scenario Offline 2>&1 | tee log-offline.txt
time ./run_local.sh openvino resnet50 cpu --scenario Offline --max-batchsize 8 2>&1 | tee log-offline-bs8.txt
#time ./run_local.sh openvino resnet50 cpu --scenario Offline --model='./models/resnet50_v1_ds.xml' 2>&1 | tee log-offline-ds.txt
time ./run_local.sh openvino resnet50 cpu --scenario SingleStream --accuracy 2>&1 | tee log-singlestream.txt
time ./run_local.sh openvino resnet50 cpu --scenario MultiStream --accuracy 2>&1 | tee log-multistream.txt
time ./run_local.sh openvino resnet50 cpu --scenario Server --accuracy 2>&1 | tee log-server.txt

time ./run_local.sh openvino resnet50 cpu --scenario Offline --accuracy 2>&1 | tee log-offline-acc.txt
time ./run_local.sh openvino resnet50 cpu --scenario Offline --max-batchsize 8 --accuracy 2>&1 | tee log-offline-bs8-acc.txt
time ./run_local.sh openvino resnet50 cpu --scenario Offline --model='./models/resnet50_v1_ds.xml' --accuracy 2>&1 | tee log-offline-ds-acc.txt

