#!/bin/bash

mo --input_model ./models/resnet50_v1.pb --input "input_tensor" --output "ArgMax" -b 32 -o models
mo --input_model ./models/resnet50_v1.pb --input "input_tensor" --output "ArgMax" -n resnet50_v1_ds -o models
