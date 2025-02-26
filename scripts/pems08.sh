#!/usr/bin/env bash

cd ..

python main.py --dataset 'PeMS08' --device 'cuda:4'  --run 1 --epoch 50 --train_length 6 --valid_length 2 --test_length 2 --gconv 'gegen' --coe_a 1 --batch_size 50 --modes 4 --multi_layer 5 --window_size 12  --horizon 3
  

