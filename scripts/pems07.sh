#!/usr/bin/env bash

cd ..

python main.py --dataset 'PeMS07' --device 'cuda:3'  --run 1 --epoch 50 --train_length 7 --valid_length 2 --test_length 1 --gconv 'gegen' --coe_a 1  --batch_size 50 --modes 4 --multi_layer 5 --window_size 12  --horizon 3
  

