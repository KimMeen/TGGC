#!/usr/bin/env bash

cd ..

python main.py --dataset 'PeMS04' --device 'cuda:2'  --run 1 --epoch 50 --train_length 6 --valid_length 2 --test_length 2 --gconv 'gegen' --activation=softmax --batch_size=64 --coe_a=0.48 --dropout_rate=0.31 --exponential_decay_step=10 --leakyrelu_rate=0.64 --lr=0.0012 --modes=4 --multi_layer=4 --optimizer=Adam --window_size 12  --horizon 3 --order 4
