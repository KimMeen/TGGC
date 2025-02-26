#!/usr/bin/env bash

cd ..

python main.py --dataset 'PeMS03' --device 'cuda:0'  --run 1 --epoch 50 --train_length 6 --valid_length 2 --test_length 2 --gconv 'gegen' --activation=softmax --batch_size=32 --coe_a=3.08  --dropout_rate=0.35 --exponential_decay_step=15 --leakyrelu_rate=0.26 --lr=0.00036 --modes=5 --multi_layer=5 --optimizer=Adam --window_size 12  --horizon 3 --order 4
