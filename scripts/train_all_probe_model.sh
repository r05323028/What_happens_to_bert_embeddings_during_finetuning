#!/bin/bash

python train_probe.py --bert_model=bert-base-uncased --output_dir=./models/bert-base-probe --n_epochs=40
python train_probe.py --bert_model=./models/mnli --output_dir=./models/bert-mnli-probe --n_epochs=40
python train_probe.py --bert_model=./models/squad --output_dir=./models/bert-squad-probe --n_epochs=40