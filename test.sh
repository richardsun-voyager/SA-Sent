#!/bin/bash
trap "exit" INT
for number in {1..200}
do
  python3 train_crf_glove.py
done
