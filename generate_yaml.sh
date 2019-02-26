#!/bin/bash
trap "exit" INT

FILE="./data/indo_preprocessed/config_crf_glove_indo_preprocessed.yaml"

for dropout_val in {0.1,0.3,0.5};
do
  for dropout2_val in {0.5,0.3,0.1};
  do
    for hidden_size_val in {64,80,128,256}
      do
        for l_dropout_val in {0.1,0.2}
        do
          for mask_dim_val in {20,30,40,50}
          do
            for c1_val in {0.1,0.2}
            do
              for c2_val in {0.01,0.02,0.04}
	      do
cat > $FILE <<EOF
common:
    exp_name: crf_rnn_indo_leiming_new_spell_new
    arch: AspectSent
    batch_size: 32

    #Map words to lower case
    embed_num: 10725
    #For elmo the emb_size is 1024
    embed_dim: 100
    mask_dim: $mask_dim_val
    if_update_embed: False

    # lstm
    #l_hidden_size: 256
    l_hidden_size: $hidden_size_val
    l_num_layers: 2 # forward and backward
    l_dropout: $l_dropout_val

    # penlaty
    C1: $c1_val
    C2: $c2_val
    if_reset: False

    opt: Adam
    dropout: $dropout_val
    dropout2: $dropout2_val
    epoch: 30
    lr: 0.003
    l2: 0.0
    adjust_every: 8
    clip_norm: 3
    k_fold: 6
    print_freq: 10
    finetune_embed: False
    # data processing
    if_replace: False
    training: True
    # traning
    if_gpu: True
    use_gpu: True
    # data_path: data/2014/data.pkl
    # dic_path: data/2014/dic.pkl

    #################Restaurant
    pretrained_embed_path: ../data/glove_indo.100d.txt
    embed_path: data/indo_preprocessed/vocab/local_emb.pkl
    data_path: data/indo_preprocessed/
    train_path: data/indo_preprocessed/train.csv.pkl
    valid_path: data/indo_preprocessed/dev.csv.pkl
    test_path: data/indo_preprocessed/test.csv.pkl
    dic_path: data/indo_preprocessed/vocab/dict.pkl


    model_path: data/models/
    log_path: data/logs/
    
    is_stanford_nlp: False
    elmo_config_file: ../data/Indo_Elmo/options.json
    elmo_weight_file: ../data/Indo_Elmo/weights.hdf5
    snapshot_dir: checkpoints/
EOF
                python3 train_crf_glove.py
              done
            done
          done
        done
      done
  done
done
