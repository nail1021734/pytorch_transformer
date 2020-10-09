# Transformer_prac

## Dataset
Please put in `data` folder.

## Train model
```shell
python train.py \
--src_data 'data/new-en' \
--tgt_data 'data/new-zh' \
--d_emb 512 \
--head_num 8 \
--encoder_num 1 \
--decoder_num 1 \
--batch_size 32 \
--epoch 100 \
--checkpoint_step 100 \
--tokenizer_train_data_path 'data/all_news'
```
##