# Transformer_prac

# Install

```shell
pip install -r requirements.txt
```
## Dataset

Please put in `data` folder.

## Train model

```shell
python train.py \
--experiment 1 \
--src_data 'data/new-en' \
--tgt_data 'data/new-zh' \
--d_emb 512 \
--head_num 8 \
--encoder_num 1 \
--decoder_num 1 \
--batch_size 32 \
--epoch 1000 \
--checkpoint_step 100 \
--tokenizer_train_data_path 'data/all_news'
```

## Inference

```shell
python translate.py \
--input 'What Failed in 2008?' \
--width 1 \
--experiment 1 \
--model_name 'model-2-epoch-55.456154-loss.pt' \
--max_seq_len 10
```
**Please modify `model_name` to fixed parameters**