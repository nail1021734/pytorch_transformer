import torch
import tqdm
from tokenizer import Tokenizer
import model
import argparse
import dataset
import numpy as np
import random
import os


def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bos', type=str)
    parser.add_argument('--tokenizer', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--max_len', type=int)
    return parser


def gen(bos, model, device, tokenizer, max_len):
    model.to(device)
    model.eval()

    src = torch.LongTensor(tokenizer.tokenize([bos])).to(device)

    tgt = torch.empty(size=(1,max_len),dtype=torch.long).fill_(tokenizer.model.pad_id()).to(device)
    tgt[0, 0] = tokenizer.model.bos_id()
    cur_len = 1

    while cur_len < max_len:
        pred = model(src=src, tgt=tgt[:,:cur_len])
        r = torch.argmax(torch.nn.functional.softmax(pred, dim=-1), dim=-1)
        tgt[:,cur_len] = r[:,cur_len-1]
        # print(tgt[:,:cur_len])
        cur_len += 1
        if r[0, -1] == tokenizer.model.eos_id():
            break
    det = tokenizer.detokenize(tgt[0,:cur_len].cpu().tolist())
    print(det)


def main(args):
    tk = Tokenizer(args.tokenizer)

    model = model.Transformer(
        d_model=512,
        h=8,
        encoder_N=3,
        decoder_N=3,
        vocab_size=tokenizer.vocab_len(),
        pad_token_id=0,
        dropout=0.1
    )

    model.load_state_dict(torch.load(args.model)['model'])

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    gen(
        bos=args.bos,
        model=model,
        device=device,
        tokenizer=tk,
        max_len=args.max_len,
        )


if __name__ == '__main__':
    parser = create_args()
    main(parser.parse_args())
