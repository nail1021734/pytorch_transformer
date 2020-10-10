import argparse
import torch
import os

from tokenizer import Tokenizer
from model import Transformer
from config import Config

def generate(
    x: str,
    beam_width: int,
    device: torch.device,
    max_seq_len: int,
    model: Transformer,
    tokenizer: Tokenizer
) -> str:
    model.eval()
    seq = torch.LongTensor([tokenizer.bos_id]).to(device)
    x = torch.LongTensor([tokenizer.encode(x, max_len=-1)]).to(device)

    accum_prob = torch.zeros(beam_width).to(device)

    for _ in range(max_seq_len):
        pred_y = model.predict(x, seq)

        top_k_in_all_beams = []
        for out_beams in range(seq.size(0)):
            top_k_prob_in_beam, top_k_index_in_beam = \
                pred_y[out_beams, -1].topk(
                    k=beam_width,
                    dim=-1
                )
            for in_beam in range(beam_width):

                prob = accum_prob[out_beams] -\
                    top_k_prob_in_beam[in_beam].log()
                prob = prob.unsqueeze(0)

                temp_seq = torch.cat([
                    seq[out_beams],
                    top_k_index_in_beam[in_beam].unsqueeze(0)
                ], dim=-1).unsqueeze(0)

                top_k_in_all_beams.append({
                    'prob': prob,
                    'seq': temp_seq
                })

        _, top_k_index_in_all_beams = torch.cat([
            beam['prob'] for beam in top_k_in_all_beams
        ]).topk(k=beam_width, dim=0)

        seq = torch.cat([
            top_k_in_all_beams[index]['seq']
            for index in top_k_index_in_all_beams
        ], dim=0)

        accum_prob = torch.cat([
            top_k_in_all_beams[index]['prob']
            for index in top_k_index_in_all_beams
        ], dim=0)

        if x.size(0) != seq.size(0):
            x = x.repeat(seq.size(0) // x.size(0), 1)

    for i in tokenizer.batch_decode(seq.tolist()):
        print(i)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input',
        help='input',
        required=True,
        type=str
    )
    parser.add_argument(
        '--width',
        help='beam search width',
        default=1,
        type=int
    )
    parser.add_argument(
        '--experiment',
        help='experiment_num',
        required=True,
        type=str
    )
    parser.add_argument(
        '--model_name',
        help='inference model name',
        required=True,
        type=str
    )
    parser.add_argument(
        '--max_seq_len',
        help='inference seq len',
        default=10,
        type=int
    )
    args = parser.parse_args()

    cfg = Config.load(args.experiment)

    tokenizer = Tokenizer()
    tokenizer.load_model('tokenizer')
    model = Transformer(
        d_model=cfg.d_emb,
        h=cfg.head_num,
        encoder_N=cfg.encoder_num,
        decoder_N=cfg.decoder_num,
        vocab_size=tokenizer.vocab_len(),
        pad_token_id=tokenizer.pad_id[0],
        dropout=0.1
    )
    model_path = os.path.join('data', str(args.experiment), args.model_name)
    model.load_state_dict(torch.load(f=model_path))
    model = model.to(device)
    x = args.input
    generate(
        x=x,
        beam_width=args.width,
        device=device,
        max_seq_len=args.max_seq_len,
        model=model,
        tokenizer=tokenizer
    )
