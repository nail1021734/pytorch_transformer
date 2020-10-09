import torch
from tokenizer import Tokenizer
from model import Transformer

def generate(
    x: str,
    beam_width: int,
    device: torch.device,
    max_seq_len: int,
    model: Transformer,
    tokenizer: Tokenizer
) -> str:
    model.eval()
    # xx = '還有，對於那些'
    # seq = torch.LongTensor([tokenizer.encode(xx, max_len=-1)]).to(device)
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

        # index = torch.argmax(pred_y[:, -1], dim=-1)
        # print(pred_y.shape)
        # seq = torch.cat((seq, index.unsqueeze(0)), dim=-1)
    # print(seq)
    print(tokenizer.batch_decode(seq.tolist()))
    for i in tokenizer.batch_decode(seq.tolist()):
        print(i)


if __name__ == "__main__":
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    tokenizer = Tokenizer()
    tokenizer.load_model('tokenizer')
    model = Transformer(
        d_model=512,
        h=8,
        encoder_N=1,
        decoder_N=1,
        vocab_size=tokenizer.vocab_len(),
        pad_token_id=0,
        dropout=0.1
    )
    # model = TransformerModel(
    #     d_emb=512,
    #     num_linear_layers=2048,
    #     num_rnn_layers=1,
    #     vocab_size=tokenizer.vocab_len(),
    #     pad_token_id=tokenizer.pad_id[0],
    #     dropout=0.1
    # )
    model.load_state_dict(torch.load(
        f='linear_data5/model-299-epoch-0.009380-loss-N=1.pt'))
    model = model.to(device)
    x = 'Everyone seems to be a loser, even if some are more affected than others.'
    generate(x=x, beam_width=2, device=device, max_seq_len=20,
             model=model, tokenizer=tokenizer)
