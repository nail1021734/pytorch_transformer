import argparse
import torch
import random
import numpy as np
import os
#import torch.utils.tensorboard

from tokenizer import Tokenizer
from dataset import TranslateDataset
from model import Transformer
from tqdm import tqdm
from optimizer import NoamOpt

# Parse argumernt from standard input.
parser = argparse.ArgumentParser()

parser.add_argument(
    '--src_data',
    help='src_data_path',
    required=True,
    type=str
)
parser.add_argument(
    '--tgt_data',
    help='tgt_data_path',
    required=True,
    type=str
)
parser.add_argument(
    '--d_emb',
    help='embedding dimension',
    required=True,
    type=int
)
parser.add_argument(
    '--head_num',
    help='head num',
    required=True,
    type=int
)
parser.add_argument(
    '--encoder_num',
    help='encoder layer num',
    required=True,
    type=int
)
parser.add_argument(
    '--decoder_num',
    help='decoder layer num',
    required=True,
    type=int
)
parser.add_argument(
    '--tokenizer_train_data_path',
    help='tokenizer training data',
    default='data/all_data',
    type=str
)
parser.add_argument(
    '--max_seq_len',
    help='max sequence length',
    default=50,
    type=int
)
parser.add_argument(
    '--batch_size',
    help='batch size',
    default=32,
    type=int
)
parser.add_argument(
    '--epoch',
    help='epoch',
    default=100,
    type=int
)
parser.add_argument(
    '--checkpoint_step',
    help='checkpoint step',
    default=100,
    type=int
)

args = parser.parse_args()

# Load or train tokenizer.
tokenizer = Tokenizer()
if os.path.exists('tokenizer.model'):
    tokenizer.load_model('tokenizer')
else:
    tokenizer.train_and_save_model(
        data_path=args.tokenizer_train_data_path,
        max_vocab_size=50000,
        model_name='tokenizer'
    )
    
# Load data.
with open(args.src_data, 'r') as input_file:
    data_x = [sentence.strip() for sentence in input_file.readlines()]
with open(args.tgt_data, 'r') as input_file:
    data_y = [sentence.strip() for sentence in input_file.readlines()]

# Tensorboard output path.
# writer = torch.utils.tensorboard.SummaryWriter('./log6')

dataset = TranslateDataset(
    x=data_x,
    y=data_y,
    tokenizer=tokenizer,
    max_seq_len=args.max_seq_len
)

data_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    shuffle=True,
    batch_size=args.batch_size,
    collate_fn=dataset.collate_fn,
)


model = Transformer(
    d_model=args.d_emb,
    h=args.head_num,
    encoder_N=args.encoder_num,
    decoder_N=args.decoder_num,
    vocab_size=tokenizer.vocab_len(),
    pad_token_id=tokenizer.pad_id[0],
    dropout=0.1
)
seed = 22
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

model.to(device)
model.train()
model.zero_grad()
criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = NoamOpt(
    model_size=args.d_emb,
    factor=2,
    warmup=4000,
    optimizer=torch.optim.Adam(
        model.parameters(),
        lr=0,
        betas=(0.9, 0.98),
        eps=1e-9
    )
)

epoch = args.epoch
total_loss = 0
iteration = 0
for i in range(epoch):
    epoch_iterator = tqdm(
        data_loader,
        desc=f'epoch: {i}, loss: {0:.6f}'
    )
    for src, tgt, y in epoch_iterator:
        iteration += 1
        src = torch.LongTensor(src).to(device)
        tgt = torch.LongTensor(tgt).to(device)
        y = torch.LongTensor(y).to(device)

        y = y.reshape(-1).to(device)
        pred = model(src, tgt)

        pred = pred.reshape(-1, tokenizer.vocab_len())
        loss = criterion(pred, y)
        epoch_iterator.set_description(
            f'epoch: {i}, loss: {loss.item():.6f}'
        )
        total_loss += loss.item()
        if iteration % args.checkpoint_step == 0:
            torch.save(
                model.state_dict(),
                os.path.join(
                    'data', f'model-{i}-epoch-{loss.item():.6f}-loss.pt')
            )
        # Tensorboard output
        # if iteration % 10 == 0:
        #     writer.add_scalar('loss', total_loss / 10, i)
        #     total_loss = 0
        loss.backward()
        optimizer.step()
        optimizer.optimizer.zero_grad()

torch.save(
    model.state_dict(),
    os.path.join('data', f'model-{i}-epoch-{loss.item():.6f}-loss.pt')
)
