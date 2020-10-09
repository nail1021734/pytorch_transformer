import torch
from typing import List
from typing import Iterable
from tokenizer import Tokenizer


class TranslateDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        x: List[str],
        y: List[str],
        tokenizer: Tokenizer,
        max_seq_len: int
    ):
        super().__init__()
        self.x = x
        self.y = y
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]

    def newRemove(self, targetlist, element):
        targetlist.remove(element)
        return targetlist

    def collate_fn(self, batch: Iterable[Iterable[str]]):
        batch_x = [data[0] for data in batch]
        batch_y = [data[1] for data in batch]
        data_x = self.tokenizer.batch_encode(
            batch_x,
            max_len=self.max_seq_len
        )
        data_tmp = self.tokenizer.batch_encode(
            batch_y,
            max_len=self.max_seq_len
        )
        data_y = [seq[1:] for seq in data_tmp]
        data_tgt = [
            self.newRemove(seq, self.tokenizer.eos_id[0]) for seq in data_tmp
        ]
        return data_x, data_tgt, data_y

# if __name__ == "__main__":
#     tokenizer = Tokenizer()
#     tokenizer.load_model('tokenizer')
#     with open('data/news-commentary-v13.zh-en.en','r') as input_file:
#         data_x = [sentence.strip() for sentence in input_file.readlines()]
#     with open('data/news-commentary-v13.zh-en.ch','r') as input_file:
#         data_y = [sentence.strip() for sentence in input_file.readlines()]
#     a = TranslateDataset(data_x, data_y, tokenizer, max_seq_len=-1)
#     data_loader = torch.utils.data.DataLoader(a, batch_size=32, collate_fn=a.collate_fn)
#     for x, tgt, y in tqdm(data_loader):


