from typing import List
from typing import Iterable
import sentencepiece as sp


class Tokenizer:
    def __init__(self):
        self.vocab_size = None
        self.model = sp.SentencePieceProcessor()

        self.bos_token = '<s>'
        self.pad_token = '<pad>'
        self.eos_token = '</s>'
        self.unk_token = '<unk>'

        self.bos_id = [1]
        self.pad_id = [0]
        self.eos_id = [2]
        self.unk_id = [3]

    def train_and_save_model(
        self,
        data_path: str,
        max_vocab_size: int,
        model_name: str,
    ):

        sp.SentencePieceTrainer.Train(
            f'--input={data_path} '
            f'--model_prefix={model_name} '
            f'--vocab_size={max_vocab_size} '
            f'--bos_id=1 '
            f'--eos_id=2 '
            f'--unk_id=3 '
            f'--pad_id=0 '
        )

        self.model.Load(f'{model_name}.model')
        self.vocab_size = self.model.get_piece_size()

    def load_model(self, model_name):
        self.model.Load(f'{model_name}.model')
        self.vocab_size = self.model.get_piece_size()

    def tokenize(self, sentence: str) -> List[str]:
        return self.model.Encode(sentence, out_type=str)

    def batch_tokenize(self, batch_sentences: List[str]) -> List[List[str]]:
        return [self.tokenize(sentence) for sentence in batch_sentences]

    def detokenize(self, tokens: List[str]) -> str:
        return self.model.DecodePieces(tokens)

    def batch_detokenize(self, batch_sentences: List[List[str]]) -> List[str]:
        return [self.detokenize(sentence) for sentence in batch_sentences]

    def encode(self, sentence: str, max_len: int) -> List[int]:
        if max_len != -1:
            sentence = self.tokenize(sentence)[:max_len - 2]
            sentence = "".join(sentence)

        token_ids = self.model.Encode(sentence, add_bos=True, add_eos=True)
        pad_len = max_len - len(token_ids)

        return token_ids + self.pad_id * pad_len

    def decode(self, token_ids: List[int]) -> str:
        return self.model.Decode(token_ids)

    def batch_encode(
        self,
        batch_sequences: List[str],
        max_len: int
    ) -> List[List[int]]:

        if max_len == -1:
            max_len = max([0] + list(map(
                len,
                [self.tokenize(sequence) for sequence in batch_sequences]
            ))) + 2
        return [
            self.encode(sequence, max_len) for sequence in batch_sequences
        ]

    def batch_decode(self, batch_sequences: List[List[int]]) -> List[str]:
        return [
            self.decode(sequence) for sequence in batch_sequences
        ]

    def convert_token_to_id(self, token: str) -> int:
        return self.model.PieceToId(token)

    def convert_id_to_token(self, id: int) -> str:
        return self.model.IdToPiece(id)

    def vocab_len(self):
        return self.vocab_size
