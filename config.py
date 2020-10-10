import os
import json
class Config:
    def __init__(self, **kwargs):
        self.experiment = kwargs['experiment']
        self.src_data = kwargs['src_data']
        self.tgt_data = kwargs['tgt_data']
        self.d_emb = kwargs['d_emb']
        self.head_num = kwargs['head_num']
        self.encoder_num = kwargs['encoder_num']
        self.decoder_num = kwargs['decoder_num']
        self.batch_size = kwargs['batch_size']
        self.epoch = kwargs['epoch']
        self.checkpoint_step = kwargs['checkpoint_step']
        self.tokenizer_train_data_path = kwargs['tokenizer_train_data_path']
    
    def __iter__(self):
        yield 'experiment', self.experiment
        yield 'src_data', self.src_data
        yield 'tgt_data', self.tgt_data
        yield 'd_emb', self.d_emb
        yield 'head_num', self.head_num
        yield 'encoder_num', self.encoder_num
        yield 'decoder_num', self.decoder_num
        yield 'batch_size', self.batch_size
        yield 'epoch', self.epoch
        yield 'checkpoint_step', self.checkpoint_step
        yield 'tokenizer_train_data_path', self.tokenizer_train_data_path

    @classmethod
    def load(experiment_number: int):
        file_path = os.path.join('data', str(experiment_number), 'config.json')
        
        with open(file_path, 'r', encoding='utf-8') as input_file:
            return cls(**json.load(input_file))

    def save(self):
        dir_path = os.path.join('data',str(self.experiment))
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        
        with open(
            os.path.join(dir_path,'config.json'),
            'w',
            encoding='utf8'
        ) as output:
            json.dump(dict(self), output, ensure_ascii=False)


