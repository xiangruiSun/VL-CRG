# coding=utf-8
# Multitask Pretraining with MLM, MRC, MRFR, SRC, OPR, CRG

import collections
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random

from param import args
from pretrain.lxmert_data import LXMERTDataset, LXMERTTorchDataset, LXMERTEvaluator
from lxrt.entry import set_visual_config
from lxrt.modeling import LXRTMultiTask
from lxrt.tokenization import BertTokenizer

DataTuple = collections.namedtuple("DataTuple", 'dataset torchdset loader evaluator')

def get_tuple(splits: str, bs: int, shuffle=False, drop_last=False, topk=-1) -> DataTuple:
    dset = LXMERTDataset(splits)
    tset = LXMERTTorchDataset(dset, topk)
    data_loader = DataLoader(
        tset, batch_size=bs, shuffle=shuffle,
        num_workers=args.num_workers, collate_fn=lambda x: x,
        drop_last=drop_last, pin_memory=True
    )
    evaluator = LXMERTEvaluator(dset)
    return DataTuple(dataset=dset, torchdset=tset, loader=data_loader, evaluator=evaluator)

train_tuple = get_tuple(args.train, args.batch_size, shuffle=True, drop_last=True)
valid_tuple = get_tuple(args.valid, 512, shuffle=False, drop_last=False, topk=5000)

LOSSES_NAME = ('MLM', 'MRC', 'MRFR', 'SRC', 'OPR', 'CRG')
MIX_RATIO = [10, 1, 1, 1, 1, 1]

def sample_task():
    weights = [r / sum(MIX_RATIO) for r in MIX_RATIO]
    return random.choices(LOSSES_NAME, weights=weights, k=1)[0]

class MultiTaskLXMERT:
    def __init__(self, max_seq_length):
        self.max_seq_length = max_seq_length
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

        set_visual_config(args)
        self.model = LXRTMultiTask.from_pretrained(
            "bert-base-uncased",
            tasks=LOSSES_NAME,
            visual_losses=args.visual_losses
        )

        self.model = self.model.cuda()
        if args.multiGPU:
            self.model = nn.DataParallel(self.model)

    def forward(self, examples, task):
        # Convert to features...
        features = [convert_example_to_features(e, self.max_seq_length, self.tokenizer) for e in examples]
        # Prepare inputs based on task (code omitted for brevity)
        # ...
        return self.model(task, **prepared_inputs)

    def train_batch(self, optim, batch):
        task = sample_task()
        optim.zero_grad()
        loss, task_losses = self.forward(batch, task)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
        optim.step()
        return loss.item(), task_losses.detach().cpu()

    def train(self, train_tuple: DataTuple, eval_tuple: DataTuple):
        from lxrt.optimization import BertAdam
        train_ld = train_tuple.loader
        t_total = len(train_ld) * args.epochs
        optim = BertAdam(self.model.parameters(), lr=args.lr, warmup=0.05, t_total=t_total)

        for epoch in range(args.epochs):
            self.model.train()
            total_loss = 0.
            for batch in train_ld:
                loss, _ = self.train_batch(optim, batch)
                total_loss += loss
            print(f"Epoch {epoch}: avg loss = {total_loss / len(train_ld):.4f}")

        self.save("final_multitask")

    def save(self, name):
        torch.save(self.model.state_dict(), f"{args.output}/{name}_LXMERT.pth")

if __name__ == "__main__":
    lxmert = MultiTaskLXMERT(max_seq_length=20)
    lxmert.train(train_tuple, valid_tuple)
