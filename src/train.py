import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import tqdm

from bert import BERT
from bertLM import BERTLM
from datasets import BERTDataset

from preprocess import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BERTTrainer:
    def __init__(
        self, 
        model, 
        train_dataloader, 
        test_dataloader=None, 
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        warmup_steps=10000,
        log_freq=10,
        device='cuda'
    ):
        self.device = device
        self.model = model.to(self.device)
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.scheduler = get_linear_schedule_with_warmup(self.optim, warmup_steps, len(train_dataloader))

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = nn.NLLLoss(ignore_index=0)

        self.log_freq = log_freq
        print(f"Total Parameters: {sum(p.numel() for p in self.model.parameters())}")

    def train(self, epoch):
        self.iteration(epoch, self.train_data, train=True)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        mode = "train" if train else "test"
        desc = f"EP_{mode}:{epoch}"
        data_iter = tqdm(data_loader, desc=desc, total=len(data_loader), bar_format="{l_bar}{r_bar}")
        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        for i, data in enumerate(data_iter, start=1):
            self.model.train(train)
            self.optim.zero_grad()
            inputs = {"input_ids": data["input_ids"].to(self.device),
                      "attention_mask": data["attention_mask"].to(self.device),
                      "token_type_ids": data["token_type_ids"].to(self.device),
                      "labels": data["labels"].to(self.device)}
            outputs = self.model(**inputs)
            loss = self.criterion(outputs.logits.view(-1, outputs.logits.shape[-1]), inputs["labels"].view(-1))
            if train:
                loss.backward()
                self.optim.step()
                self.scheduler.step()
            correct = torch.argmax(outputs.logits, dim=-1).eq(inputs["labels"]).sum().item()
            avg_loss += loss.item()
            total_correct += correct
            total_element += inputs["labels"].nelement()

            if i % self.log_freq == 0 or i == len(data_iter):
                postfix = {"epoch": epoch,
                           "iter": i,
                           "avg_loss": avg_loss / i,
                           "avg_acc": total_correct / total_element * 100,
                           "loss": loss.item()}
                data_iter.write(str(postfix))
        avg_loss /= len(data_iter)
        total_acc = total_correct / total_element * 100
        print(f"EP{epoch}, {mode}: avg_loss={avg_loss}, total_acc={total_acc}")

if __name__ == "__main__":
    conv_path, line_path = download_and_extract_data()
    pairs = get_dialogue_pairs(conv_path, line_path)
    tokenizer = train_tokenizer(pairs)

    vocab_size = len(tokenizer.vocab)
    hidden_size = 768
    num_hidden_layers = 6
    num_attention_head = 12

    intermediate_size = 4 * hidden_size
    dropout = 0.1
    max_positional_embeddings = 512
    layer_norm_eps = 1e-12

    train_data = BERTDataset(pairs, seq_len=MAX_LEN, tokenizer=tokenizer)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, pin_memory=True)

    bert_model = BERT(
                num_layers=num_hidden_layers,
                num_heads=num_attention_head,
                d_model=hidden_size,
                fully_connected_dim=intermediate_size,
                input_vocab_size=vocab_size,
                maximum_position_encoding=max_positional_embeddings,
                dropout_rate=dropout,
                layernorm_eps=layer_norm_eps
            )
    bert_lm = BERTLM(bert_model, len(tokenizer.vocab))

    bert_trainer = BERTTrainer(bert_lm, train_loader, device=device)
    epochs = 5

    for epoch in range(epochs):
        bert_trainer.train(epoch)
        