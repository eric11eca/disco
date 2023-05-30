import string
import torch
import numpy as np
import textdistance as tdist

from dataclasses import dataclass
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from distiller.db import update, delete

class BaseFilter():
    models: dict = {}

    def _normalize(self, text):
        return self._strip_punctuation_and_casing(text)
    
    def _strip_punctuation_and_casing(self, s):
        s = s.lower()
        s = s.translate(str.maketrans('', '', string.punctuation))
        return s
    
    def run(self, outputs, cache, **kwargs):
        raise NotImplementedError

@dataclass
class HFModelContainer():
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    label2id: dict
    id2label: dict
    name: str

    def __dict__(self):
        return {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "label2id": self.label2id,
            "id2label": self.id2label,
            "name": self.name
        }
    
class FilterDataLoader():
    def __init__(self, dataset, batch_size):
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: x
        )


class SentencePairHeuristicFilter(BaseFilter):
    def heuristic_filtering(self, data, mode=None, forbidden=[], negations=[]):
        sentence1 = self._normalize(data['sentence1'])
        sentence2 = self._normalize(data['sentence2'])
        gen_out = self._normalize(data['gen_out'])
        if mode == "sentence1":
            new_input = self._normalize(data['new_sentence1'])
        else:
            new_input = self._normalize(data['new_sentence2'])

        if gen_out == sentence2:
            return True, "copied_sentence2"
        elif sentence2 in gen_out:
            return True, "coverd_sentence2"
        elif gen_out in sentence2:
            return True, "coverd_by_sentence2"
        elif gen_out in sentence1:
            return True, "coverd_by_sentence1"
        elif tdist.jaccard(gen_out.split(), sentence2.split()) > 0.6:
            return True, "overlap_sentence2"
        elif tdist.jaccard(gen_out.split(), sentence1.split()) > 0.35:
            return True, "overlap_sentence1"
        elif np.any([x in data['gen_out'] for x in forbidden]):
            return True, "forbidden_phrase"
        elif data['new_label'] == "contradiction" and np.any([x in gen_out for x in negations]):
            return True, "negation_word"
        elif "_" in data['gen_out']:
            return True, "large_gap"
        elif "________" in data['gen_out']:
            return True, "large_gap"
        elif mode == "sentence1" and tdist.jaccard(new_input.split(), sentence2.split()) > 0.8:
            return True, "overlap_sentence2_all"
        elif mode == "sentence2" and tdist.jaccard(new_input.split(), sentence1.split()) > 0.8:
            return True, "overlap_sentence1_all"
        return False, "none"

    def run(self, outputs, cache, **kwargs):
        discards = {}
        accepted = []
        for record in outputs:
            blocked, reason = self.heuristic_filtering(
                record, record['mode'], 
                kwargs["forbidden"], kwargs["negations"])
            if blocked:
                delete(cache, {"gen_out": record['gen_out']})
                discards[reason] = discards.get(reason, 0) + 1
            else:
                accepted.append(record)
        return accepted


class SentencePairModelFilter(BaseFilter):
    def tensorize(self, original, counter, teacher, device):
        tokenizer = teacher.tokenizer
        input_seqs = [tokenizer(x[0],x[1]).input_ids for x in original]
        counter_seqs = [tokenizer(x[0],x[1]).input_ids for x in counter]

        input_seq_pair = tokenizer.batch_encode_plus(
            original,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max([len(x) for x in input_seqs]),
        )
        counter_seq_pair = tokenizer.batch_encode_plus(
            counter,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max([len(x) for x in counter_seqs]),
        )

        input_seq_pair = input_seq_pair.to(device)
        counter_seq_pair = counter_seq_pair.to(device)

        return input_seq_pair, counter_seq_pair

    def predict(self, original, counter, teacher, device):
        input_seq_pair, counter_seq_pair = self.tensorize(
            original, counter, teacher, device)
        
        model = teacher.model
        with torch.no_grad():
            logits = model(**input_seq_pair).logits
            logits_counter = model(**counter_seq_pair).logits
            preds = torch.softmax(logits, -1)
            counter_preds = torch.softmax(logits_counter, -1)
            return preds, counter_preds

    def encode_label(self, label, to_label, device):
        prev_label_pt = torch.Tensor(label).long().unsqueeze(-1)
        new_label_pt = torch.Tensor(to_label).long().unsqueeze(-1)
        prev_label_ids = torch.zeros(
            prev_label_pt.size()[0], 3).scatter_(-1, prev_label_pt, 1)
        new_label_ids = torch.zeros(
            new_label_pt.size()[0], 3).scatter_(-1, new_label_pt, 1)
        return prev_label_ids.to(device), new_label_ids.to(device)

    def critic_metric(self, preds, preds_counter, prev_label_ids, new_label_ids):
        natural = torch.sum(torch.mul(preds, prev_label_ids) -
                            torch.mul(preds_counter, prev_label_ids), -1)
        counter = torch.sum(torch.mul(preds_counter, new_label_ids) -
                            torch.mul(preds, new_label_ids), -1)

        return natural, counter

    def ensemble(self, scores, mode="softmax"):
        if (mode == "softmax"):
            scores_prob = torch.softmax(scores, -1)
            totals = torch.mul(scores_prob, scores)
            return torch.sum(totals, -1)

    def preprocess_batch(self, batch):
        original = [[record['sentence1'], record['sentence2']] for record in batch]
        if batch[0]['mode'] == "sentence1":
            counter = [(record['new_sentence1'], record['sentence2']) for record in batch]
        else:
            counter = [(record['sentence1'], record['new_sentence2']) for record in batch]
        return original, counter
    
    def load_model(self, model_name, device):
        print(f"Loading model {model_name}")
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model = model.to(device)
        model.eval()
        teacher = HFModelContainer(
            model=model,
            tokenizer=AutoTokenizer.from_pretrained(model_name),
            label2id = model.config.label2id,
            id2label = model.config.id2label,
            name=model_name
        )
        return teacher

    def run(self, batch, cache, **kwargs):
        scores1, scores2 = [], []
        original, counter = self.preprocess_batch(batch)
        
        for model_name in kwargs["model_names"]:
            if model_name not in self.models:
                teacher = self.load_model(model_name, kwargs["device"])
                self.models[model_name] = teacher
            else:
                teacher = self.models[model_name]
            src_label = [teacher.label2id[x['label']] for x in batch]
            tar_label = [teacher.label2id[x['new_label']] for x in batch]

            preds, counter_preds = self.predict(
                original, counter, teacher, kwargs["device"])
            prev_label_ids, new_label_ids = self.encode_label(
                src_label, tar_label, kwargs["device"])
            score = self.critic_metric(
                preds, counter_preds,
                prev_label_ids, new_label_ids)
            scores1.append(score[0])
            scores2.append(score[1])

        voting1 = self.ensemble(torch.stack(
            scores1, -1)).to("cpu").numpy().tolist()
        voting2 = self.ensemble(torch.stack(
            scores2, -1)).to("cpu").numpy().tolist()

        for i, s in enumerate(zip(voting1, voting2)):
            if s[1] > kwargs["threshold"] / 2:  # and s[1] > s[0]:
                accepted = True
            else:
                accepted = False
            update(cache, {"gen_out": batch[i]['gen_out']},
                {"$set": {"accept": accepted, "score": s[1]}})
            batch[i]['accept'] = accepted
            batch[i]['score'] = s[1]

        return batch
