import json
import string
import torch
import numpy as np
import textdistance as tdist

from tqdm import tqdm
from evaluate import load
from pprint import pprint
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from .db import update, query

class BaseFilter():
    def _normalize(self, text):
        return self._strip_punctuation_and_casing(text)
    
    def _strip_punctuation_and_casing(self, s):
        s = s.lower()
        s = s.translate(str.maketrans('', '', string.punctuation))
        return s
    
    def run(self, outputs, cache, mode):
        raise NotImplementedError


class SentencePairHeuristicFilter(BaseFilter):
    def heuristic_filtering(self, data, mode=None, forbidden=[], negations=[]):
        sentence1 = self._normalize(data["sentence1"])
        sentence2 = self._normalize(data["sentence2"])
        gen_out = self._normalize(data["gen_out"])
        new_input = self._normalize(data[f"new_{mode}"])

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
        elif np.any([x in data["gen_out"] for x in forbidden]):
            return True, "forbidden_phrase"
        elif data["new_label"] == "contradiction" and np.any([x in gen_out for x in negations]):
            return True, "negation_word"
        elif "_" in data["gen_out"]:
            return True, "large_gap"
        elif "________" in data["gen_out"]:
            return True, "large_gap"
        elif mode == "sentence1" and tdist.jaccard(new_input.split(), sentence2.split()) > 0.8:
            return True, "overlap_sentence2_all"
        elif mode == "sentence2" and tdist.jaccard(new_input.split(), sentence1.split()) > 0.8:
            return True, "overlap_sentence1_all"
        return False, "none"

    def run(self, outputs, cache, mode):
        discards = {}
        accepted = []
        for record in outputs:
            blocked, reason = self.heuristic_filtering(record, mode)
            if blocked:
                record["accept"] = False
                update(cache, {"guid": record["guid"]}, {
                       "$set": {"accept": False}})
                discards[reason] += 1
            else:
                accepted.append(record)
        pprint(discards)
        return accepted


class SentencePairModelFilter(BaseFilter):
    def __init__(self, cache, mode, model_names) -> None:
        self.tokenizers = {}
        self.models = {}
        self.configs = {}
        self.cache = cache
        self.mode = mode
        self.global_counter = 0

        for name in model_names:
            tokenizer = AutoTokenizer.from_pretrained(name)
            model = AutoModelForSequenceClassification.from_pretrained(name)
            config = AutoConfig.from_pretrained(name)
            model.cuda()
            model.eval()
            self.tokenizers[name] = tokenizer
            self.models[name] = model
            self.configs[name] = config

    def predict(self, batch, batch_counter, model_name):
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]

        input_seq_pair = tokenizer.batch_encode_plus(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
            return_token_type_ids=False
        )
        counter_seq_pair = tokenizer.batch_encode_plus(
            batch_counter,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
            return_token_type_ids=False
        )

        input_seq_pair = input_seq_pair.to(self.device)
        counter_seq_pair = counter_seq_pair.to(self.device)

        with torch.no_grad():
            logits = model(**input_seq_pair).logits
            logits_counter = model(**counter_seq_pair).logits

            preds = torch.softmax(logits, -1)
            counter_preds = torch.softmax(logits_counter, -1)

            return preds, counter_preds

    def encode_label(self, label, to_label):
        prev_label_pt = torch.Tensor(label).long().unsqueeze(-1)
        new_label_pt = torch.Tensor(to_label).long().unsqueeze(-1)
        prev_label_ids = torch.zeros(
            prev_label_pt.size()[0], 3).scatter_(-1, prev_label_pt, 1)
        new_label_ids = torch.zeros(
            new_label_pt.size()[0], 3).scatter_(-1, new_label_pt, 1)
        return prev_label_ids.to(self.device), new_label_ids.to(self.device)

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

    def preprocess_batch(self, counter_data):
        batch = list(zip(counter_data["premise"], counter_data["hypothesis"]))
        if self.mode == "premise":
            original = counter_data["premise"]
        else:
            original = counter_data["hypothesis"]
        perturbations = list(
            zip(original,
                counter_data["span_prev"],
                counter_data["gen_out"])
        )

        p_counter = [p.replace(s, gen) for p, s, gen in perturbations]
        if self.mode == "premise":
            batch_counter = list(zip(p_counter, counter_data["hypothesis"]))
        else:
            batch_counter = list(zip(counter_data["premise"], p_counter))
        label = {k: [v.label2id[x] for x in counter_data["label"]] for k, v in self.models.items()}
        to_label = {k: [v.label2id[x] for x in counter_data["new_label"]] for k, v in self.models.items()}
        return batch, batch_counter, label, to_label

    def post_process_batch(self, counter_data, batch_counter):
        filtered_data = []
        for i in range(len(batch_counter)):
            guid = counter_data["guid"][i]
            record = query(self.cache, {"guid": guid})
            if record["accept"]:
                filtered_data.append(record)
        return filtered_data

    def filter(self, counter_data, threshold=0.5):
        batch, batch_counter, label, to_label = self.preprocess_batch(
            counter_data)

        scores1 = []
        scores2 = []

        for model_name in self.hf_model_names:
            preds, counter_preds = self.predict(
                batch, batch_counter, model_name)
            prev_label_ids, new_label_ids = self.encode_label(
                label, to_label)
            score = self.critic_metric(preds, counter_preds,
                                       prev_label_ids, new_label_ids)
            scores1.append(score[0])
            scores2.append(score[1])

        voting1 = self.ensemble(torch.stack(
            scores1, -1)).to("cpu").numpy().tolist()
        voting2 = self.ensemble(torch.stack(
            scores2, -1)).to("cpu").numpy().tolist()

        for i, s in enumerate(zip(voting1, voting2)):
            guid = counter_data["guid"][i]
            record = query(self.cache, {"guid": guid})

            if s[1] > threshold / 2:  # and s[1] > s[0]:
                self.global_counter += 1
                accepted = True
            else:
                accepted = False

            update(
                self.cache,
                {"guid": record["guid"]},
                {"$set": {"accept": accepted, "score": s[1]}}
            )

        return self.post_process_batch(counter_data, batch_counter)


def collect_accepted(cache):
    accepted = []
    rejected = []

    cursor = list(cache.find({}))
    for record in tqdm(cursor):
        del record["_id"]
        if record["accept"]:
            accepted.append(record)
        else:
            rejected.append(record)

    return accepted, rejected


class FilterDataset(Dataset):
    def __init__(self, counter_data):
        self.counter_data = counter_data

    def __getitem__(self, index):
        return self.counter_data[index]

    def __len__(self):
        return len(self.counter_data)
