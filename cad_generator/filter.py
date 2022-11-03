import json
import string
import torch
import numpy as np
import textdistance as tdist

from tqdm import tqdm
from evaluate import load
from pprint import pprint
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from .base import (
    label2id,
    forbidden_phrases,
    negation_words,
    FilterDataset
)


class AutomaticHeuristicFilter:
    def _strip_punctuation_and_casing(self, s):
        s = s.lower()
        s = s.translate(str.maketrans('', '', string.punctuation))
        return s

    def _normalize(self, text):
        return self._strip_punctuation_and_casing(text)

    def heuristic_filtering(self, data, mode):
        premise = self._normalize(data["premise"])
        hypothesis = self._normalize(data["hypothesis"])
        gen_out = self._normalize(data["gen_out"])

        if isinstance(data[f"new_{mode}"], list):
            new_input = self._normalize(data[f"new_{mode}"][0])
        else:
            new_input = self._normalize(data[f"new_{mode}"])

        if gen_out == hypothesis:
            return True, "copied_hypothesis"
        elif hypothesis in gen_out:
            return True, "coverd_hypothesis"
        elif gen_out in hypothesis:
            return True, "coverd_by_hypothesis"
        elif gen_out in premise:
            return True, "coverd_by_premise"
        elif tdist.jaccard(gen_out.split(), hypothesis.split()) > 0.6:
            return True, "overlap_hypothesis"
        elif tdist.jaccard(gen_out.split(), premise.split()) > 0.35:
            return True, "overlap_premise"
        elif np.any([x in data["gen_out"] for x in forbidden_phrases]):
            return True, "forbidden_phrase"
        elif data["new_label"] == "contradiction" and np.any([x in gen_out for x in negation_words]):
            return True, "negation_word"
        elif "_" in data["gen_out"]:
            return True, "large_gap"
        elif "________" in data["gen_out"]:
            return True, "large_gap"
        elif mode == "premise" and tdist.jaccard(new_input.split(), hypothesis.split()) > 0.5:
            return True, "overlap_hypothesis_all"
        elif mode == "hypothesis" and tdist.jaccard(new_input.split(), hypothesis.split()) > 0.8:
            return True, "overlap_hypothesis_all"
        return False, "none"

    def run(self, outputs, cache, mode):
        discards = {
            'overlap_hypothesis': 0,    # gen_output largely overlaps the hypothesis
            'copied_hypothesis': 0,     # gen_output == hypothesis
            'coverd_hypothesis': 0,     # gen_output covers the hypothesis
            'coverd_by_premise': 0,
            'overlap_premise': 0,
            'coverd_by_hypothesis': 0,  # gen_output is covered by the hypothesis
            'forbidden_phrase': 0,      # examples contain phrase from instructions
            'negation_word': 0,         # examples contain negation words for contradiction
            'large_gap': 0,             # examples contain large gap: ___________
            'overlap_hypothesis_all': 0
        }

        accepted = []
        for record in outputs:
            blocked, reason = self.heuristic_filtering(record, mode)
            if blocked:
                record["accept"] = False
                cache.set(record['guid'], json.dumps(record))
                discards[reason] += 1
            else:
                accepted.append(record)
        pprint(discards)
        return accepted


class PerplexityFilter:

    def __init__(self):
        self.perplexity = load(
            "perplexity", module_type="metric")

    def run(self, data, cache, mode, threshold=50):
        accepted = []
        counter_ds = FilterDataset(data)
        loader = DataLoader(counter_ds, batch_size=16, shuffle=False)

        accepted_id = []
        rejected_id = []
        for i, batch in enumerate(tqdm(loader, position=0, leave=True)):
            ppls = self.perplexity.compute(
                predictions=batch[f"new_{mode}"],
                model_id='gpt2'
            )['perplexities']

            accepted_ppl = np.where(np.array(ppls) < threshold)[0]
            rejected_ppl = np.where(np.array(ppls) >= threshold)[0]

            accepted_id += [batch['guid'][idx] for idx in accepted_ppl]
            rejected_id += [batch['guid'][idx] for idx in rejected_ppl]

            # for idx in accepted_ppl:
            #     record = {
            #         "guid": batch['guid'][idx],
            #         "premise": batch['premise'][idx],
            #         "hypothesis": batch['hypothesis'][idx],
            #         "label": batch['label'][idx],
            #         "new_label": batch['new_label'][idx],
            #         "prefix": batch['prefix'][idx],
            #         "suffix": batch['suffix'][idx],
            #         "span_prev": batch['span_prev'][idx],
            #         "gen_out": batch['gen_out'][idx],
            #         "new_premise": batch['new_premise'][idx],
            #         "accept": True
            #     }
            #     accepted.append(record)

        for guid in accepted_id:
            record = json.loads(cache.get(guid))
            accepted.append(record)
            cache.set(guid, json.dumps(record))

        for guid in rejected_id:
            record = json.loads(cache.get(guid))
            record["accept"] = False
            cache.set(guid, json.dumps(record))

        return accepted


class NLIEnsembleFilter:

    def __init__(self, cache, mode, local_rank) -> None:
        self.hf_model_names = [
            # "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
            # "ynie/xlnet-large-cased-snli_mnli_fever_anli_R1_R2_R3-nli",
            # "alisawuffles/roberta-large-wanli"
            "Joelzhang/deberta-v3-large-snli_mnli_fever_anli_R1_R2_R3-nli"
        ]

        self.device = f"cuda:{local_rank}"
        self.tokenizers = {}
        self.models = {}
        self.cache = cache
        self.mode = mode
        self.global_counter = 0

        for name in self.hf_model_names:
            tokenizer = AutoTokenizer.from_pretrained(name)
            model = AutoModelForSequenceClassification.from_pretrained(name)
            model = model.to(self.device)
            model.eval()
            self.tokenizers[name] = tokenizer
            self.models[name] = model

    def predict(self, batch, batch_counter, model_name):
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]

        input_seq_pair = tokenizer.batch_encode_plus(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            # max_length=256,
            return_token_type_ids=False
        )
        input_seq_pair = input_seq_pair.to(self.device)

        counter_seq_pair = tokenizer.batch_encode_plus(
            batch_counter,
            return_tensors="pt",
            truncation=True,
            padding=True,
            # max_length=256,
            return_token_type_ids=False
        )
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
        label = [label2id[x] for x in counter_data["label"]]
        to_label = [label2id[x] for x in counter_data["new_label"]]

        return batch, batch_counter, label, to_label

    def post_process_batch(self, counter_data, batch_counter):
        filtered_data = []
        for i in range(len(batch_counter)):
            guid = counter_data["guid"][i]
            record = json.loads(self.cache.get(guid))
            if self.mode == "premise":
                record["new_premise"] = batch_counter[i][0]
            else:
                record["new_hypothesis"] = batch_counter[i][0]

            # record["accept"] = counter_data["accept"][i].item()
            self.cache.set(guid, json.dumps(record))

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
            record = json.loads(self.cache.get(guid))
            record["score"] = s[1]
            if s[1] > threshold / 2:  # and s[1] > s[0]:
                self.global_counter += 1
                record["accept"] = True
            else:
                record["accept"] = False
            self.cache.set(guid, json.dumps(record))

        return self.post_process_batch(counter_data, batch_counter)


def collect_accepted(cache):
    accepted = []
    rejected = []

    for guid in cache.keys():
        record = json.loads(cache.get(guid))
        if record["accept"]:
            accepted.append(record)
        else:
            rejected.append(record)

    return accepted, rejected
