import random
import json

from tqdm import tqdm
from counterfactual_filter import read_jsonl, write_jsonl

from base import (
    label_map,
    label_map_insert
)

def build_problems(args, cache):
    train_ds = read_jsonl(args.data_pth)
    train_subset = train_ds[args.start:args.end]
    to_label = args.to_label
    label_new = label_map[to_label]

    problems = []
    guids = []

    for data in tqdm(train_subset):
        premise = data["premise"]
        hypothesis = data["hypothesis"]
        label = data["label"]
        label_curr = label_map[label]
        
        try:
            all_spans = list(set(data["new_span_p"]))
        except KeyError:
            all_spans = list(set(data["all_spans_p"]))

        if len(all_spans) > 0:
            k = min(len(all_spans), 2)
            spans_to_mask = random.choices(all_spans, k=k)
        else:
            spans_to_mask = []

        for i, span in enumerate(spans_to_mask):
            guid = f"{data['guid']}_{i}"
            prompt = f"story: {premise.replace(span, '[blank]')}\n conclusion: {hypothesis}\n [blank] should be:"

            probelm = {
                "guid": guid,
                "premise": premise,
                "hypothesis": hypothesis,
                "label": label,
                "new_label": to_label,
                "prompt": prompt,
                "span_prev": span,
                "gen_out": "",
                "accept": False
            }

            if cache.exists(guid) > 0:
                record = json.loads(cache.get(guid))
                if not record["accept"]:
                    problems.append({
                        "premise": premise,
                        "hypothesis": hypothesis
                    })
                    guids.append(guid)
            else:
                cache.set(guid, json.dumps(probelm))
                problems.append({
                    "premise": premise,
                    "hypothesis": hypothesis
                })
                guids.append(guid)

    return guids, problems

def build_problems_insertion(args, cache):
    train_ds = read_jsonl(args.data_pth)
    train_subset = train_ds[args.start:args.end]
    to_label = args.to_label
    label_new = label_map[to_label]

    problems = []
    guids = []

    for data in tqdm(train_subset):
        premise = data["premise"]
        hypothesis = data["hypothesis"]
        label = data["label"]
        label_curr = label_map[label]
        
        try:
            all_spans = list(set(data["new_span_p"]))
        except KeyError:
            all_spans = list(set(data["all_spans_p"]))

        if len(all_spans) > 0:
            k = min(len(all_spans), 2)
            spans_to_mask = random.choices(all_spans, k=k)
        else:
            spans_to_mask = []

        for i, span in enumerate(spans_to_mask):
            guid = f"{data['guid']}_{i}"
            prompt = f"{premise.replace(span, '[insert]')} Based on the above context, {label_map_insert[to_label]} {hypothesis}".split('[insert]')
            if len(prompt) < 2:
                continue
            prefix = prompt[0]
            suffix = prompt[1]
            probelm = {
                "guid": guid,
                "premise": premise,
                "hypothesis": hypothesis,
                "label": label,
                "new_label": to_label,
                "prefix": prefix,
                "suffix": suffix,
                "span_prev": span,
                "gen_out": "",
                "accept": False
            }

            if cache.exists(guid) > 0:
                record = json.loads(cache.get(guid))
                if not record["accept"]:
                    problems.append({
                        "premise": premise,
                        "hypothesis": hypothesis
                    })
                    guids.append(guid)
            else:
                cache.set(guid, json.dumps(probelm))
                problems.append({
                    "premise": premise,
                    "hypothesis": hypothesis
                })
                guids.append(guid)

    return guids, problems


def build_problems_anli(args, cache):
    train_ds = read_jsonl(args.data_pth)
    train_subset = train_ds[args.start:args.end]
    label_new = label_map[to_label]

    problems = []
    guids = []

    instruction = "Complete the conclusion with creative content so that it is true based on the story. Do not repeat the story. "

    for data in tqdm(train_subset):
        premise = data["premise"]
        hypothesis = data["hypothesis"]
        label = data["label"]
        
        prompt = f"story: {premise}\n conclusion: {hypothesis}\n output:"

        probelm = {
            "guid": guid,
            "premise": premise,
            "hypothesis": hypothesis,
            "label": label,
            "new_label": to_label,
            "prompt": prompt,
            "span_prev": span,
            "gen_out": "",
            "accept": False
        }

    return problems, instruction