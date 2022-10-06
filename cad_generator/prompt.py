import json
import random

from tqdm import tqdm
from .base import label_map_insert
from .utils import read_jsonl


def build_problems(args, cache):
    train_ds = read_jsonl(args.data_pth)
    train_subset = train_ds[args.start:args.end]

    problems = []
    guids = []

    for data in tqdm(train_subset):
        premise = data["premise"]
        hypothesis = data["hypothesis"]
        label = data["label"]

        if args.mode == "premise":
            all_spans = list(set(data["all_spans_p"]))
        else:
            all_spans = list(set(data["all_spans_h"]))
        
        # if len(all_spans) > 0:
        #     k = min(len(all_spans), 2)
        #     spans_to_mask = random.choices(all_spans, k=k)
        # else:
        #     spans_to_mask = []

        for i, span in enumerate(all_spans):
            guid = f"{data['guid']}_{i}"
            if args.mode == "premise":
                prompt = f"story: {premise.replace(span, '[blank]')}\n conclusion: {hypothesis}\n [blank] should be:"
            else:
                prompt = f"story: {premise}\n conclusion: {hypothesis.replace(span, '[blank]')}\n [blank] should be:"

            probelm = {
                "guid": guid,
                "premise": premise,
                "hypothesis": hypothesis,
                "label": label,
                "new_label": args.to_label,
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
    label_new = label_map_insert[args.to_label]

    problems = []
    guids = []

    for data in tqdm(train_subset):
        premise = data["premise"]
        hypothesis = data["hypothesis"]
        label = data["label"]

        if args.mode == "premise":
            all_spans = list(set(data["all_spans_p"]))
        else:
            all_spans = list(set(data["all_spans_h"]))

        if len(all_spans) > 0:
            k = min(len(all_spans), 2)
            spans_to_mask = random.choices(all_spans, k=k)
        else:
            spans_to_mask = []

        for i, span in enumerate(spans_to_mask):
            guid = f"{data['guid']}_{i}"
            if args.mode == "premise":
                prompt = f"{premise.replace(span, '[insert]')} Based on the above context, {label_new} {hypothesis}".split('[insert]')
            else:
                prompt = f"{premise}\n Based on the above context, {label_new} {hypothesis.replace(span, '[insert]')}".split('[insert]')
            
            if len(prompt) < 2:
                continue
            
            prefix = prompt[0]
            suffix = prompt[1]
            
            probelm = {
                "guid": guid,
                "premise": premise,
                "hypothesis": hypothesis,
                "label": label,
                "new_label": args.to_label,
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