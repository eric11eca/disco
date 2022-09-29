import redis
import json
import string
import logging
import argparse
import pprint
import wandb
import numpy as np
import textdistance as tdist

from counterfactual_filter import write_jsonl

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

runner = wandb.init(
    project="gpt3-counterfactual-generation",
    entity="causal_scaffold",
    name="gpt_generation_logger"
)

cache_map = {
    "snli": {
        "e2c": 10,
        "c2e": 11,
        "e2n": 2,
        "c2n": 3
    },
    "wanli": {
        "e2c": 4,
        "c2e": 5,
        "e2n": 6,
        "c2n": 7
    }
}

forbidden_phrases = ['It is true', 'It is false',
                     'it is true', 'it is false',
                     '[blank]', 'Fill in the blank:']

negation_words = [
    "not", "no", "none",
    "doesn’t", "isn’t", "wasn’t",
    "shouldn’t", "wouldn’t",
    "couldn’t", "won’t", "can’t", "don’t"
]


def delete_multiple_element(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)


def strip_punctuation_and_casing(s):
    s = s.lower()
    s = s.translate(str.maketrans('', '', string.punctuation))
    return s


def simple_filtering(raw_data):
    try:
        gen_out = strip_punctuation_and_casing(raw_data["gen_out"])
    except:
        gen_out = strip_punctuation_and_casing(raw_data["span_to"])
    premise = strip_punctuation_and_casing(raw_data["premise"])
    if isinstance(raw_data['new_premise'], list):
        new_premise = strip_punctuation_and_casing(raw_data['new_premise'][0])
    else:
        new_premise = strip_punctuation_and_casing(raw_data['new_premise'])
    hypothesis = strip_punctuation_and_casing(raw_data["hypothesis"])

    if gen_out == hypothesis:
        return True, "copied_hypothesis"
    elif hypothesis in gen_out:
        return True, "coverd_hypothesis"
    elif gen_out in hypothesis:
        return True, "coverd_by_hypothesis"
    elif tdist.jaccard(gen_out.split(), hypothesis.split()) > 0.6:
        return True, "overlap_hypothesis"
    elif tdist.jaccard(gen_out.split(), premise.split()) > 0.35:
        return True, "overlap_premise"
    elif tdist.jaccard(new_premise.split(), hypothesis.split()) > 0.5:
        return True, "overlap_hypothesis_all"
    elif np.any([x in gen_out for x in forbidden_phrases]):
        return True, "forbidden_phrase"
    elif raw_data["new_label"] == "contradiction" and np.any([x in gen_out for x in negation_words]):
        return True, "negation_word"
    elif raw_data["new_label"] == "neutral" and "_" in gen_out:
        return True, "large_gap"
    return False, "none"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default="e2c",
                        help="type of counterfactual: e2c | c2e | e2n | c2n")
    parser.add_argument('--dataset', type=str, default="snli",
                        help="target dataset for counterfactual generation")

    args = parser.parse_args()

    r_cache = redis.Redis(host='localhost', port=6379, db=cache_map[args.dataset][args.type])

    logger.info(f"Number of total data: {len(r_cache.keys())}")

    accepted = []
    rejected = []

    discards = {
        'overlap_hypothesis': 0,    # gen_output largely overlaps the hypothesis
        'copied_hypothesis': 0,     # gen_output == hypothesis
        'coverd_hypothesis': 0,     # gen_output covers the hypothesis
        'overlap_premise': 0,
        'coverd_by_hypothesis': 0,  # gen_output is covered by the hypothesis
        'forbidden_phrase': 0,      # examples contain phrase from instructions
        'negation_word': 0,         # examples contain negation words for contradiction
        'large_gap': 0,             # examples contain large gap: ___________
        'overlap_hypothesis_all': 0
    }

    for guid in r_cache.keys():
        record = json.loads(r_cache.get(guid))
        if record["accept"]:
            blocked, reason = simple_filtering(record)
            if blocked:
                record["accept"] = False
                r_cache.set(guid, json.dumps(record))
                rejected.append(record)
                discards[reason] += 1
            else:
                accepted.append(record)
        else:
            rejected.append(record)
            #r_cache.delete(guid)

    logger.info(f"Number of accepted data: {len(accepted)}")
    logger.info(f"Number of rejected data: {len(rejected)}")

    pprint.pprint(discards)

    accepted_path = f"./data/filtered/{args.dataset}/{args.type}.jsonl"
    #write_jsonl(accepted, accepted_path)

    artifact = wandb.Artifact(
        f"{args.dataset}_{args.type}",
        type='dataset'
    )
    artifact.add_file(accepted_path)
    runner.log_artifact(artifact)
