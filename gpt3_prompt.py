import openai
import uuid
import random
import argparse
import logging
import redis
import json
import torch
import wandb
import numpy as np

from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset

from base import (
    label_map,
    label_map_insert,
    cache_map, 
    type_map,
    Example, 
    Prompt, 
    FilterDataset
)

from demonstration import (
    build_masked_nli_perturbation,
    demonstration_search
)

from token import api_token, organization_token
from prompt import build_problems, build_problems_insertion
from counterfactual_filter import read_jsonl, write_jsonl
from counterfactual_filter import NLICounterfactualFilter

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

openai.api_key = api_token
openai.organization = organization_token

runner = wandb.init(
    project="gpt3-counterfactual-generation",
    entity="causal_scaffold",
    name="gpt_generation_logger"
)

def prompt_perturbation_insertion(args, to_label, cache, encoder=None):
    generation_outputs = []

    logger.info("Build prompt: enumerate problems")
    guids, problems = build_problems_insertion(args, cache)
    
    logger.info(f"Prompting {len(problems)} problems ...")
    for i, guid in enumerate(guids):
        record = json.loads(cache.get(guid))
        if not record["accept"]:
            if i > 0 and i % 100 == 0:
                logger.info(
                    f"=============== Prompting progress: {len(generation_outputs)} problems ===============")

            if "new_premise" in record:
                del record["new_premise"]

            response = openai.Completion.create(
                model="text-davinci-002",
                prompt=record['prefix'],
                suffix=record['suffix'],
                temperature=0.8,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0.8,
                presence_penalty=0.5,
                stop=["stop", "\n", "."]
            )

            output = response['choices'][0]['text'].replace("\n", "").strip()
            record["gen_out"] = output
            cache.set(guid, json.dumps(record))
            generation_outputs.append(record)

    logger.info(f"Receiving {len(generation_outputs)} generation outputs")
    return generation_outputs


def prompt_perturbation(args, to_label, cache, encoder=None):
    logger.info("Build prompt: sample demonstrations")
    instruction, perturbations = build_masked_nli_perturbation(
        args.demo_pth,
        to_label
    )

    logger.info(f"Prompting Instruction: {instruction}")
    random.shuffle(perturbations)

    logger.info("Build prompt: enumerate problems")
    guids, problems = build_problems(args, cache)
    gpt_prompt = Prompt()

    if args.prompt_search:
        examples_selected = demonstration_search(
            args, perturbations, problems, encoder)

        assert len(examples_selected) == len(problems)

    generation_outputs = []
    logger.info(f"Prompting {len(problems)} problems ...")

    for i, guid in enumerate(guids):
        record = json.loads(cache.get(guid))
        if not record["accept"]:
            if i > 0 and i % 100 == 0:
                logger.info(
                    f"=============== Prompting progress: {len(generation_outputs)} problems ===============")

            if args.prompt_search:
                examples = [perturbations[j] for j in examples_selected[i]]
                examples.reverse()
            else:
                random.shuffle(perturbations)
                examples = perturbations[:args.num_neighbors]

            gpt_prompt.delete_all_examples()
            for example in examples:
                demonstration = Example(example["prompt"], example["output"])
                gpt_prompt.add_example(demonstration)

            if "new_premise" in record:
                del record["new_premise"]

            prompt = gpt_prompt.craft_query(
                record['prompt'],
                instruction=instruction)
            
            write_jsonl([prompt], "./prompt_example.jsonl")
            response = openai.Completion.create(
                engine="text-davinci-002",
                prompt=prompt,
                top_p=1.0,
                temperature=0.8,
                max_tokens=256,
                frequency_penalty=0.8,
                presence_penalty=0.5
            )

            output = response['choices'][0]['text'].replace("\n", "").strip()
            record["gen_out"] = output
            cache.set(guid, json.dumps(record))
            generation_outputs.append(record)

    logger.info(f"Receiving {len(generation_outputs)} generation outputs")

    return generation_outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default="e2c",
                        help="type of counterfactual: e2c | c2e | e2n | c2n")
    parser.add_argument('--mode', type=str, default="completion", 
                        help="type of gpt-3 generation: completion | insertion")
    parser.add_argument('--out_dir', type=str, default="./data/output",
                        help="output directory")
    parser.add_argument('--filter_dir', type=str, default="./data/filtered",
                        help="filtered output directory")
    parser.add_argument('--dataset', type=str, default="snli",
                        help="base dataset for conterfactual generation")
    parser.add_argument('--demo_dir', type=str, default="./data/examples",
                        help="demonstration directory")
    parser.add_argument('--start', type=int, default=0,
                        help="start position of the dataset")
    parser.add_argument('--end', type=int, default=100,
                        help="end position of the dataset")
    parser.add_argument('--version', type=str, default="0",
                        help="data record version")
    parser.add_argument('--num_neighbors', type=int, default=8,
                        help="number of nearest neighbors")
    parser.add_argument('--embed_type', type=str, default="CLS",
                        help="type of embedding method")
    parser.add_argument('--metric', type=str, default="euclidean",
                        help="distance metric used for nearest neighbor")
    parser.add_argument('--encoder_name', type=str, default="roberta-large-nli-mean-tokens",
                        help="name of the encoder for sentence embedding")
    parser.add_argument('--prompt_search', action='store_true', default=False,
                        help="enable dynamically selecting demonstrations")

    args = parser.parse_args()
    label_pair = type_map[args.type]
    label = label_pair[0]
    to_label = label_pair[1]
    
    args.lable = label
    args.to_label = to_label
    args.generate = True
    
    args.data_pth = f"./data/{args.dataset}/{label}_esnli.jsonl"
    args.demo_pth = f"{args.demo_dir}/{args.dataset}/{label}_{to_label}.jsonl"
    args.out_pth = f"{args.out_dir}/{args.dataset}_{label}_{to_label}_{args.start}_{args.end}.jsonl"

    cache = redis.Redis(
        host='localhost', 
        port=6379, 
        db=cache_map[args.dataset][args.type]
    )

    nli_filter = NLICounterfactualFilter(cache)

    if args.prompt_search:
        encoder = SentenceTransformer(args.encoder_name)
    else:
        encoder = None

    logger.info("Collecting counterfactuals from GPT-3")

    prompt_function = {
        "insertion": prompt_perturbation_insertion,
        "completion": prompt_perturbation
    }

    if args.generate:
        counter_data = prompt_function[args.mode](args, to_label, cache, encoder)
    else:
        counter_data = []
        for key in cache.keys():
            record = json.loads(cache[key])
            if "new_premise" in record:
                del record["new_premise"]
            if not record["accept"]:
                counter_data.append(record)

    counter_ds = FilterDataset(counter_data)

    logger.info("Filtering generated counterfactual data")
    loader = DataLoader(counter_ds, batch_size=16, shuffle=False)

    logger.info("Filter running ...")
    for i, batch in enumerate(tqdm(loader)):
        filtered_batch = nli_filter.filter(batch, threshold=0.3)
        if i > 0:
            write_jsonl(filtered_batch, args.out_pth, mode="a")
        else:
            write_jsonl(filtered_batch, args.out_pth, mode="w")

    artifact = wandb.Artifact(
        f"{args.dataset}_{label}_{to_label}_{args.start}_{args.end}",
        type='dataset'
    )
    artifact.add_file(args.out_pth)
    runner.log_artifact(artifact)

    logger.info("Counterfactual collection complete.")
