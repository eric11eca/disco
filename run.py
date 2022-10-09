import json
import wandb
import openai
import redis
import argparse
import logging

from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader

from cad_generator.base import (
    cache_map,
    type_map,
    FilterDataset
)

from cad_generator.api import api_token, organization_token

from cad_generator.generator import (
    prompt_perturbation,
    prompt_perturbation_insertion
)

from cad_generator.utils import write_jsonl
from cad_generator.filter import (
    NLIEnsembleFilter,
    AutomaticHeuristicFilter,
    PerplexityFilter,
    collect_accepted
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

openai.organization = organization_token

runner = wandb.init(
    project="gpt3-counterfactual-generation",
    entity="causal_scaffold",
    name="gpt_generation_logger"
)


def get_cache(args):
    if args.gen_type == "completion":
        cache_idx = 0
    else:
        cache_idx = 1

    logger.info(
        f"Cache Index: {cache_map[args.dataset][args.type][cache_idx]}")
    cache = redis.Redis(
        host='localhost',
        port=6379,
        db=cache_map[args.dataset][args.type][cache_idx]
    )

    return cache


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default="e2c",
                        help="type of counterfactual: e2c | c2e | e2n | c2n")
    parser.add_argument('--gen_type', type=str, default="completion",
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
    parser.add_argument('--api_idx', type=int, default=0,
                        help="data record version")
    parser.add_argument('--local_rank', type=str, default="0",
                        help="locak rank of cuda devices")
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

    openai.api_key = api_token[args.api_idx]

    args.lable = label
    args.to_label = to_label
    args.generate = True

    if args.dataset == "snli":
        args.data_pth = f"./data/input/{args.dataset}/{label}_spans.jsonl"
    else:
        args.data_pth = f"./data/input/{args.dataset}/{label}.jsonl"
    args.demo_pth = f"{args.demo_dir}/{args.dataset}/{label}_{to_label}.jsonl"
    args.out_pth = f"{args.out_dir}/{args.dataset}_{label}_{to_label}_{args.start}_{args.end}.jsonl"

    if args.dataset == "anli":
        args.mode = "hypothesis"
    else:
        args.mode = "premise"

    cache = get_cache(args)

    if args.prompt_search:
        encoder = SentenceTransformer(args.encoder_name)
    else:
        encoder = None

    logger.info("Collecting counterfactuals from GPT-3")

    prompt_function = {
        "insertion": prompt_perturbation_insertion,
        "completion": prompt_perturbation
    }

    alien_counter_data = []
    if args.generate:
        counter_data = prompt_function[args.gen_type](args, cache, encoder)
    else:
        counter_data = []
        for key in cache.keys():
            record = json.loads(cache[key])
            record[f"new_{args.mode}"] = record[args.mode].replace(
                record["span_prev"], record["gen_out"]
            )
            if args.gen_type == "insertion" and "prompt" in record:
                alien_counter_data.append(record)
            else:
                counter_data.append(record)
    logger.info(f"Filtering {len(counter_data)} counterfactuals")

    heuristic_filter = AutomaticHeuristicFilter()
    ppl_filter = PerplexityFilter()
    nli_filter = NLIEnsembleFilter(cache, args.mode, args.local_rank)
    logger.info(
        f"Initialized {len(nli_filter.hf_model_names)} filtering models")

    logger.info(f"Filtering thorugh Heuristic Filter ...")
    h_filtered = heuristic_filter.run(counter_data, cache, args.mode)

    counter_ds = FilterDataset(h_filtered)
    loader = DataLoader(counter_ds, batch_size=32, shuffle=False)

    logger.info("Filtering through NLI Ensemble Filter ...")
    filtered_batch = []
    for i, batch in enumerate(tqdm(loader)):
        filtered_batch += nli_filter.filter(batch, threshold=0.3)

    write_jsonl(filtered_batch, args.out_pth, mode="w")

    accepted_batch = [
        record for record in filtered_batch if record["accept"]]
    logger.info(f"Accepted {len(accepted_batch)} counterfactuals")

    logger.info("Filtering through Perplexity Filter ...")
    p_filtered = ppl_filter.run(
        accepted_batch, cache, args.mode, threshold=150)

    artifact = wandb.Artifact(
        f"{args.dataset}_{label}_{to_label}_{args.start}_{args.end}",
        type='dataset'
    )
    artifact.add_file(args.out_pth)
    runner.log_artifact(artifact)

    accepted, rejected = collect_accepted(cache)
    logger.info(f"Number of total accepted data: {len(accepted)}")
    logger.info(f"Number of total rejected data: {len(rejected)}")

    accepted_path = f"./data/filtered/{args.dataset}/{args.type}_{args.gen_type}.jsonl"
    write_jsonl(accepted, accepted_path)
    artifact = wandb.Artifact(
        f"{args.dataset}_{args.type}_{args.gen_type}",
        type='dataset'
    )
    artifact.add_file(accepted_path)
    runner.log_artifact(artifact)

    logger.info("Counterfactual collection complete.")
