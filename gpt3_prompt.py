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
from torch.utils.data import DataLoader, Dataset

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise
from sentence_transformers import SentenceTransformer

from counterfactual_filter import read_jsonl, write_jsonl
from counterfactual_filter import NLICounterfactualFilter

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

label_map = {
    "entailment": "true",
    "contradiction": "false",
    "neutral": "irrelevant"
}

label_counter = {
    "entailment": ["contradiction", "neutral"],
    "contradiction": ["entailment", "neutral"],
    "neutral": ["entailment", "contradiction"],
}

cache_map = {
    "entailment->contradiction": 0,
    "contradiction->entailment": 1,
    "entailment->neutral": 2,
    "contradiction->neutral": 3
}

openai.api_key = "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"
openai.organization = "org-w7nKit9OwsqNNO3i9GmXe5uk"

runner = wandb.init(
    project="gpt3-counterfactual-generation",
    entity="causal_scaffold",
    name="gpt_generation_logger"
)


class Example:
    """Stores an input, output pair and formats it to prime the model."""

    def __init__(self, inp, out):
        self.input = inp
        self.output = out
        self.id = uuid.uuid4().hex

    def get_input(self):
        """Returns the input of the example."""
        return self.input

    def get_output(self):
        """Returns the intended output of the example."""
        return self.output

    def get_id(self):
        """Returns the unique ID of the example."""
        return self.id

    def as_dict(self):
        return {
            "input": self.get_input(),
            "output": self.get_output(),
            "id": self.get_id(),
        }


class Prompt:
    """The main class for a user to create a prompt for GPT3"""

    def __init__(self) -> None:
        self.examples = []

    def add_example(self, ex):
        """
        Adds an example to the object.
        Example must be an instance of the Example class.
        """
        assert isinstance(ex, Example), "Please create an Example object."
        self.examples.append(ex)

    def delete_example(self, id):
        """Delete example with the specific id."""
        if id in self.examples:
            del self.examples[id]

    def delete_all_examples(self):
        self.examples = []

    def get_example(self, id):
        """Get a single example."""
        return self.examples.get(id, None)

    def get_all_examples(self):
        """Returns all examples as a list of dicts."""
        return {k: v.as_dict() for k, v in self.examples.items()}

    def craft_query(self, input, instruction=""):
        """Creates the query for the API request."""
        prompt = f"{instruction} \n\n"
        for example in self.examples:
            prompt += f"{example.get_input()}\n{example.get_output()}\n\n"
        prompt += input

        return prompt


class FilterDataset(Dataset):
    def __init__(self, counter_data):
        self.counter_data = counter_data

    def __getitem__(self, index):
        return self.counter_data[index]

    def __len__(self):
        return len(self.counter_data)


def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def decode(args, data_loader, tok=None, model=None):
    embeddings = []

    if args.encoder_name == 'roberta-base' or args.encoder_name == 'roberta-large':
        print("Using non Sentence Transformer models")
        for corpus_tmp in tqdm(data_loader):
            encoding = tok.batch_encode_plus(
                corpus_tmp, padding=True, truncation=True)
            sentence_batch, attn_mask = encoding["input_ids"], encoding["attention_mask"]
            sentence_batch, attn_mask = torch.LongTensor(
                sentence_batch), torch.LongTensor(attn_mask)

            with torch.no_grad():
                embedding_output_batch = model(sentence_batch, attn_mask)
                if args.embed_type == 'mean':
                    sentence_embeddings = mean_pooling(
                        embedding_output_batch, attn_mask)
                elif args.embed_type == 'CLS':
                    sentence_embeddings = embedding_output_batch[0][:, 0, :]
            embeddings.append(sentence_embeddings.detach().cpu().numpy())
            del sentence_batch, attn_mask, embedding_output_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        print("Using Sentence Transformer models")
        for corpus_tmp in tqdm(data_loader):
            sentence_embeddings = model.encode(corpus_tmp)
            embeddings.append(sentence_embeddings)

    return np.concatenate(embeddings, axis=0)


def build_problems(args, cache):
    train_ds = read_jsonl(args.data_pth)
    train_subset = train_ds[args.start:args.end]
    problems = []
    guids = []
    label_new = label_map[to_label]
    #is_neutral = args.to_label == "neutral"
    is_neutral = False

    instruction = f"Complete the story with creative content so that the conclusion is {label_new}. Do not repeat the conclusion."

    for data in tqdm(train_subset):
        premise = data["premise"]
        hypothesis = data["hypothesis"]
        label = data["label"]
        try:
            all_spans = list(set(data["new_span_p"]))
            if len(all_spans) > 0:
                spans_to_mask = random.choices(all_spans, k=2)
            else:
                spans_to_mask = []
        except KeyError:
            print(data)
            continue

        for i, span in enumerate(spans_to_mask):
            guid = f"{data['guid']}_{i}"

            if is_neutral:
                masked = premise.replace(span, '[blank]').split('[blank]')
                if len(masked) == 1:
                    continue
                prompt = masked[0]
                suffix = f"{masked[1].strip()}"

                probelm = {
                    "guid": guid,
                    "premise": premise,
                    "hypothesis": hypothesis,
                    "label": label,
                    "new_label": to_label,
                    "prompt": prompt,
                    "suffix": suffix,
                    "span_prev": span,
                    "gen_out": "",
                    "accept": False
                }

            else:
                prompt = f"story: {premise.replace(span, '[blank]')}\n conclusion: {hypothesis}\n output:"
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

            if cache.exists(guid) == 0:
                cache.set(guid, json.dumps(probelm))

            problems.append({
                "premise": premise,
                "hypothesis": hypothesis
            })

            guids.append(guid)

    return guids, problems


def build_masked_nli_perturbation(demo_pth, to_label, prompt_type=1):
    examples = []
    counter_data = read_jsonl(demo_pth)
    label_new = label_map[to_label]

    instruction = f"Complete the story with creative content so that the conclusion is {label_new}. Do not repeat the conclusion."

    for data in counter_data:
        premise = data["premise"]
        hypothesis = data["hypothesis"]

        span_prev = data["span_changed"]
        span_new = data["span_to"]
        prompt = f"story: {premise.replace(span_prev, '[blank]')}\n conclusion: {hypothesis}\n output:"

        examples.append(
            {
                "premise": premise,
                "hypothesis": hypothesis,
                "prompt": prompt,
                "output": span_new
            })

    return instruction, examples


def demonstration_search(args, examples, problems, encoder):
    demos = [x["premise"] + x["hypothesis"] for x in examples]
    inputs = [x["premise"] + x["hypothesis"] for x in problems]

    demo_loader = DataLoader(demos, batch_size=16, shuffle=False)
    input_loader = DataLoader(inputs, batch_size=16, shuffle=False)

    emb_train = decode(args, demo_loader, model=encoder)
    emb_dev = decode(args, input_loader, model=encoder)

    if args.metric == "euclidean":
        nbrs = NearestNeighbors(
            n_neighbors=args.num_neighbors,
            algorithm='ball_tree',
            n_jobs=-1
        ).fit(emb_train)
        _, indices = nbrs.kneighbors(emb_dev)
    elif args.metric == "cosine":
        dist_matrix = pairwise.cosine_similarity(X=emb_dev, Y=emb_train)
        if reversed:
            _, indices = torch.topk(-torch.from_numpy(dist_matrix),
                                    k=args.num_neighbors, dim=-1)
        else:
            _, indices = torch.topk(torch.from_numpy(
                dist_matrix), k=args.num_neighbors, dim=-1)
        indices = indices.numpy()

    return indices


def prompt_perturbation(args, cache, encoder):
    logger.info("Build prompt: sample demonstrations")
    instruction, perturbations = build_masked_nli_perturbation(
        args.demo_pth,
        args.to_label
    )
    print(instruction)

    random.shuffle(perturbations)

    logger.info("Build prompt: enumerate problems")
    guids, problems = build_problems(args, cache)
    gpt_prompt = Prompt()

    if args.prompt_search:
        examples_selected = demonstration_search(
            args, perturbations, problems, encoder)

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

            response = openai.Completion.create(
                engine="text-davinci-002",
                prompt=prompt,
                # n=1,
                # best_of=2,
                top_p = 1.0,
                temperature=0.8,
                max_tokens=256,
                frequency_penalty=0.7,
                presence_penalty=0.8
            )

            output = response['choices'][0]['text'].replace("\n", "").strip()
            record["gen_out"] = output
            cache.set(guid, json.dumps(record))
            generation_outputs.append(record)

    logger.info(f"Receiving {len(generation_outputs)} generation outputs")

    return generation_outputs


def prompt_perturbation_neutral(args, cache, encoder):
    logger.info("Build prompt: enumerate problems")
    guids, problems = build_problems(args, cache)

    generation_outputs = []
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
                engine="text-davinci-002",
                prompt=record['prompt'],
                suffix=record['suffix'],
                n=1,
                stop=["stop", "\n", "\n\n"],
                temperature=0.75,
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
    parser.add_argument('--label', type=str, default="entailment",
                        help="the current label: entailment | contradiction | neutral")
    parser.add_argument('--to_label', type=str, default="contradiction",
                        help="the desired label: entailment | contradiction | neutral")
    parser.add_argument('--mode', type=str, default="premise",
                        help="perturbation mode: premise | hypothesis")
    parser.add_argument('--prompt_type', type=int, default=0,
                        help="choose prompt format: 0 | 1")
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

    label = args.label
    to_label = args.to_label
    mode = args.mode

    args.data_pth = f"./data/{args.dataset}/{label}.jsonl"
    args.demo_pth = f"{args.demo_dir}/{label}_{to_label}.jsonl"
    args.out_pth = f"{args.out_dir}/{args.dataset}_{label}_{to_label}_{args.start}_{args.end}.jsonl"

    cache = redis.Redis(host='localhost', port=6379,
                        db=cache_map[f"{label}->{to_label}"])
    nli_filter = NLICounterfactualFilter(cache)

    if args.prompt_search:
        encoder = SentenceTransformer(args.encoder_name)
    else:
        encoder = None

    logger.info("Collecting counterfactuals from GPT-3")

    counter_data = prompt_perturbation(args, cache, encoder)

    #counter_data = read_jsonl("data/output/contradiction_entailment_premise_1.jsonl")
    #for cd in counter_data:
    #    cache.set(cd["guid"], json.dumps(cd))
    """counter_data = []
    for key in cache.keys():
        record = json.loads(cache[key])
        if "new_premise" in record:
            del record["new_premise"]
        if not record["accept"]:
            counter_data.append(record)"""
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
