import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise

from base import (
    label_map,
    label_map_insert
)

from counterfactual_filter import read_jsonl, write_jsonl

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

def build_masked_nli_perturbation(demo_pth, to_label):
    examples = []
    counter_data = read_jsonl(demo_pth)
    label_new = label_map[to_label]

    instruction = f"Complete the story with creative content so that the conclusion is {label_new}. Do not repeat the conclusion."

    for data in counter_data:
        premise = data["premise"]
        hypothesis = data["hypothesis"]
        label_curr = label_map[data["label"]]

        span_prev = data["span_changed"]
        span_new = data["span_to"]
        prompt = f"story: {premise.replace(span_prev, '[blank]')}\n conclusion: {hypothesis}\n [blank] should be:"
        # context = f"story: {premise}\n conclusion: {hypothesis}\n"
        # inference = f"The conclusion is {label_curr} based on the story." 
        # prompt = f"{context} {inference} To make it {label_new} I need to change {span_prev} to:"

        examples.append(
            {
                "premise": premise,
                "hypothesis": hypothesis,
                "prompt": prompt,
                "output": span_new
            })

    return instruction, examples