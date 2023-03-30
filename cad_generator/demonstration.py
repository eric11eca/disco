import torch
import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer

from tqdm import tqdm


class NearestNeighbotSampler:
    """Sample Examples for In-context Learning based on Nearest Neighbors"""

    def __init__(self, args):
        if args.prompt_search:
            self.encoder = SentenceTransformer(args.encoder_name)
        else:
            self.encoder = None

    def mean_pooling(self, model_output, attention_mask):
        """Perform mean pooling on the model output to get one fixed sized sentence vector

        :param model_output: last layer hidden-state of the first token of the sequence
        :param attention_mask: attention mask of the inputs
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def decode(self, data_loader, tok=None, model=None):
        """Decode the data_loader into embeddings"""
        embeddings = []
        if self.args.encoder_name == 'roberta-base' or self.args.encoder_name == 'roberta-large':
            for corpus_tmp in tqdm(data_loader):
                encoding = tok.batch_encode_plus(
                    corpus_tmp, padding=True, truncation=True)
                sentence_batch, attn_mask = encoding["input_ids"], encoding["attention_mask"]
                sentence_batch, attn_mask = torch.LongTensor(
                    sentence_batch), torch.LongTensor(attn_mask)

                with torch.no_grad():
                    embedding_output_batch = model(sentence_batch, attn_mask)
                    if self.args.embed_type == 'mean':
                        sentence_embeddings = self.mean_pooling(
                            embedding_output_batch, attn_mask)
                    elif self.args.embed_type == 'CLS':
                        sentence_embeddings = embedding_output_batch[0][:, 0, :]
                embeddings.append(sentence_embeddings.detach().cpu().numpy())
                del sentence_batch, attn_mask, embedding_output_batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            for corpus_tmp in tqdm(data_loader):
                sentence_embeddings = model.encode(corpus_tmp)
                embeddings.append(sentence_embeddings)
        return np.concatenate(embeddings, axis=0)

    def demonstration_search(self, examples, problems, encoder):
        """Search for nearest neighbors in the demonstration set"""
        demos = [x["premise"] + x["hypothesis"] for x in examples]
        inputs = [x["premise"] + x["hypothesis"] for x in problems]

        demo_loader = DataLoader(demos, batch_size=16, shuffle=False)
        input_loader = DataLoader(inputs, batch_size=16, shuffle=False)

        emb_train = self.decode(demo_loader, model=encoder)
        emb_dev = self.decode(input_loader, model=encoder)

        if self.args.metric == "euclidean":
            nbrs = NearestNeighbors(
                n_neighbors=self.args.num_neighbors,
                algorithm='ball_tree',
                n_jobs=-1
            ).fit(emb_train)
            _, indices = nbrs.kneighbors(emb_dev)
        elif self.args.metric == "cosine":
            dist_matrix = pairwise.cosine_similarity(X=emb_dev, Y=emb_train)
            if reversed:
                _, indices = torch.topk(-torch.from_numpy(dist_matrix),
                                        k=self.args.num_neighbors, dim=-1)
            else:
                _, indices = torch.topk(torch.from_numpy(
                    dist_matrix), k=self.args.num_neighbors, dim=-1)
            indices = indices.numpy()

        return indices
