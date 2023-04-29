import wandb
import openai
import hydra
import logging
import transformers

from tqdm import tqdm
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from cad_generator.db import get_database
from cad_generator.utils import write_jsonl
from cad_generator.prompt.core import Task
from cad_generator.generator import Generator
from cad_generator.filter import (
    NLIEnsembleFilter,
    AutomaticHeuristicFilter,
    PerplexityFilter,
    collect_accepted,
    FilterDataset,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

transformers.logging.set_verbosity_error()


class NeuroCrowdRunner:
    def __init__(self, args, cache, task_container: Task):
        self.args = args
        self.task_container = task_container
        self.generator = Generator(args, cache)
        self.generation_outputs = []

    def run_task_setup(self):
        task = self.task_container.get_task()
        task.run_setup()


@hydra.main(config_path="config/secret/", config_name="keys")
def set_up_api(keys: DictConfig):
    openai.organization = keys.organization_token
    openai.api_key = keys.api_token1


@hydra.main(config_path="config", config_name="config")
def main(args: DictConfig):
    runner = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.name,
    )

    cache = get_database(args.dataset, args.type)


if __name__ == "__main__":
    set_up_api()
    main()

    logger.info("Counterfactual collection complete.")
