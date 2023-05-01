import wandb
import openai
import hydra
import logging
import transformers

from hydra import initialize
from omegaconf import DictConfig

from distiller.db import get_database
from distiller.prompt.core import Task
from distiller.prompt.retrieval import get_task_class
from distiller.generator import Generator
from distiller.utils import write_jsonl
# from distiller.filter import (
#     NLIEnsembleFilter,
#     AutomaticHeuristicFilter,
#     PerplexityFilter,
#     collect_accepted,
#     FilterDataset,
# )

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger("distiller.runner")

transformers.logging.set_verbosity_error()


class DistillerRunner:
    def __init__(self, args, cache, task_container: Task):
        self.args = args
        self.generator = Generator(args, cache)
        self.generation_outputs = []
        initialize(config_path="./templates", version_base="1.3")

    def compose_loop(self):
        logger.info("Loading input data and composing prompts ...")
        self.task_container = get_task_class(
            self.args.task_name)
        querys = self.task_container.build_prompts()
        logger.info(f"Composed {len(self.task_container)} prompts.")
        return querys
    
    def generate_loop(self, querys):
        logger.info("Generating augmentation data ...")
        generation_outputs = self.generator.batch_generate(querys)
        logger.info(f"Generated {len(generation_outputs)} data points.")
        
        logger.info("Postprocessing generation outputs ...")
        self.task_container.postprocess_generation(
            self.cache, generation_outputs)
        logger.info("Postprocessing complete, updates persisted to cache.")
        return generation_outputs
    
    def filter_loop(self):
       raise NotImplementedError
    
    def write_outputs(self, generation_outputs):
        logger.info("Writing outputs ...")
        write_jsonl(
            generation_outputs,
            self.args.output_pth,
        )
        logger.info(f"Outputs written to {self.args.output_pth}.")
    
    def main_loop(self):
        querys = self.compose_loop()
        generation_outputs = self.generate_loop(querys)
        if self.args.do_filter:
            self.filter_loop()
        self.write_outputs(generation_outputs)


@hydra.main(config_path="config/secret/", config_name="keys")
def set_up_api(keys: DictConfig):
    openai.organization = keys.organization_token
    openai.api_key = keys.api_token1


@hydra.main(config_path="config", config_name="config")
def main(args: DictConfig):
    # wandb_run = wandb.init(
    #     project=args.project,
    #     entity=args.entity,
    #     name=args.name,
    # )

    cache = get_database(args.dataset, args.template_name)
    runner = DistillerRunner(args, cache)
    runner.main_loop()



if __name__ == "__main__":
    set_up_api()
    main()
