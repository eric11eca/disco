import os
import time
import wandb
import openai
import hydra
import logging
import transformers

from hydra import initialize
from omegaconf import DictConfig, open_dict

from distiller.db import get_database
from distiller.prompt.core import Task
from distiller.prompt.retrieval import get_task_class
from distiller.generator import Generator
from distiller.utils import write_json
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
    def __init__(self, args, input_cache, output_cache):
        self.args = args
        self.generator = Generator(args)
        self.input_cache = input_cache
        self.output_cache = output_cache
        self.generation_outputs = []
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        initialize(config_path="./templates", version_base="1.3.2")

    def compose_loop(self):
        logger.info("Loading input data and composing prompts ...")
        self.task_container = get_task_class(
            self.args.task_name)
        querys = self.task_container.build_prompts(self.args, self.input_cache)
        logger.info(f"Composed {len(querys)} prompts.")
        return querys
    
    def generate_loop(self, querys):
        logger.info("Generating augmentation data ...")
        generation_outputs = self.generator.batch_generate(querys)
        logger.info(f"Generated {len(generation_outputs)} data points.")
        
        logger.info("Postprocessing generation outputs ...")
        generation_outputs = self.task_container.postprocess_generation(
            self.output_cache, generation_outputs)
        logger.info("Postprocessing complete, updates persisted to cache.")
        return generation_outputs
    
    def filter_loop(self):
       raise NotImplementedError
    
    def write_outputs(self, generation_outputs):
        meta_data = {
            "dataset": self.args.dataset,
            "template_name": self.args.template_name,
            "gen_type": self.args.gen_type,
            "source_label": self.args.source_label,
            "target_label": self.args.target_label,
            "start": self.args.start,
            "end": self.args.end
        }

        output_file = {
            "meta_data": meta_data,
            "outputs": generation_outputs
        }

        timestr = time.strftime("%Y%m%d-%H%M%S")
        output_pth = os.path.join(
            self.args.output_dir,
            f"{timestr}.json",
        )
        write_json(output_file, output_pth,)
        logger.info(f"Outputs written to {output_pth}.")
    
    def main_loop(self):
        querys = self.compose_loop()
        generation_outputs = self.generate_loop(querys)
        if self.args.do_filter:
            self.filter_loop()
        self.write_outputs(generation_outputs)


def setup_path(args):
    category = f"{args.source_label}_{args.target_label}"
    output_dir = os.path.join(
        args.data_dir, args.dataset, "output", category)
    if(not os.path.isdir(output_dir)):
        os.makedirs(output_dir, exist_ok=True)

    source = f"{args.source_label}.jsonl"
    input_pth = os.path.join(
        args.data_dir, args.dataset, "input", source)
    
    demo_pth = os.path.join(
        args.data_dir, args.dataset, "examples", category)
    
    aug_pth = os.path.join(
        args.data_dir, args.dataset, "augment", category)

    with open_dict(args):
        args.output_dir = output_dir
        args.input_pth = input_pth
        args.demo_pth = demo_pth
        args.aug_pth = aug_pth


@hydra.main(config_path="config/secret/", config_name="keys", version_base="1.3.2")
def set_up_api(keys: DictConfig):
    openai.organization = keys.organization_token
    openai.api_key = keys.api_token


@hydra.main(config_path="config", config_name="config", version_base="1.3.2")
def main(args: DictConfig):
    # wandb_run = wandb.init(
    #     project=args.project,
    #     entity=args.entity,
    #     name=args.name,
    # )

    input_cache, output_cache = get_database(args.dataset, args.template_name)
    runner = DistillerRunner(args, input_cache, output_cache)
    setup_path(args)
    runner.main_loop()



if __name__ == "__main__":
    set_up_api()
    main()