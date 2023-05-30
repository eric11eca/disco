import os
import time
import torch
import openai
import hydra
import logging
import transformers
from tqdm import tqdm

from hydra import initialize
from omegaconf import DictConfig, open_dict

from distiller.db import get_database, get_all
from distiller.prompt.retrieval import get_task_class
from distiller.generator import Generator
from distiller.utils import write_json
from distiller.filter.core import FilterDataLoader

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger("distiller.runner")

transformers.logging.set_verbosity_error()


class DistillerRunner:
    def __init__(self, args, input_cache, output_cache):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.backends.mps.is_available():
            self.device = "mps"
        self.generator = Generator(args)
        self.input_cache = input_cache
        self.output_cache = output_cache
        self.generation_outputs = []
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        initialize(config_path="./templates", version_base="1.3.2")
        self.task_container = get_task_class(self.args.task_name)

    def compose_loop(self):
        logger.info("Loading input data and composing prompts ...")
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
    
    def filter_loop(self, generation_outputs):
        logger.info("Filtering generation outputs ...")
        dataloader = FilterDataLoader(generation_outputs, 16).dataloader
        filters = self.task_container.FILTERS
        filter_args = self.task_container.FILTER_ARGS
        filter_args["device"] = self.device
        for filter_class in filters:
            task_filter = filter_class()
            for batch in tqdm(dataloader):
                task_filter.run(batch, self.output_cache, **filter_args)
        logger.info("Filtering complete.")
    
    def filter_all_loop(self):
        logger.info("Filtering all previous generation outputs ...")
        all_outputs = get_all(self.output_cache)
        # all_records = [
        #     self.task_container.Prompt.from_dict(**record) for record in all_outputs]
        dataloader = FilterDataLoader(all_outputs, 16).dataloader
        filters = self.task_container.FILTERS
        filter_args = self.task_container.FILTER_ARGS
        filter_args["device"] = self.device
        for filter_class in filters:
            task_filter = filter_class()
            for batch in tqdm(dataloader):
                task_filter.run(batch, self.output_cache, **filter_args)
        logger.info("Filtering complete.")
        self.report(all_outputs)
            
    def report(self, outputs):
        accepted = [record for record in outputs if record['accept']]
        rejected = [record for record in outputs if not record['accept']]
        logger.info(f"Report: {len(accepted)} accepted and {len(rejected)} rejected.")
        return accepted
    
    def write_outputs(self, generation_outputs):
        meta_data = {
            "dataset": self.args.dataset,
            "template_name": self.args.template_name,
            "gen_type": self.args.gen_type,
            "engine": self.args.model_name,
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

        all_outputs = get_all(self.output_cache)
        accepted = self.report(all_outputs)
        augment_file = {
            "meta_data": meta_data,
            "outputs": accepted
        }
        augment_pth = os.path.join(
            self.args.aug_pth,
            f"{timestr}.json",
        )
        write_json(augment_file, augment_pth)
        logger.info(f"Augmentations written to {augment_pth}.")
    
    def main_loop(self):
        querys = self.compose_loop()
        generation_outputs = self.generate_loop(querys)
        self.filter_loop(generation_outputs)
        self.write_outputs(generation_outputs)
        


def setup_path(args):
    category = f"{args.source_label}_{args.target_label}"
    output_dir = os.path.join(
        args.data_dir, args.dataset, "output", category)
    
    source = f"{args.source_label}.jsonl"
    input_dir = os.path.join(
        args.data_dir, args.dataset, "input", source)
    
    demo_dir = os.path.join(
        args.data_dir, args.dataset, "examples", category)
    
    aug_dir = os.path.join(
        args.data_dir, args.dataset, "augment", category)
    
    if(not os.path.isdir(output_dir)):
        os.makedirs(output_dir, exist_ok=True)
    if(not os.path.isdir(input_dir)):
        os.makedirs(input_dir, exist_ok=True)
    if(not os.path.isdir(demo_dir)):
        os.makedirs(demo_dir, exist_ok=True)
    if(not os.path.isdir(aug_dir)):
        os.makedirs(aug_dir, exist_ok=True)
    
    with open_dict(args):
        args.output_dir = output_dir
        args.input_pth = input_dir
        args.demo_pth = demo_dir
        args.aug_pth = aug_dir


@hydra.main(config_path="config/secret/", config_name="keys", version_base="1.3.2")
def set_up_api(keys: DictConfig):
    openai.organization = keys.organization_token
    openai.api_key = keys.api_token


@hydra.main(config_path="config", config_name="config", version_base="1.3.2")
def main(args: DictConfig):
    input_cache, output_cache = get_database(args.dataset, args.template_name)
    runner = DistillerRunner(args, input_cache, output_cache)
    setup_path(args)
    if args.do_filter:
        runner.filter_all_loop()
    else:
        runner.main_loop()

if __name__ == "__main__":
    set_up_api()
    main()