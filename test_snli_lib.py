import hydra
import openai

from omegaconf import OmegaConf, DictConfig
from hydra import initialize

from distiller.generator import Generator
from distiller.db import get_database
from distiller.prompt.lib.snli import SNLITask

cache = get_database(
    dataset="snli",
    type="masked_cad_premise"
)

args = OmegaConf.create({
    'model_name': 'gpt-3-003',
    'gen_type': 'completion',
    'target_label': 'entailment',
    'template_name': 'masked_cad_premise',
    'data_pth': 'data/snli/input/contradiction.jsonl',
    'demo_pth': 'data/snli/examples/contradiction_entailment.jsonl',
})

@hydra.main(config_path="./config/secret/", config_name="keys")
def set_up_api(keys: DictConfig):
    openai.organization = keys.organization_token
    openai.api_key = keys.api_token

def main():
    set_up_api()

    initialize(config_path="./templates", version_base="1.3")
    querys = SNLITask.build_prompts(args, cache)
    generator = Generator(args, cache)
    generation_outputs = generator.batch_generate(querys)
    SNLITask.postprocess_generation(cache, generation_outputs)

if __name__ == "__main__":
    main()