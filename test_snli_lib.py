from omegaconf import OmegaConf
from hydra import initialize

from distiller.db import get_database
from distiller.prompt.lib.snli import SNLITask

cache = get_database(
    dataset="snli",
    type="masked_cad_premise"
)

initialize(config_path="./templates", version_base="1.3")

args = OmegaConf.create({
    'label': 'entailment',
    'template_name': 'masked_cad_premise',
    'data_pth': 'data/snli/input/contradiction.jsonl',
    'demo_pth': 'data/snli/examples/contradiction_entailment.jsonl',
})

def main():
    SNLITask.build_prompts(args, cache)


if __name__ == "__main__":
    main()