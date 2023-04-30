import os
import json
import yaml
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger("cad_generator.util")


def to_jsonl(data):
    return json.dumps(data).replace("\n", "")


def write_file(data, path, mode="w", **kwargs):
    with open(path, mode=mode, **kwargs) as f:
        f.write(data)


def read_jsonl(path, mode="r", **kwargs):
    ls = []
    with open(path, mode, **kwargs) as f:
        for line in f:
            ls.append(json.loads(line))
    return ls


def write_jsonl(data, path, mode="w"):
    assert isinstance(data, list)
    lines = [to_jsonl(elem) for elem in data]
    write_file("\n".join(lines) + "\n", path, mode=mode)


def read_from_yaml_file(dataset_name):
    """
    Reads a file containing a prompt collection.
    """
    yaml_path = f"./template/{dataset_name}"
    if not os.path.exists(yaml_path):
        logging.warning(
            f"Tried instantiating `DatasetTemplates` for {dataset_name}, but no prompts found. "
            "Please ignore this warning if you are creating new prompts for this dataset."
        )
        return {}
    yaml_dict = yaml.load(open(yaml_path, "r"), Loader=yaml.FullLoader)
    return yaml_dict
