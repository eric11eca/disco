from hydra import compose, initialize

from cad_generator.api import GPTQuery
from cad_generator.utils import read_jsonl
from cad_generator.prompt.core import labels_to_bimap, Task
from cad_generator.prompt.template.sentence_pair_classification import (
    SentencePairPrompt,
    SentencePairExample,
    SentencePairComposer,
    SentencePairExampleReader
)


class AdversarialNliTask(Task):
    Example = SentencePairExample
    Prompt = SentencePairPrompt
    Composer = SentencePairComposer
    ExampleReader = SentencePairExampleReader

    TASK = "anli"
    NUM_CHOICES = 3
    LABELS = ["entailment", "neutral", "contradiction"]
    LABEL_TO_ID, ID_TO_LABEL = labels_to_bimap(LABELS)

    ATTR_MAP = {
        "e2c": ["entailment", "contradiction"],
        "c2e": ["contradiction", "entailment"],
        "e2n": ["entailment", "neutral"],
        "c2n": ["contradiction", "neutral"],
        "n2c": ["neutral", "contradiction"],
        "n2e": ["neutral", "entailment"],
    }  # Attributions for controlling the counterfactuals

    NO_GEN_PHRASES = [
        'it is true', 'it is false',
        'it is true', 'it is false',
        '[blank]', 'story:', 'conclusion:',
        'context:', 'answer:', 'statement:'
    ]

    @staticmethod
    def _load_examples(args):
        examples_from_file = read_jsonl(args.demo_pth)
        examples = [AdversarialNliTask.Example(
            **example) for example in examples_from_file]
        return examples

    @staticmethod
    def _load_template(task, template_name):
        initialize(config_path="templates")
        cfg = compose(config_name=task)
        template = cfg.templates[template_name]
        return template

    @staticmethod
    def _craft(record, demonstration, instruction, model):
        gpt_prompt = GPTQuery()
        gpt_prompt.add_instruction(instruction)
        gpt_prompt.add_example(demonstration)
        prompt = gpt_prompt.craft_query(record, model)
        return prompt

    @staticmethod
    def _load_base_data(args):
        base_data = read_jsonl(args.data_pth)
        return base_data

    @classmethod
    def build_prompts(cls, args, cache):
        instances = cls._load_base_data(args)

        template = cls.load_template(
            cls.TASK, args.generator_params.template)
        instruction = template.instruction

        records = cls.Composer.read_and_compose(
            instances, cache, template)

        demonstration = cls.ExampleReader.read_examples(args)

        for record in records:
            record["query"] = cls._craft(
                record,
                demonstration,
                instruction,
                args.gen_type)
        return records
