import os

from hydra import compose, initialize

from distiller.api import GPTQuery
from distiller.utils import read_jsonl
from distiller.prompt.core import labels_to_bimap, Task
from distiller.prompt.template.sentence_pair_classification import (
    SentencePairPrompt,
    SentencePairExample,
    SentencePairComposer,
    SentencePairExampleReader
)


class SNLITask(Task):
    Example = SentencePairExample
    Prompt = SentencePairPrompt
    Composer = SentencePairComposer
    ExampleReader = SentencePairExampleReader

    TASK = "snli"
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
        examples = [SNLITask.Example(
            **example) for example in examples_from_file]
        return examples

    @staticmethod
    def _load_template(task, template_name):  
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

        template = cls._load_template(
            cls.TASK, args.template_name)
        
        instruction = cls.Composer._compose_instruction(
            template.instruction,
            {
                "answer_choices": template.answer_choices,
                "label": args.label
            },
        )

        records = cls.Composer.read_and_compose(
            instances, cache, template)
        
        print(records[0])

        demonstration = cls.ExampleReader.jsonl_file_reader(args.demo_pth)

        print(demonstration[0])

        for record in records:
            record.query = cls._craft(
                record,
                demonstration,
                instruction,
                args.gen_type)
        return records