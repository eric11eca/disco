import random

from hydra import compose

from distiller.api import GPTQuery
from distiller.utils import read_jsonl
from distiller.db import update
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
        for example in demonstration:
            gpt_prompt.add_example(example)
        prompt = gpt_prompt.craft_query(record, model)
        return prompt

    @staticmethod
    def _load_base_data(args):
        base_data = read_jsonl(args.data_pth)
        return base_data

    @classmethod
    def build_prompts(cls, args, cache):
        instances = cls._load_base_data(args) 

        templates = cls._load_template(
            cls.TASK, args.template_name)
        
        instruction = cls.Composer._compose_instruction(
            templates.instruction,
            {
                "answer_choices": templates.answer_choices,
                "label": args.target_label
            },
        )

        records = cls.Composer.read_and_compose(
            args, cache, instances[:10], templates)

        demonstration = cls.ExampleReader.jsonl_file_reader(args.demo_pth)
        demonstration = random.choices(demonstration, k=4)

        querys = []
        for record in records:
            query = cls._craft(
                record,
                demonstration,
                instruction,
                args.gen_type)
            querys.append((record, query))
        return querys
    
    @classmethod
    def postprocess_generation(cls, cache, generated):        
        for record in generated:
            old_data = record.__dict__()[record.mode]
            span_prev = record.span_prev
            new_data = old_data.replace(span_prev, record.gen_out)
            update(cache, {"guid": record.guid}, {
                "$set": {
                    "gen_out": record.gen_out,
                    "score": 0.0,
                    f"new_{record.mode}": new_data
                }})

        return generated