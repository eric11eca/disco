import random

from hydra import compose

from distiller.api import GPTQuery
from distiller.utils import read_jsonl
from distiller.db import insert
from distiller.prompt.core import labels_to_bimap, Task
from distiller.prompt.template.sentence_pair_classification import (
    SentencePairPrompt,
    SentencePairExample,
    SentencePairComposer,
    SentencePairExampleReader,
)
from distiller.filter.retrieval import FILTER_DICT

class SNLITask(Task):
    Example = SentencePairExample
    Prompt = SentencePairPrompt
    Composer = SentencePairComposer
    ExampleReader = SentencePairExampleReader

    TASK = "snli"
    NUM_CHOICES = 3
    LABELS = ["entailment", "neutral", "contradiction"]
    LABEL_TO_ID, ID_TO_LABEL = labels_to_bimap(LABELS)

    FILTER_ARGS = {
        "threshold": 0.5,
        "model_names": [
            # "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
            # "ynie/xlnet-large-cased-snli_mnli_fever_anli_R1_R2_R3-nli",
            # "alisawuffles/roberta-large-wanli"
            "Joelzhang/deberta-v3-large-snli_mnli_fever_anli_R1_R2_R3-nli"
        ],
        "forbidden": [
            "it is true",
            "it is false",
            "it is true",
            "it is false",
            "[blank]",
            "story:",
            "conclusion:",
            "context:",
            "answer:",
            "statement:",
        ],
        "negations": [
            "no", "not", "none", "nobody", "nothing", "neither", "nowhere", "never",
            "isn't", "wasn't", "shouldn't", "wouldn't", "couldn't", "won't", "can't",
            "don't", "doesn't", "didn't", "aren't", "weren't", "shouldn't've", "wouldn't've",
            "couldn't've", "won't've", "can't've", "don't've", "doesn't've", "didn't've",
            "aren't've", "weren't've", "should not", "would not", "could not", "will not",
        ]
    }

    FILTERS = [
        FILTER_DICT.SENTENCE_PAIR_HEURISTIC.value,
        FILTER_DICT.SENTENCE_PAIR_MODEL.value
    ]


    @staticmethod
    def _load_examples(args):
        examples_from_file = read_jsonl(args.demo_pth)
        examples = [SNLITask.Example(**example) for example in examples_from_file]
        return examples

    @staticmethod
    def _load_template(task, template_name, gen_type):
        cfg = compose(config_name=task)
        templates = cfg.templates[template_name]
        if(gen_type == "completion"):
            template = templates.template
        elif(gen_type == "insert"):
            template = templates.template_insert
        else:
            template = templates.template
        return templates, template

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
        base_data = read_jsonl(args.input_pth)
        return base_data

    @classmethod
    def build_prompts(cls, args, cache):
        instances = cls._load_base_data(args)[args.start: args.end]
        templates, template = cls._load_template(cls.TASK, args.template_name, args.gen_type)
        instruction = cls.Composer._compose_instruction(
            templates.instruction,
            {"answer_choices": templates.answer_choices, "label": args.target_label},
        )

        records = cls.Composer.read_and_compose(args, cache, instances, templates)
        if(args.no_demo):
            demonstration = []
        else:
            demonstration = cls.ExampleReader.jsonl_file_reader(args.demo_pth, template)
            demonstration = random.choices(demonstration, k=4)

        querys = []
        for record in records:
            query = cls._craft(record, demonstration, instruction, args.gen_type)
            querys.append((record, query))
        return querys

    @classmethod
    def postprocess_generation(cls, cache, generated):
        normalized = []
        for record in generated:
            record_dict = record.__dict__()
            old_data = record_dict[record.mode]
            span_prev = record.span_prev
            new_data = old_data.replace(span_prev, record.gen_out)
            record_dict["gen_out"] = record.gen_out
            record_dict["score"] = 0.0
            record_dict[f"new_{record.mode}"] = new_data
            insert(cache, record_dict)
            record_dict.pop("_id")
            normalized.append(record_dict)
        return normalized
