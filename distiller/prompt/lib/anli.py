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
from distiller.filter.retrieval import FILTER_DICT


class AdversarialNliTask(Task):
    Example = SentencePairExample
    Prompt = SentencePairPrompt
    Composer = SentencePairComposer
    ExampleReader = SentencePairExampleReader

    TASK = "anli"
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
