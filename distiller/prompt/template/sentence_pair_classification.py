from typing import List, Optional
from dataclasses import dataclass

from distiller.prompt.core import (
    BasePrompt,
    BaseExample,
    BaseComposer,
    BaseExampleReader,
)

from distiller.utils import read_jsonl
from distiller.db import get_database


@dataclass
class SentencePairPrompt(BasePrompt):
    guid: str
    sentence1: str
    sentence2: str
    label: str
    new_label: str
    sentence1_spans: Optional[List[str]]
    sentence2_spans: Optional[List[str]]
    prompt: str
    prefix: Optional[str]
    suffix: Optional[str]
    span_prev: Optional[str]
    gen_out: str
    score: float
    accept: bool

    def __dict__(self) -> dict:
        return {
            "guid": self.guid,
            "sentence1": self.sentence1,
            "sentence2": self.sentence2,
            "label": self.label,
            "new_label": self.new_label,
            "prompt": self.prompt,
            "prefix": self.prefix,
            "suffix": self.suffix,
            "span_prev": self.span_prev,
            "gen_out": self.gen_out,
            "score": self.score,
            "accept": self.accept,
        }


@dataclass
class SentencePairExample(BaseExample):
    guid: str
    text_input: str
    text_output: str

    def __dict__(self) -> dict:
        return {
            "guid": self.guid,
            "text_input": self.text_input,
            "text_output": self.text_output,
        }


class SentencePairExampelReader(BaseExampleReader):
    @ staticmethod
    def _read(instance):
        """Reads a single json line from the target file. Modify here when the json schema changes

        :param instance: the instance to be read
        :rtype instance: situation_modeling.readers.input_example.InputBase
        """
        guid = instance["guid"]
        text_input = instance["input"]
        text_output = instance["output"]
        return SentencePairExample(
            guid=guid,
            text_input=text_input,
            text_output=text_output
        )


class SentencePairComposer(BaseComposer):
    @ staticmethod
    def _build_promtp_instance(
        template,
        template_insert,
        instance,
        span, 
        answer_choices,
        prompt_idx,
    ):
        guid = instance["guid"]
        sentence1 = instance["sentence1"]
        sentence2 = instance["sentence2"]
        label = instance["label"]
        new_label = instance["new_label"]

        render_items = {
            "sentence1": sentence1,
            "sentence2": sentence2,
            "span": span,
            "label": new_label,
            "answer_choices": answer_choices
        }

        prompt = SentencePairComposer._compose_prompt(
            template=template,
            render_items=render_items
        )

        prompt_insert = SentencePairComposer._compose_prompt(
            template=template_insert,
            render_items=render_items
        )
        prefix = prompt_insert.split("[insert]")[0]
        suffix = prompt_insert.split("[insert]")[1]

        prompt_instance = SentencePairPrompt(
            guid=f"{guid}-id={prompt_idx}",
            sentence1=sentence1,
            sentence2=sentence2,
            label=label,
            new_label=new_label,
            prompt=prompt,
            prefix=prefix,
            suffix=suffix,
            span_prev=span,
            gen_out="",
            score=0.0,
            accept=False,
        )
        return prompt_instance

    @ staticmethod
    def _read(instance, templates, mode):
        """Reads a single json line from the target file. Modify here when the json schema changes

        :param instance: the instance to be read
        :param templates: the templates to be used for prompt
        :rtype problem: List[SentencePairPrompt]
        """
        sentence1_spans = list(set(instance["sentence1_spans"]))
        sentence2_spans = list(set(instance["sentence2_spans"]))
        problems = []

        template = templates.template
        template_insert = templates.template_insert
        answer_choices = templates.answer_choices

        if mode == "sentence1":
            problems1 = [
                SentencePairComposer._build_promtp_instance(
                    template, template_insert, instance, span, answer_choices, i)
                for i, span in enumerate(sentence1_spans)]
            problems.extend(problems1)
        elif mode == "sentence2":
            problems2 = [
                SentencePairComposer._build_promtp_instance(
                    template, template_insert, instance, span, answer_choices, i)
                for i, span in enumerate(sentence2_spans)]
            problems.extend(problems2)
        return problems

    @classmethod
    def read_and_compose(cls, instances, cache, templates):
        """The method responsible for parsing in the input file. Implemented here
        to make the overall pipeline more transparent.

        :param instances: instances to be processed for perturbation
        :param cache: the database instance
        :param templates: the templates to be used for prompt
        """
        mode = templates.mode
        seed_records = []

        for instance in instances:
            records = cls._read(instance, templates, mode)
            seed_records.extend(records),

        for record in seed_records:
            record["accept"] = cls._commit(record, cache)

        seed_records = [
            record for record in seed_records if not record["accept"]]

        return seed_records


def main():
    cache = get_database(
        dataset="snli",
        type="masked_cad_premise"
    )

    example_reader = SentencePairExampelReader

    input_file = "data/snli/input/neutral.jsonl"
    input_instances = read_jsonl(input_file)
    examples = example_reader.jsonl_file_reader(input_instances)
    print(examples)

if __name__ == "__main__":
    main()

