from tqdm import tqdm
from typing import List, Optional
from dataclasses import dataclass

from distiller.prompt.core import (
    BasePrompt,
    BaseExample,
    BaseComposer,
    BaseExampleReader,
)


@dataclass
class SentencePairPrompt(BasePrompt):
    guid: str
    sentence1: str
    sentence2: str
    label: str
    new_label: str
    prompt: str
    gen_out: str
    score: float
    accept: bool
    mode: str = None
    prefix: Optional[str] = None
    suffix: Optional[str] = None
    span_prev: Optional[str] = None
    sentence1_spans: Optional[List[str]] = None
    sentence2_spans: Optional[List[str]] = None
    new_sentence1: Optional[str] = None
    new_sentence2: Optional[str] = None

    def __dict__(self) -> dict:
        defualt = {
            "guid": self.guid,
            "sentence1": self.sentence1,
            "sentence2": self.sentence2,
            "label": self.label,
            "new_label": self.new_label,
            "prompt": self.prompt,
            "gen_out": self.gen_out,
            "score": self.score,
            "accept": self.accept,
        }

        if self.mode:
            defualt["mode"] = self.mode
        if self.prefix:
            defualt["prefix"] = self.prefix
        if self.suffix:
            defualt["suffix"] = self.suffix
        if self.span_prev:
            defualt["span_prev"] = self.span_prev
        if self.sentence1_spans:
            defualt["sentence1_spans"] = self.sentence1_spans
        if self.sentence2_spans:
            defualt["sentence2_spans"] = self.sentence2_spans

        return defualt
    
    @classmethod
    def from_dict(cls, **kwargs):
        """
        Creates a SentencePairPrompt from a dict. 
        Modify here when the json schema changes
        """
        return cls(
            guid=kwargs.get("guid", None),
            sentence1=kwargs.get("sentence1", None),
            sentence2=kwargs.get("sentence2", None),
            label=kwargs.get("label", None),
            new_label=kwargs.get("new_label", None),
            prompt=kwargs.get("prompt", None),
            gen_out=kwargs.get("gen_out", None),
            score=kwargs.get("score", None),
            accept=kwargs.get("accept", None),
            mode=kwargs.get("mode", None),
            prefix=kwargs.get("prefix", None),
            suffix=kwargs.get("suffix", None),
            span_prev=kwargs.get("span_prev", None),
            sentence1_spans=kwargs.get("sentence1_spans", None),
            sentence2_spans=kwargs.get("sentence2_spans", None),
            new_sentence1=kwargs.get("new_sentence1", None),
            new_sentence2=kwargs.get("new_sentence2", None)
        )


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


class SentencePairExampleReader(BaseExampleReader):
    @staticmethod
    def _read(instance):
        """Reads a single json line from the target file. Modify here when the json schema changes

        :param instance: the instance to be read
        :rtype instance: situation_modeling.readers.input_example.InputBase
        """
        guid = instance["guid"]
        text_input = instance["input"]
        text_output = instance["output"]
        return SentencePairExample(
            guid=guid, text_input=text_input, text_output=text_output
        )


class SentencePairComposer(BaseComposer):
    @staticmethod
    def _build_promtp_instance(
        template,
        template_insert,
        instance,
        new_label,
        span,
        mode,
        answer_choices,
        prompt_idx,
    ):
        guid = instance["guid"]
        sentence1 = instance["sentence1"]
        sentence2 = instance["sentence2"]
        label_curr = instance["label"]

        render_items = {
            "sentence1": sentence1,
            "sentence2": sentence2,
            "span": span,
            "label": new_label,
            "answer_choices": answer_choices,
        }

        prompt = SentencePairComposer._compose_prompt(
            template=template, render_items=render_items
        )

        prompt_insert = SentencePairComposer._compose_prompt(
            template=template_insert, render_items=render_items
        )

        prefix = prompt_insert.split("[insert]")[0]
        suffix = prompt_insert.split("[insert]")[1]

        prompt_instance = SentencePairPrompt(
            guid=f"{guid}-id={prompt_idx}",
            sentence1=sentence1,
            sentence2=sentence2,
            label=label_curr,
            new_label=new_label,
            prompt=prompt,
            prefix=prefix,
            suffix=suffix,
            span_prev=span,
            gen_out="",
            score=0.0,
            accept=False,
        )

        if mode != "sentences":
            prompt_instance.mode = mode

        return prompt_instance

    @staticmethod
    def _read(instance, templates, target_label):
        """Reads a single json line from the target file. Modify here when the json schema changes

        :param instance: the instance to be read
        :param templates: the templates to be used for prompt
        :param target_label: the target label for the generated data
        :rtype problem: List[SentencePairPrompt]
        """
        sentence1 = instance["sentence1"]
        sentence2 = instance["sentence2"]
        sentence1_spans = list(set(instance["sentence1_span"]))
        sentence2_spans = list(set(instance["sentence2_span"]))

        template = templates.template
        template_insert = templates.template_insert
        answer_choices = templates.answer_choices

        problems = []
        if templates.mode == "sentence1":
            problems1 = [
                SentencePairComposer._build_promtp_instance(
                    template,
                    template_insert,
                    instance,
                    target_label,
                    span,
                    templates.mode,
                    answer_choices,
                    i,
                )
                for i, span in enumerate(sentence1_spans)
                if span in sentence1
            ]
            problems.extend(problems1)
        elif templates.mode == "sentence2":
            problems2 = [
                SentencePairComposer._build_promtp_instance(
                    template,
                    template_insert,
                    instance,
                    target_label,
                    span,
                    templates.mode,
                    answer_choices,
                    i,
                )
                for i, span in enumerate(sentence2_spans)
                if span in sentence2
            ]
            problems.extend(problems2)
        return problems

    @classmethod
    def read_and_compose(cls, args, cache, instances, templates):
        """The method responsible for parsing in the input file. Implemented here
        to make the overall pipeline more transparent.

        :param args: the arguments passed in from the command line
        :param cache: the database instance
        :param instances: instances to be processed for perturbation
        :param templates: the templates to be used for prompt
        """
        seed_records = []

        print("Reading and composing prompts...")
        for instance in tqdm(instances):
            records = cls._read(instance, templates, args.target_label)
            seed_records.extend(records),

        # print("Committing prompts...")
        # for record in tqdm(seed_records):
        #     cls._commit(record, cache)

        seed_records = [record for record in seed_records]

        return seed_records
