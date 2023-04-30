import json
import yaml
import uuid
import random
import logging

from enum import Enum
from jinja2 import BaseLoader, Environment, meta

from distiller.db import query, insert
from distiller.prompt.datastructure import BiMap
from distiller.utils import read_jsonl

env = Environment(loader=BaseLoader)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger("cad_generator.prompt.core")


class TaskTypes(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    SPAN_COMPARISON_CLASSIFICATION = "span_comparison_classification"
    MULTIPLE_CHOICE = "multiple_choice"
    SQUAD_STYLE_QA = "squad_style_qa"
    TAGGING = "tagging"
    MASKED_LANGUAGE_MODELING = "masked_language_modeling"
    EMBEDDING = "embedding"
    MULTI_LABEL_SPAN_CLASSIFICATION = "multi_label_span_classification"
    SPAN_PREDICTION = "span_prediction"
    UNDEFINED = "undefined"


class BasePrompt:
    def __dict__(self) -> dict:
        return {
            "guid": self.guid,
            "score": self.score,
            "accept": self.accept,
        }

    def __str__(self) -> str:
        return json.dumps(self.__dict__(), indent=2)


class BaseExample:
    """Stores an input, output pair and formats it to prime the model."""
    guid: str
    text_input: str
    text_output: str

    def __dict__(self) -> dict:
        return {
            "guid": self.guid,
            "text_input": self.score,
            "text_output": self.accept,
        }

    def __str__(self) -> str:
        return json.dumps(self.__dict__(), indent=2)


class BaseComposer:
    def __init__(self, cache, templates):
        self.cache = cache
        self.templates = templates

    @staticmethod
    def _compose_prompt(template, render_items):
        """Compose a prompt for the sentence pair
           Can be overriden by subclasses

        :param template: the template to be used for prompt
        :param render_items: vrariables for template rendering
        :rtype prompt: str
        """
        prompt = env.from_string(template).render(**render_items)
        return prompt
    
    @staticmethod
    def _compose_instruction(instruction, render_items):
        return env.from_string(instruction).render(**render_items)

    @staticmethod
    def _read(instance, templates):
        """process a sinlge instance.

        :param instance: the instance to be read
        :param templates: the templates to be used for prompt
        """
        NotImplemented

    @staticmethod
    def _commit(record, cache):
        """Commit the record to the database. Modify here when the database schema changes

        :param record: the record to be committed
        :param cache: the database instance
        """
        db_record = query(cache, {"guid": record.guid})
        if db_record is not None and db_record["accept"]:
            return True
        else:
            insert(cache, record.__dict__())
            return False

    @classmethod
    def read_and_compose(cls, instances, cache, templates):
        """The method responsible for parsing in the input file. Implemented here
        to make the overall pipeline more transparent.

        :param instances: instances to be processed for perturbation
        :param cache: the database instance
        :param templates: the templates to be used for prompt
        """
        all_guids = []
        for instance in instances:
            records = cls._read(instance, templates)
            guids = [cls._commit(record, cache) for record in records]
            guids = [guid for guid in guids if guid != ""]
            all_guids.extend(guids)
        return all_guids


class BaseExampleReader:
    @ staticmethod
    def _compose_prompt(template, render_items):
        """Compose a prompt for the sentence pair
           Can be overriden by subclasses

        :param template: the template to be used for prompt
        :param render_items: vrariables for template rendering
        :rtype prompt: str
        """
        prompt = env.from_string(template).render(**render_items)
        return prompt

    @staticmethod
    def _read(instance):
        """Reads a single json line from the target file. Modify here when the json schema changes

        :param instance: the instance to be read
        :rtype instance: situation_modeling.readers.input_example.InputBase
        """
        NotImplemented

    @classmethod
    def jsonl_file_reader(cls, demo_pth):
        """The method responsible for parsing in the input file. Implemented here
        to make the overall pipeline more transparent.

        :param demo_pth: the path to the demonstration file
        """
        demonstrations = []
        instances = read_jsonl(demo_pth)
        for instance in instances:
            demonstration = cls._read(instance)
            demonstrations.append(demonstration)
        return demonstrations
    

class Task:
    Example = None
    Prompt = None
    Composer = None
    ExampleReader = None

    @staticmethod
    def prompt_search(args, examples):
        return random.choices(examples, k=args.num_neighbors)

    @classmethod
    def build_prompts(cls, args, cache):
        raise NotImplementedError


class Template(yaml.YAMLObject):
    """
    A prompt template.
    """

    yaml_tag = "!Template"

    def __init__(self, name, jinja, answer_choices=None):
        """
        Creates a prompt template.
        A prompt template is expressed in Jinja. It is rendered using an example
        from the corresponding Hugging Face datasets library (a dictionary). The
        separator ||| should appear once to divide the template into prompt and
        output. Generally, the prompt should provide information on the desired
        behavior, e.g., text passage and instructions, and the output should be
        a desired response.
        :param name: unique name (per dataset) for template
        :param jinja: template expressed in Jinja
        :param answer_choices: Jinja expression for answer choices. Should produce
                               a ||| delimited string of choices that enumerates
                               the possible completions for templates that should
                               be evaluated as ranked completions. If None, then
                               the template is open-ended. This list is accessible
                               from within Jinja as the variable `answer_choices`.
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.jinja = jinja
        self.answer_choices = answer_choices

    def get_id(self):
        """
        Returns the id of the template
        :return: unique id for template
        """
        return self.id

    def get_name(self):
        """
        Returns the name of the template
        :return: unique (per dataset) name for template
        """
        return self.name

    def get_reference(self):
        """
        Returns the bibliographic reference (or author) for the template
        :return: reference as a string
        """
        return self.reference

    def get_answer_choices_expr(self):
        """
        Returns a Jinja expression for computing the answer choices from an example.
        :return: String, or None if no answer choices
        """
        return self.answer_choices

    def get_answer_choices_list(self, example):
        """
        Returns a list of answer choices for a given example
        :return: list of strings, or None if get_answer_choices_expr is None
        """
        jinja = self.get_answer_choices_expr()
        if jinja is None:
            return None

        rtemplate = env.from_string(jinja)
        protected_example = self._escape_pipe(example)
        rendered_choices = rtemplate.render(**protected_example)
        return [self._unescape_pipe(answer_choice.strip()) for answer_choice in rendered_choices.split("|||")]

    def get_fixed_answer_choices_list(self):
        """
        Returns a list of answer choices that is static across examples, if possible
        :return: list of strings, or None if no static list exists
        """
        jinja = self.get_answer_choices_expr()
        if jinja is None:
            return None

        parse = env.parse(jinja)
        variables = meta.find_undeclared_variables(parse)
        if len(variables) == 0:
            rtemplate = env.from_string(jinja)
            rendered_choices = rtemplate.render()
            return [answer_choice.strip() for answer_choice in rendered_choices.split("|||")]
        else:
            return None


def labels_to_bimap(labels):
    """Creates mappings from label to id, and from id to label. See details in docs for BiMap.
    Args:
        labels: sequence of label to map to ids.
    Returns:
        Tuple[Dict, Dict]: mappings from labels to ids, and ids to labels.
    """
    label2id, id2label = BiMap(a=labels, b=list(range(len(labels)))).get_maps()
    return label2id, id2label
