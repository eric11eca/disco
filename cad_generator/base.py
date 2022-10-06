import uuid

from torch.utils.data import Dataset

label_map = {
    "entailment": "true",
    "contradiction": "false",
    "neutral": "irrelevant"
}

label2id = {
    "entailment": 0,
    "contradiction": 2,
    "neutral": 1
}

label_map_insert = {
    "entailment": "We definitely can conclude that",
    "contradiction": "We definitely can not conclude that",
    "neutral": "It is possible that"
}

cache_map = {
    "snli": {
        "e2c": (0, 10),
        "c2e": (1, 11),
        "e2n": (2, 12),
        "c2n": (3, 13),
        "n2c": (16, 17),
        "n2e": (18, 19),
    },
    "anli": {
        "e2c": (4, 8),
        "c2e": (5, 9),
        "e2n": (6, 14),
        "c2n": (7, 15),
        "n2c": (20, 21),
        "n2e": (22, 23),
    },
    "wanli": {
        "e2c": (24, 25),
        "c2e": (26, 27),
        "e2n": (28, 29),
        "c2n": (30, 31)
    },
}

type_map = {
    "e2c": ["entailment", "contradiction"],
    "c2e": ["contradiction", "entailment"],
    "e2n": ["entailment", "neutral"],
    "c2n": ["contradiction", "neutral"],
    "n2c": ["neutral", "contradiction"],
    "n2e": ["neutral", "entailment"],
}

forbidden_phrases = [
    'It is true', 'It is false',
    'it is true', 'it is false',
    '[blank]', 'story:', 'conclusion:',
    'context:', 'answer:'
]

negation_words = [
    "not", "no", "none",
    "doesn’t", "isn’t", "wasn’t",
    "shouldn’t", "wouldn’t",
    "couldn’t", "won’t", "can’t", "don’t"
]


class Example:
    """Stores an input, output pair and formats it to prime the model."""

    def __init__(self, inp, out):
        self.input = inp
        self.output = out
        self.id = uuid.uuid4().hex

    def get_input(self):
        """Returns the input of the example."""
        return self.input

    def get_output(self):
        """Returns the intended output of the example."""
        return self.output

    def get_id(self):
        """Returns the unique ID of the example."""
        return self.id

    def as_dict(self):
        return {
            "input": self.get_input(),
            "output": self.get_output(),
            "id": self.get_id(),
        }


class Prompt:
    """The main class for a user to create a prompt for GPT3"""

    def __init__(self) -> None:
        self.examples = []

    def add_example(self, ex):
        """
        Adds an example to the object.
        Example must be an instance of the Example class.
        """
        assert isinstance(ex, Example), "Please create an Example object."
        self.examples.append(ex)

    def delete_example(self, id):
        """Delete example with the specific id."""
        if id in self.examples:
            del self.examples[id]

    def delete_all_examples(self):
        self.examples = []

    def get_example(self, id):
        """Get a single example."""
        return self.examples.get(id, None)

    def get_all_examples(self):
        """Returns all examples as a list of dicts."""
        return {k: v.as_dict() for k, v in self.examples.items()}

    def craft_query(self, input, instruction=""):
        """Creates the query for the API request."""
        prompt = f"{instruction} \n\n"
        for example in self.examples:
            prompt += f"{example.get_input()} {example.get_output()}\n\n"
        prompt += input

        return prompt


class FilterDataset(Dataset):
    def __init__(self, counter_data):
        self.counter_data = counter_data

    def __getitem__(self, index):
        return self.counter_data[index]

    def __len__(self):
        return len(self.counter_data)
