from torch.utils.data import DataLoader, Dataset

label_map = {
    "entailment": "true",
    "contradiction": "false",
    "neutral": "irrelevant"
}

label_map_insert = {
    "entailment": "We definitely can conclude that",
    "contradiction": "We definitely can not conclude that",
    "neutral": "It is possible that"
}

cache_map = {
    "snli": {
        "e2c": 10,
        "c2e": 11,
        "e2n": 2,
        "c2n": 3
    },
    "wanli": {
        "e2c": 4,
        "c2e": 5,
        "e2n": 6,
        "c2n": 7
    }
}

type_map = {
    "e2c": ["entailment", "contradiction"],
    "c2e": ["contradiction", "entailment"],
    "e2n": ["entailment", "neutral"],
    "c2n": ["contradiction", "neutral"]
}

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