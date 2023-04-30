from distiller.prompt.core import BaseExample


negation_words = [
    "not", "no", "none",
    "doesn’t", "isn’t", "wasn’t",
    "shouldn’t", "wouldn’t",
    "couldn’t", "won’t", "can’t", "don’t"
]


class GPTQuery:
    """The main class for a user to create a query for OPENAI's API."""

    def __init__(self) -> None:
        self.examples = []
        self.instruction = ""
        self.craft_methods = {
            "chat": self.craft_chat_query,
            "completion": self.craft_competion_query,
            "insert": self.craft_insert_query
        }

    def add_instruction(self, instruction):
        """Adds an instruction to the prompt."""
        self.instruction = instruction

    def add_example(self, ex):
        """
        Adds an example to the object.
        Example must be an instance of the Example class.
        """
        assert isinstance(
            ex, BaseExample), "Please create an BaseExample object."
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

    def craft_chat_query(self, record):
        """Creates the query for the ChatGPT API request."""
        prompt = {
            "instruction": self.instruction,
            "examples": self.examples,
            "input": record["prompt"]
        }
        return prompt

    def craft_competion_query(self, record):
        """Creates the query for the Compeletion API request."""
        if self.instruction == "":
            prompt = ""
        else:
            prompt = f"{self.instruction} \n\n"
        for example in self.examples:
            prompt += f"{example.text_input}\n{example.text_output}\n\n"
        prompt += record.prompt

        return prompt

    def craft_insert_query(self, record):
        """Creates the query for the Insert API request."""
        prompt = {
            "prefix": record["prefix"],
            "suffix": record["suffix"]
        }

        return prompt

    def craft_query(self, record, model="completion"):
        """Creates the query for the API request."""
        craft_method = self.craft_methods[model]
        prompt = craft_method(record)
        return prompt
