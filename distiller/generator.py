import openai
import logging

from tqdm import tqdm
from torch.utils.data import DataLoader

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger("distiller.generator")

generator_models = {
    "gpt-3-002": "text-davinci-003",
    "gpt-3-003": "text-davinci-003",
    "chatgpt": "gpt-3.5-turbo",
    "gpt-4": "gpt-4"
}


class Generator:
    def __init__(self, args, cache):
        self.args = args
        self.cache = cache
        self.generation_outputs = []
        self.num_records = 0

        self.generation_modes = {
            "completion": self.completion,
            "insertion": self.insertion,
            "chat": self.chat
        }

        self.init_generator_model(args)

    def init_generator_model(self, args):
        assert args.gen_type in self.generation_modes.keys(), "Invalid generation mode!"
        self.gen_type = args.gen_type

        assert args.model_name in generator_models.keys(), "Invalid model name!"
        if self.gen_type == "completion":
            assert not args.model_name == "chatgpt", "ChatGPT model does not support completion mode!"
            assert not args.model_name == "gpt-4", "GPT-4 model does not support completion mode!"
        elif self.gen_type == "insertion":
            assert not args.model_name == "chatgpt", "ChatGPT model does not support insertion mode!"
            assert not args.model_name == "gpt-4", "GPT-4 model does not support insertion mode!"
        elif self.gen_type == "chat":
            assert not args.model_name == "gpt-3-002", "GPT-3-002 model does not support chat mode!"
            assert not args.model_name == "gpt-3-003", "GPT-3-003 model does not support chat mode!"

        self.model = generator_models[args.model_name]
        self.generator = self.generation_modes.get(self.gen_type)

    def preprocess(self, querys):
        logger.info(f"Querying {len(querys)} problems from DB ...")
        dataloader = DataLoader(
            querys, batch_size=100, shuffle=False, collate_fn=lambda x: x)

        self.dataloader = dataloader
        self.num_records = len(querys)

    def insertion(self, prompt):
        response = openai.Completion.create(
            model=self.model,
            prompt=prompt['prefix'],
            suffix=prompt['suffix'],
            temperature=0.8,
            max_tokens=256,
            top_p=1.0,
            frequency_penalty=0.8,
            presence_penalty=0.5,
            stop=["stop", "\n", "."]
        )
        return response

    def completion(self, prompt):
        response = openai.Completion.create(
            model=self.model,
            prompt=prompt,
            top_p=1.0,
            temperature=0.8,
            max_tokens=256,
            frequency_penalty=0.8,
            presence_penalty=0.5
        )
        return response

    def chat(self, prompt):
        messages = [{"role": "system", "content": prompt['instruction']}]
        for example in prompt['examples']:
            messages.append(
                {"role": "user", "content": example.text_input})
            messages.append(
                {"role": "assistant", "content": example.text_output})
        messages.append({"role": "user", "content": prompt['problem']})
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages
        )
        return response

    def postprocess(self, response, record):
        output = response['choices'][0]['text']
        output = output.replace("\n", "").strip()
        record.gen_out = output
        return record

    def batch_generate(self, querys):
        self. preprocess(querys)

        logger.info(f"Prompting {self.num_records} problems ...")
        generation_outputs = []

        for batch in tqdm(self.dataloader):
            batch_outputs = self.generate(batch)
            logger.info(
                f"Collecting {len(batch_outputs)} generation ...")
            generation_outputs.extend(batch_outputs)

        logger.info(f"Total {len(generation_outputs)} generation outputs collected")
        return generation_outputs

    def generate(self, batch):
        batch_generation_outputs = []
        for record, query in batch:
            if record is None:
                logger.info("Warning: record not found, skipping ...")
                continue
            if record.accept:
                logger.info("Warning: accepted record, skipping ...")
                continue
            response = self.generator(query)
            updated_record = self.postprocess(response, record)
            batch_generation_outputs.append(updated_record)
        return batch_generation_outputs
