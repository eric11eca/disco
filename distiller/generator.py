import openai
import logging

from torch.utils.data import DataLoader
from distiller.db import update


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger("cad_generator.generator")


class Generator:
    def __init__(self, args, cache):
        self.args = args
        self.cache = cache
        self.generation_outputs = []

        self.generation_modes = {
            "completion": self.completion,
            "insertion": self.insertion,
            "chat": self.chat
        }

        self.num_records = 0

    def preprocess(self, records):
        logger.info(f"Querying {len(records)} problems from DB ...")
        dataloader = DataLoader(
            records, batch_size=100, shuffle=False, collate_fn=lambda x: x)

        self.dataloader = dataloader
        self.num_records = len(records)

    def insertion(self, prompt):
        response = openai.Completion.create(
            model="text-davinci-003",
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
            engine="text-davinci-003",
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
            model="gpt-3.5-turbo",
            messages=messages
        )
        return response

    def postprocess(self, response, record):
        output = response['choices'][0]['text']
        output = output.replace("\n", "").strip()
        record["gen_out"] = output
        new_input = record[self.args.mode].replace(
            record["span_prev"], output)
        record[f"new_{self.args.mode}"] = new_input
        return record

    def batch_generate(self, records):
        self. preprocess(records)

        logger.info(f"Prompting {len(self.problems)} problems ...")
        generation_outputs = []

        for _, batch in enumerate(self.dataloader):
            batch_outputs = self.generate(batch)
            logger.info(
                f"Writing {len(batch_outputs)} generation into DB ...")

            for record in batch_outputs:
                update(self.cache, {"guid": record["guid"]}, {
                    "$set": {
                        "gen_out": record["gen_out"],
                        "score": 0.0,
                        f"new_{self.args.mode}": record[f"new_{self.args.mode}"]
                    }})

            generation_outputs.extend(batch_outputs)

        logger.info(f"Receiving {len(generation_outputs)} generation outputs")
        return generation_outputs

    def generate(self, batch):
        batch_generation_outputs = []
        for record in batch:
            if record is None:
                logger.info("Warning: record not found, skipping ...")
                continue
            if record["accept"]:
                logger.info("Warning: accepted record, skipping ...")
                continue

            generator = self.generation_modes.get(self.args.gen_type, None)
            if generator is None:
                raise NotImplementedError("Unknown Generation Mode")
            response = generator(record['query'])
            updated_record = self.postprocess(response, record)
            batch_generation_outputs.append(updated_record)
        return batch_generation_outputs
