import json
import openai
import random
import logging

from .base import (
    Example,
    Prompt
)

from .demonstration import (
    build_masked_nli_perturbation,
    demonstration_search
)

from .prompt import (
    build_problems,
    build_problems_insertion
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger("cad_generator.generator")


def prompt_perturbation_insertion(args, cache, encoder=None):
    generation_outputs = []

    logger.info("Build prompt: enumerate problems")
    guids, problems = build_problems_insertion(args, cache)

    logger.info(f"Prompting {len(problems)} problems ...")
    for i, guid in enumerate(guids):
        record = json.loads(cache.get(guid))
        if not record["accept"]:
            if i > 0 and i % 100 == 0:
                logger.info(
                    f"=============== Prompting progress: {len(generation_outputs)} problems ===============")

            response = openai.Completion.create(
                model="text-davinci-002",
                prompt=record['prefix'],
                suffix=record['suffix'],
                temperature=0.8,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0.8,
                presence_penalty=0.5,
                stop=["stop", "\n", "."]
            )

            output = response['choices'][0]['text'].replace("\n", "").strip()
            record["gen_out"] = output
            record["score"] = 0.0
            record[f"new_{args.mode}"] = record[args.mode].replace(
                record["span_prev"], output
            )
            cache.set(guid, json.dumps(record))
            generation_outputs.append(record)

    logger.info(f"Receiving {len(generation_outputs)} generation outputs")
    return generation_outputs


def prompt_perturbation(args, cache, encoder=None):
    logger.info("Build prompt: sample demonstrations")
    instruction, perturbations = build_masked_nli_perturbation(args)

    logger.info(f"Prompting Instruction: {instruction}")
    random.shuffle(perturbations)

    logger.info("Build prompt: enumerate problems")
    guids, problems = build_problems(args, cache)
    gpt_prompt = Prompt()

    if args.prompt_search:
        examples_selected = demonstration_search(
            args, perturbations, problems, encoder)

        assert len(examples_selected) == len(problems)

    generation_outputs = []
    logger.info(f"Prompting {len(problems)} problems ...")

    for i, guid in enumerate(guids):
        record = json.loads(cache.get(guid))
        if not record["accept"]:
            if i > 0 and i % 100 == 0:
                logger.info(
                    f"=============== Prompting progress: {len(generation_outputs)} problems ===============")

            if args.prompt_search:
                examples = [perturbations[j] for j in examples_selected[i]]
                examples.reverse()
            else:
                random.shuffle(perturbations)
                examples = perturbations[:args.num_neighbors]

            gpt_prompt.delete_all_examples()
            for example in examples:
                demonstration = Example(example["prompt"], example["output"])
                gpt_prompt.add_example(demonstration)

            prompt = gpt_prompt.craft_query(
                record['prompt'],
                instruction=instruction)

            response = openai.Completion.create(
                engine="text-davinci-002",
                prompt=prompt,
                top_p=1.0,
                temperature=0.8,
                max_tokens=256,
                frequency_penalty=0.8,
                presence_penalty=0.5
            )

            output = response['choices'][0]['text']
            output = output.replace("\n", "").strip()
            record["gen_out"] = output.replace(".", "")
            record[f"new_{args.mode}"] = record[args.mode].replace(
                record["span_prev"], output
            )
            cache.set(guid, json.dumps(record))
            generation_outputs.append(record)

    logger.info(f"Receiving {len(generation_outputs)} generation outputs")

    return generation_outputs
