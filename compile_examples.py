import redis
import json

from counterfactual_filter import read_jsonl, write_jsonl

r_cache= r_cache = redis.Redis(host='localhost', port=6379, db=1)

accepted = []

for guid in r_cache.keys():
    record = json.loads(r_cache.get(guid))
    if record["accept"]:
       accepted.append(record)

curr_examples = read_jsonl("./data/examples/entailment_contradiction.jsonl")[:20]
examples = []

for acc in accepted:
    examples.append({
        "premise": acc["premise"],
        "hypothesis": acc["hypothesis"],
        "gold_label": acc["label"],
        "new_label": "entailment",
        "span_changed": acc["span_prev"],
        "span_to": acc["gen_out"],
    })

curr_examples += examples
print(len(curr_examples))
write_jsonl(curr_examples, "./data/examples/entailment_contradiction.jsonl")
