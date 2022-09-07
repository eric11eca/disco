import redis
import json
import string

from counterfactual_filter import read_jsonl, write_jsonl

e2c = read_jsonl("./data/e2c-exp.jsonl")
r_cache = redis.Redis(host='localhost', port=6379, db=0)


keys = []
for data in e2c:
    keys.append(data['premise'])

for guid in r_cache.keys():
    record = json.loads(r_cache.get(guid))
    if record['premise'] in keys:
        r_cache.delete(guid)
