import json

# e2c = read_jsonl("./data/e2c-exp.jsonl")
# r_cache = redis.Redis(host='localhost', port=6379, db=0)


# keys = []
# for data in e2c:
#     keys.append(data['premise'])

# for guid in r_cache.keys():
#     record = json.loads(r_cache.get(guid))
#     if record['premise'] in keys:
#         r_cache.delete(guid)


def to_jsonl(data):
    return json.dumps(data).replace("\n", "")


def write_file(data, path, mode="w", **kwargs):
    with open(path, mode=mode, **kwargs) as f:
        f.write(data)


def read_jsonl(path, mode="r", **kwargs):
    # Manually open because .splitlines is different from iterating over lines
    ls = []
    with open(path, mode, **kwargs) as f:
        for line in f:
            ls.append(json.loads(line))
    return ls


def write_jsonl(data, path, mode="w"):
    assert isinstance(data, list)
    lines = [to_jsonl(elem) for elem in data]
    write_file("\n".join(lines) + "\n", path, mode=mode)
