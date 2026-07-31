from pathlib import Path
import json


class AttrDict(dict):

    __getattr__ = dict.__getitem__

    def __setattr__(self, key, value):
        self[key] = value


def __dict2attr(
    data: dict,
) -> AttrDict:

    attr = AttrDict()

    if isinstance(data, dict):
        for key, value in data.items():
            attr[key] = __dict2attr(value)
    elif isinstance(data, list):
        return [__dict2attr(x) for x in data]
    else:
        return data

    return attr


def json2attr(
    json_path: Path,
) -> AttrDict:

    with open(str(json_path), 'r') as f:
        json_data = json.load(f)
    return __dict2attr(json_data)
