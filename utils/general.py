import sys
import json
import pickle


__all__ = [
    "save_json",
    "load_json",
    "save_pickle",
    "load_pickle",
    "save_script_args",
]


def save_json(data: dict, path: str):
    with open(path, "w+") as file:
        json.dump(data, file, indent = 2)


def load_json(path: str):
    with open(path) as file:
        return json.load(file)


def save_pickle(data, path: str):
    with open(path, 'wb') as file:
        pickle.dump(data, file)


def load_pickle(path: str):
    with open(path, 'rb') as file:
        return pickle.load(file)


def save_script_args(path):
    # Command to run the current program
    CMD = ' '.join(sys.argv)
    CMD = "python {}\n".format(CMD)
    
    # Save this command in file
    with open(path, 'a+') as file:
        file.write(CMD)
       
