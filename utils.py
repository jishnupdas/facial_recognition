import os
import json

cascadePath = "haarcascade_frontalface_default.xml"
faces_library = "faces_lib.json"

def save_to_disk(face_dict, path="faces_lib.json"):
    if os.path.exists(path):  # read existing data
        with open(path, "r") as f:
            data = json.load(f)
    else:
        data = {}
    with open(path, "w+") as f:  # add new data
        data.update(face_dict)
        json.dump(data, f)  # save to disk


def get_faces_lib(path="faces_lib.json"):
    if os.path.exists(path):  # read existing data
        with open(path, "r") as f:
            data = json.load(f)
    else:
        data = {}
    return data