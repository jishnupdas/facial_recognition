import os
import json
from PIL import Image
import cv2
import numpy as np
from pygame import mixer

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


# function to get the images and label data
def get_images_and_labels(path):
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert("L")  # grayscale
        img_numpy = np.array(PIL_img, "uint8")
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y : y + h, x : x + w])
            ids.append(id)
    return faceSamples, ids


def get_camera(camera_id=0, height=480, width=640):
    cam = cv2.VideoCapture(camera_id)
    cam.set(3, width)  # set video width
    cam.set(4, height)  # set video height
    return cam


def get_alarm_sound(path="alarm.wav"):
    mixer.init()
    sound = mixer.Sound(path)
    return sound
