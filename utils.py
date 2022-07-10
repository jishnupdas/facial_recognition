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


def register_face():
    """
    register a face and save 30 images of the face to disk
    """
    cam = get_camera(camera_id=0, height=480, width=640)

    face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    # For each person, enter one numeric face id
    face_id = input("\n enter user id and press <enter> ==>  ")
    name = input("\n enter user's name and press <enter> ==>  ")
    print("\n [INFO] Initializing face capture. Look the camera and wait ...")
    # Initialize individual sampling face count
    count = 0
    while True:
        ret, img = cam.read()
        img = cv2.flip(img, 1)  # flip video image vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            # Save the captured image into the datasets folder
            cv2.imwrite(
                f"dataset/User.{face_id}.{count}.jpg",
                gray[y : y + h, x : x + w],
            )
            cv2.imshow("image", img)

        k = cv2.waitKey(100) & 0xFF  # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= 30:  # Take 30 face sample and stop video
            break
    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()
    # save the name of the person
    save_to_disk({face_id: name})


def train_model():
    # Path for face image database
    path = "dataset"
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces, ids = get_images_and_labels(path)
    recognizer.train(faces, np.array(ids))
    # Save the model into trainer/trainer.yml
    recognizer.write("trainer/trainer.yml")
    print(f"\n [INFO] {len(np.unique(ids))} faces trained. Exiting Program")


def detect_faces():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer/trainer.yml")
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    font = cv2.FONT_HERSHEY_SIMPLEX

    names = get_faces_lib()

    # Initialize and start realtime video capture
    cam = get_camera(camera_id=0, height=480, width=640)
    # Define min window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)
    while True:
        ret, img = cam.read()
        img = cv2.flip(img, 1)  # Flip vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y : y + h, x : x + w])

            # If confidence is less them 100 ==> "0" : perfect match
            name = names.get(str(id), "unknown") if confidence < 100 else "unknown"

            if confidence < 100:
                sound = get_alarm_sound()
                try:
                    sound.play()
                except Exception as e:
                    print("\n [ERROR] error while playing sound: ", e)

            confidence = " {:.2f}".format(confidence)
            cv2.putText(img, str(name), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(
                img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1
            )

        cv2.imshow("camera", img)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:  # press 'ESC' to quit
            break

    cam.release()
    cv2.destroyAllWindows()
