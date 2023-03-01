import numpy
from pygame import mixer
from keras.models import load_model
from time import sleep
from tensorflow.keras.utils import img_to_array
#from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
from parameters import *
from scipy.spatial import distance
from imutils import face_utils as face
from pygame import mixer
import dlib
import time
import cv2
from tkinter import *
import tkinter.messagebox

root = Tk()
root.geometry('500x570')
frame = Frame(root, relief=RIDGE, borderwidth=2)
frame.pack(fill=BOTH, expand=1)
root.title('Driver Cam')
frame.config(background='light blue')
label = Label(frame, text="Driver Cam", bg='light blue', font=('Times 35 bold'))
label.pack(side=TOP)
filename = PhotoImage(
    file=r"E:\Project work\new drowisiness\demo.png")
background_label = Label(frame, image=filename)
background_label.pack(side=TOP)
lower = np.array([0, 70, 30], dtype="uint8")
upper = np.array([50, 255, 255], dtype="uint8")
first_time = 1


def hel():
    help(cv2)


def Contri():
    tkinter.messagebox.showinfo("Contributors",
                                "\n1.Vaibhav Lokhande\n2. Prathamesh Patil\n3. Parth Patil \n4. Saurabh Gaikwad\n")


def anotherWin():
    tkinter.messagebox.showinfo("About",
                                'Driver Cam version v1.0\n Made Using\n-OpenCV\n-Numpy\n-Tkinter\n In Python 3')


menu = Menu(root)
root.config(menu=menu)

subm1 = Menu(menu)
menu.add_cascade(label="Tools", menu=subm1)
subm1.add_command(label="Open CV Docs", command=hel)

subm2 = Menu(menu)
menu.add_cascade(label="About", menu=subm2)
subm2.add_command(label="Driver Cam", command=anotherWin)
subm2.add_command(label="Contributors", command=Contri)


def exitt():
    exit()


def webdetRec():
    face_classifier = cv2.CascadeClassifier(
        r'C:\Users\DELL\Downloads\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\haarcascade_frontalface_default.xml')
    classifier = load_model(
        r'C:\Users\DELL\Downloads\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\model.h5')

    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                label_position = (x, y)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Emotion Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def get_max_area_rect(rects):
    if len(rects) == 0: return
    areas = []
    for rect in rects:
        areas.append(rect.area())
    return rects[areas.index(max(areas))]


def get_eye_aspect_ratio(eye):
    vertical_1 = distance.euclidean(eye[1], eye[5])
    vertical_2 = distance.euclidean(eye[2], eye[4])
    horizontal = distance.euclidean(eye[0], eye[3])
    return (vertical_1 + vertical_2) / (horizontal * 2)  # aspect ratio of eye


def get_mouth_aspect_ratio(mouth):
    horizontal = distance.euclidean(mouth[0], mouth[4])
    vertical = 0
    for coord in range(1, 4):
        vertical += distance.euclidean(mouth[coord], mouth[8 - coord])
    return vertical / (horizontal * 3)  # mouth aspect ratio


def blink():
    capture =cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    blink_cascade = cv2.CascadeClassifier('CustomBlinkCascade.xml')

    mixer.init()
    distracton_initlized = False
    eye_initialized = False
    mouth_initialized = False

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        r"E:\Project work\new drowisiness\shape_predictor_68_face_landmarks.dat")

    ls, le = face.FACIAL_LANDMARKS_IDXS["left_eye"]
    rs, re = face.FACIAL_LANDMARKS_IDXS["right_eye"]

    cap = cv2.VideoCapture(0)

    fps_couter = 0
    fps_to_display = 'initializing...'
    fps_timer = time.time()
    while True:
        _, frame = cap.read()
        fps_couter += 1
        frame = cv2.flip(frame, 1)
        if time.time() - fps_timer >= 1.0:
            fps_to_display = fps_couter
            fps_timer = time.time()
            fps_couter = 0
        cv2.putText(frame, "FPS :" + str(fps_to_display), (frame.shape[1] - 100, frame.shape[0] - 10), \
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # frame = imutils.resize(frame, width=900)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)
        rect = get_max_area_rect(rects)

        if rect != None:

            distracton_initlized = False

            shape = predictor(gray, rect)
            shape = face.shape_to_np(shape)

            leftEye = shape[ls:le]
            rightEye = shape[rs:re]
            leftEAR = get_eye_aspect_ratio(leftEye)
            rightEAR = get_eye_aspect_ratio(rightEye)

            inner_lips = shape[60:68]
            mar = get_mouth_aspect_ratio(inner_lips)

            eye_aspect_ratio = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (255, 255, 255), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (255, 255, 255), 1)
            lipHull = cv2.convexHull(inner_lips)
            cv2.drawContours(frame, [lipHull], -1, (255, 255, 255), 1)

            cv2.putText(frame, "EAR: {:.2f} MAR{:.2f}".format(eye_aspect_ratio, mar), (10, frame.shape[0] - 10), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if eye_aspect_ratio < EYE_DROWSINESS_THRESHOLD:

                if not eye_initialized:
                    eye_start_time = time.time()
                    eye_initialized = True

                if time.time() - eye_start_time >= EYE_DROWSINESS_INTERVAL:
                    alarm_type = 0
                    cv2.putText(frame, "YOU ARE SLEEPY...\nPLEASE TAKE A BREAK!", (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    if not distracton_initlized and not mouth_initialized and not mixer.music.get_busy():
                        mixer.music.load(
                            r"F:\Project work\Drowsiness-monitoring-Using-OpenCV-Python-master\data\audio_files\short_horn.mp3")
                        mixer.music.play()
            else:
                eye_initialized = False
                if not distracton_initlized and not mouth_initialized and mixer.music.get_busy():
                    mixer.music.stop()

            if mar > MOUTH_DROWSINESS_THRESHOLD:

                if not mouth_initialized:
                    mouth_start_time = time.time()
                    mouth_initialized = True

                if time.time() - mouth_start_time >= MOUTH_DROWSINESS_INTERVAL:
                    alarm_type = 0
                    cv2.putText(frame, "YOU ARE YAWNING...\nDO YOU NEED A BREAK?", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    if not mixer.music.get_busy():
                        mixer.music.load(
                            r"F:\Project work\Drowsiness-monitoring-Using-OpenCV-Python-master\data\audio_files\short_horn.mp3")
                        mixer.music.play()
            else:
                mouth_initialized = False
                if not distracton_initlized and not eye_initialized and mixer.music.get_busy():
                    mixer.music.stop()


        else:

            alarm_type = 1
            if not distracton_initlized:
                distracton_start_time = time.time()
                distracton_initlized = True

            if time.time() - distracton_start_time > DISTRACTION_INTERVAL:

                cv2.putText(frame, "EYES ON ROAD", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                if not eye_initialized and not mouth_initialized and not mixer.music.get_busy():
                    mixer.music.load(
                        r"F:\Project work\Drowsiness-monitoring-Using-OpenCV-Python-master\data\audio_files\long_horn.mp3")
                    mixer.music.play()

        cv2.imshow("Drowsiness Detection   ", frame)
        key = cv2.waitKey(5) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__=='__main__':
    blink()

but4 = Button(frame, padx=5, pady=5, width=39, bg='white', fg='black', relief=GROOVE, command=webdetRec,
              text='Record Emotions', font=('helvetica 15 bold'))
but4.place(x=5, y=250)

but5 = Button(frame, padx=5, pady=5, width=39, bg='white', fg='black', relief=GROOVE, command=blink,
              text='Detect Drowsiness With Sound', font=('helvetica 15 bold'))
but5.place(x=5, y=350)

but5 = Button(frame, padx=5, pady=5, width=5, bg='white', fg='black', relief=GROOVE, text='EXIT', command=exitt,
              font=('helvetica 15 bold'))
but5.place(x=210, y=478)

root.mainloop()
