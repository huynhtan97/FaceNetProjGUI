#!/usr/bin/env python
import cv2
import os
import facenet_recognition
import shutil
import sys
from matplotlib import pyplot as plt
from io import StringIO
import numpy as np
import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        faces = face_cascade.detectMultiScale(image, 1.3, 5)

        # Display text to image in opencv if no face detected - publish mqtt message
        noFaceDetected = False
        if (len(faces) != 1):
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, 'ADJUST FACE PLEASE', (20, 250), font, 1, (0, 0, 255), 3, cv2.LINE_AA)
            noFaceDetected = True
            if(noFaceDetected == True):
                publish.single(topic='ledStatus', payload='Off', hostname='broker.hivemq.com', protocol=mqtt.MQTTv31)

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        ret, jpeg = cv2.imencode('.jpg', image)

        return jpeg.tobytes()
