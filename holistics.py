from typing import Annotated
import cv2 as cv
import mediapipe as mp

mpDrawing = mp.solutions.drawing_utils
mpHolistic = mp.solutions.holistic
holistic = mpHolistic.Holistic(static_image_mode=True)

image = cv.imread('data\pose.jpg')
# Convert the BGR image to RGB before processing
result = holistic.process(cv.cvtColor(image,cv.COLOR_BGR2RGB))
# Draw pose, left and right hands, and face landmarks on the image.
annotated =image.copy()
mpDrawing.draw_landmarks(annotated,result.face_landmarks,mpHolistic.FACE_CONNECTIONS)
mpDrawing.draw_landmarks(annotated,result.left_hand_landmarks,mpHolistic.HAND_CONNECTIONS)
mpDrawing.draw_landmarks(annotated,result.right_hand_landmarks,mpHolistic.HAND_CONNECTIONS)
mpDrawing.draw_landmarks(annotated,result.pose_landmarks,mpHolistic.POSE_CONNECTIONS)

cv.imshow('holistic', annotated)
cv.waitKey(0)