
import cv2 as cv
import mediapipe as mp

def Rescale(frame, scale=0.75):
    # FOR PICTURES,VIDEO,LIVE
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dim = (width, height)
    return cv.resize(frame, dim, interpolation=cv.INTER_AREA)

mpDrawing = mp.solutions.drawing_utils
mpHolistic = mp.solutions.holistic
landmarks = mpDrawing.DrawingSpec(color= (200,220,140),thickness=1, circle_radius=1)
connect = mpDrawing.DrawingSpec(color=(255, 0, 255),thickness=1, circle_radius=2)
poseConnect = mpDrawing.DrawingSpec(color=(254,255,225),thickness=2, circle_radius=2)
holistic = mpHolistic.Holistic(static_image_mode=True)

image = cv.imread('data\pose.jpg')
# Convert the BGR image to RGB before processing
result = holistic.process(cv.cvtColor(image,cv.COLOR_BGR2RGB))
# Draw pose, left and right hands, and face landmarks on the image.
annotated =image.copy()
mpDrawing.draw_landmarks(annotated,result.face_landmarks,mpHolistic.FACE_CONNECTIONS,landmarks,connect)
mpDrawing.draw_landmarks(annotated,result.left_hand_landmarks,mpHolistic.HAND_CONNECTIONS,landmarks,connect)
mpDrawing.draw_landmarks(annotated,result.right_hand_landmarks,mpHolistic.HAND_CONNECTIONS,landmarks,connect)
mpDrawing.draw_landmarks(annotated,result.pose_landmarks,mpHolistic.POSE_CONNECTIONS,landmarks,poseConnect)

cv.imshow('holistic', annotated)
cv.waitKey(0)