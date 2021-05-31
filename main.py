import time

import pyvirtualcam
import numpy as np
import cv2
import mediapipe as mp
import PongGameModule as pong

cap = cv2.VideoCapture(0)  # capture video from the webcam

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Show the frames per second
cTime = 0
pTime = 0



def send_frames(cam, frame):
    cam.send(frame)
    cam.sleep_until_next_frame()


def get_image_from_webcam():
    ret, frame_bgr = cap.read()  # Get the image frames from the webcam
    return frame_bgr


if cap.isOpened():
    webcamWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get the dimensions of the webcam video
    webcamHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    game = pong.PongGame(webcamWidth, webcamHeight , 10 , np.array([int(webcamWidth/2), int(webcamHeight/2)]))

    vCam = pyvirtualcam.Camera(width=webcamWidth, height=webcamHeight, fps=30)  # Start the virtual Camera
    print(f'Using virtual camera: {vCam.device}')

    while True:
        frame_bgr = get_image_from_webcam()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        game.move_ball()

        handResults = hands.process(frame_rgb)
        if handResults.multi_hand_landmarks:
            for handLms in handResults.multi_hand_landmarks:
                # mpDraw.draw_landmarks(frame_bgr, handLms, mpHands.HAND_CONNECTIONS)
                # mpDraw.draw_landmarks(frame_rgb, handLms, mpHands.HAND_CONNECTIONS)

                for id, lm in enumerate(handLms.landmark):
                    cx, cy = int(lm.x*webcamWidth), int(lm.y*webcamHeight)
                    if id == 8:
                        game.check_collision(np.array([cx,cy]))
                        cv2.circle(frame_rgb, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                        cv2.circle(frame_bgr, (cx, cy), 15, (255, 0, 255), cv2.FILLED)


        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(frame_bgr, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.circle(frame_bgr, (game.pos[0], game.pos[1]), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(frame_rgb, (game.pos[0], game.pos[1]), 15, (255, 0, 255), cv2.FILLED)

        cv2.imshow("hands detection" , frame_bgr)
        send_frames(vCam, frame_rgb)



        c = cv2.waitKey(1)
        if c == 27:
            break


# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")


