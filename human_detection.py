import numpy as np
import cv2

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

video = cv2.VideoCapture(1)

while True:
    ret,frame = video.read()

    if not ret:
        print("Image not captured properly")
        break
    frame = cv2.resize(frame, (640, 480))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    boxes, _ = hog.detectMultiScale(frame, winStride=(8, 8))

    boxes = np.array([[x,y,x+w,y+h] for (x,y,w,h) in boxes])

    for (xA, yA, xB, yB) in boxes:
        cv2.rectangle(frame, (xA,yA), (xA, yB), (0,255,0), 3)
    
    cv2.imshow('Human_Detection',frame)
    if cv2.waitKey(1) == ord('q'):
        break
video.release()
cv2.destroyAllWindows()

