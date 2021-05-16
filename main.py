import tensorflow as tf
import cv2
import dlib
import numpy as np
import face

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cap.set(3, 480)
    # height = cap.get(cv2.CAP_PROP_XI_HEIGHT)
    # width = cap.get(cv2.CAP_PROP_XI_WIDTH)
    while cap.isOpened():
        ret_flag, frame = cap.read()
        detector = dlib.get_frontal_face_detector()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.putText(frame, "faces: " + str(len(faces)), (20, 50), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow("Face Detect", cv2.flip(frame, 1))
        k = cv2.waitKey(1)
        if k == ord('q') or k == 27:
            break
        # if k == ord('s'): 保存人脸
        # if k == ord('d'): 人脸识别
    cap.release()
    cv2.destroyAllWindows()
