import cv2
import dlib

cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier(r'D:\Program\ProgramData\Anaconda3\envs\tensorflow\Lib\site-packages\cv2\data'
                                    r'\haarcascade_frontalface_default.xml')
while cap.isOpened():
    ret_flag, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 人脸检测
    faces = faceCascade.detectMultiScale(
        gray,  # 灰度图，使用灰度图会提高效率
        scaleFactor=1.05,  # 图片缩小的值
        minNeighbors=20,  # 判断次数
        minSize=(72, 72)  # 人脸最小尺寸
    )
    for (x, y, w, h) in faces:
        # 画脸的方框
        # 图像源，原点，终点，线的颜色，粗细
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, "face number:" + str(len(faces)), (10, 30), font, 1.0, (0, 255, 255), 1)
    cv2.imshow("Face Detect", frame)
    k = cv2.waitKey(10)
    if k == 27:
        break
    #保存人脸
    # if k == ord('s'):

    # if k == ord('d'): 人脸识别
cap.release()
cv2.destroyAllWindows()
