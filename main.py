import cv2
import tools

cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier(r'D:\Program\ProgramData\Anaconda3\envs\tensorflow\Lib\site-packages\cv2\data'
                                    r'\haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_DUPLEX
is_save_face = bool(input("是否需要保存图片："))
path = "face_images/"
origin_count, total_count = 0, 0
if is_save_face:
    face_id = int(input("人脸编号："))
    face_count = int(input("人脸数量："))
    path = path + str(face_id)
    origin_count = tools.mkdir(path)
    total_count = origin_count + face_count

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
        if is_save_face:
            if len(faces) > 1:
                cv2.putText(frame, "人脸数量大于1，无法保存" + str(len(faces)), (10, 80), font, 1.0, (0, 255, 255), 1)
            else:
                fac_gray = gray[y: (y + h), x: (x + w)]
                origin_count += 1
                cv2.imwrite(path + '/' + str(origin_count) + '.jpg', fac_gray)

    cv2.putText(frame, "人脸数量" + str(len(faces)), (10, 30), font, 1.0, (0, 255, 255), 1)
    cv2.imshow("人脸识别", frame)
    k = cv2.waitKey(10)
    if origin_count >= total_count:
        if origin_count == total_count:
            print("图片保存完成")
        is_save_face = False
    if k == 27:
        break
    # if k == ord('d'): 人脸识别
cap.release()
cv2.destroyAllWindows()
