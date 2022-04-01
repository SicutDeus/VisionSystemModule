"""
import cv2

print(cv2.__version__)

capture = cv2.VideoCapture(0)
i = 0

while True:
    ret, img = capture.read()
    cv2.imshow('From Camera', img)
    print(img.shape)

    cv2.imshow('Color red', img[:, :, 1])
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img3 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('Transform1', img2)
    cv2.imshow('Transform2', img3)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break
    # if i%10 == 0:
        # cv2.imwrite('P:\\PyCharmProjects\\pythonProject\\Images\\' + str(i) + '.jpg', img)

    i = i+1

capture.release()
cv2.destroyAllWindows()
"""

import cv2
import time

CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class_names = []
with open("Example/Classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

vc = cv2.VideoCapture(0)

net = cv2.dnn.readNet("Example/yolov4.weights", "Example/yolov4.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)

while cv2.waitKey(1) < 1:
    (grabbed, frame) = vc.read()
    if not grabbed:
        exit()

    start = time.time()
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    end = time.time()

    start_drawing = time.time()
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_names[classid], score)
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    end_drawing = time.time()

    fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (
    1 / (end - start), (end_drawing - start_drawing) * 1000)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow("detections", frame)