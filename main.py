from asyncore import read
from cv2 import VideoCapture, waitKey
import numpy as np
import imutils
import time
import cv2
import os
import math
import io
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import image

from itertools import chain 
from constants import *

LABELS = open(YOLO_LABELS_PATH).read().strip().split('\n')

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype='uint8')

print('Loading YOLO from disk...')

# TINY YOLO
# neural_net = cv2.dnn.readNetFromDarknet(YOLOV_CFG_PATH, YOLOV_WEIGHTS_PATH)

# YOLOV2
# neural_net = cv2.dnn.readNetFromDarknet(YOLOV2_CFG_PATH, YOLOV2_WEIGHTS_PATH)

# YOLOV3
neural_net = cv2.dnn.readNetFromDarknet(YOLOV3_CFG_PATH, YOLOV3_WEIGHTS_PATH)

# YOLOV4
# neural_net = cv2.dnn.readNetFromDarknet(YOLOV4_CFG_PATH, YOLOV4_WEIGHTS_PATH)
layer_names = neural_net.getLayerNames()
layer_names = [layer_names[i - 1] for i in neural_net.getUnconnectedOutLayers()]

vs = cv2.VideoCapture(VIDEO_PATH)

# while(True):
#     ret, frame = vs.read()
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
    

writer = None
(W, H) = (None, None)

# try:
#     if(imutils.is_cv2()):
#         prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT
#     else:
#         prop = cv2.CAP_PROP_FRAME_COUNT
#     total = int(vs.get(prop))
#     print('Total frames detected are: ', total)
# except Exception as e:
#     print(e)
#     total = -1

h = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))

# b_image = np.zeros((h, w, 3), np.uint8)
# cv2.imwrite('./output/img.png', b_image)


# size = (w, h)
# result = cv2.VideoWriter('./output/birdeye.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, size, True)
st = time.time()
while True:
    # print("Hey")
    (grabbed, frame) = vs.read()
    cv2.imshow('frame', frame)
    plt.axes(xlim = (0, 1280), ylim = (0, 720))

    # image = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)
    # cv2.imshow('Bird-eye', image)

    # data = image.imread('./output/img.png')

    # plt.plot(200, 350, marker='v', color='black')

    # im = Image.open(img_buf)
    # im.show(title = "bleh")



    if not grabbed:
        break
    
    if W is None or H is None:
        H, W = (frame.shape[0], frame.shape[1])

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    neural_net.setInput(blob)

    start_time = time.time()
    layer_outputs = neural_net.forward(layer_names)
    end_time = time.time()
    
    boxes = []
    confidences = []
    classIDs = []
    lines = []
    box_centers = []

    for output in layer_outputs:
        for detection in output:
            
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            if confidence > 0.5 and classID == 0:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype('int')
                
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                
                box_centers = [centerX, centerY]

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    
    if len(idxs) > 0:
        unsafe = []
        count = 0
        
        for i in idxs.flatten():
            
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            centeriX = boxes[i][0] + (boxes[i][2] // 2)
            centeriY = boxes[i][1] + (boxes[i][3] // 2)
            # plt.plot(centeriX, centeriX, marker = 'x', color="black")
            
            #  cv2.drawMarker(image, (centeriX, centeriY), (255, 0, 0), cv2.MARKER_CROSS, 10, 2)

            # print(centeriX , centeriY)
            color = [int(c) for c in COLORS[classIDs[i]]]
            text = '{}: {:.4f}'.format(LABELS[classIDs[i]], confidences[i])

            idxs_copy = list(idxs.flatten())
            idxs_copy.remove(i)

            for j in np.array(idxs_copy):
                centerjX = boxes[j][0] + (boxes[j][2] // 2)
                centerjY = boxes[j][1] + (boxes[j][3] // 2)
                distance = math.sqrt(math.pow(centerjX - centeriX, 2) + math.pow(centerjY - centeriY, 2))

                if distance <= SAFE_DISTANCE:
                    cv2.line(frame, (boxes[i][0] + (boxes[i][2] // 2), boxes[i][1]  + (boxes[i][3] // 2)), (boxes[j][0] + (boxes[j][2] // 2), boxes[j][1] + (boxes[j][3] // 2)), (0, 0, 255), 2)
                    unsafe.append([centerjX, centerjY])
                    unsafe.append([centeriX, centeriY])

            if centeriX in chain(*unsafe) and centeriY in chain(*unsafe):
                count += 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(frame, (50, 50), (450, 90), (0, 0, 0), -1)
            cv2.putText(frame , text , (centeriX , centeriY) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , color , 2)
            cv2.putText(frame, 'No. of people unsafe: {}'.format(count), (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 3)

    # img_buf = io.BytesIO()
    # plt.savefig(img_buf, format='png')
    # im = Image.open(img_buf)
    # im.show()
    # plt.clf()
   
        
    if writer is None:

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, 30,(frame.shape[1], frame.shape[0]), True)

        # if total > 0
        # st = time.time()
        # elap = (end_time - start_time)
        # print('Single frame took {:.4f} seconds'.format(elap))
        # print('Estimated total time to finish: {:.4f} seconds'.format(elap * total))

    writer.write(frame)

    if (waitKey(10) == 27):
        print("ESC key pressed. Stopping the video.")
        break


    # result.write(image)  

et = time.time()
print("This took %.2f seconds" % (et - st))

# print("Total time taken")
print('Cleaning up...')
# result.release()
writer.release()
vs.release()     