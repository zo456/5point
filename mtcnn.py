import os 
import cv2
import numpy as np

from natsort import natsorted
from mtcnn import MTCNN

vid_path = os.listdir('./input/')[0]

def FrameCap(path):
    if not os.path.exists('./frames'):
        os.makedirs('frames')
    vid = cv2.VideoCapture(path)
    cnt = 0
    flag = 1
    while flag:
        flag, image = vid.read()
        if image is not None:
            cv2.imwrite(f'./frames/frame{cnt}.png', image)
            cnt += 1
        else:
            flag = 0

def fivePoints(filename):
    img = cv2.imread(filename)
    detection = detector.detect_faces(img)[0]
    landmarks = detection["keypoints"]
    eye_l = list(landmarks["left_eye"])
    eye_r = list(landmarks["right_eye"])
    mouth_l = list(landmarks["mouth_left"])
    mouth_r = list(landmarks["mouth_right"])
    nose = list(landmarks["nose"])
    return eye_l, eye_r, mouth_l, mouth_r, nose, img



FrameCap('./input' + os.sep + vid_path)

frame_list = natsorted(os.listdir("./frames/"))

detector = MTCNN()

base = fivePoints('./frames' + os.sep + frame_list[0])
base = np.matrix([np.float32(base[0]), np.float32(base[1]), np.float32(base[2])])

if not os.path.exists('./aligned/'):
    os.makedirs('aligned')
for item in frame_list:
    try:
        landmarks = fivePoints('./frames/' + os.sep + item)
        img = landmarks[5]
        landmarks = np.matrix([np.float32(landmarks[0]), np.float32(landmarks[1]), np.float32(landmarks[2])])
        M = cv2.getAffineTransform(landmarks, base)
        rows, cols, ch = img.shape
        new_img = cv2.warpAffine(img, M, (cols, rows))
        cv2.imwrite('./aligned/' + os.sep + item, new_img)
    except:
        print(item)

aligned = []
for filename in natsorted(os.listdir('./aligned/')):
    img = cv2.imread('./aligned' + os.sep + filename)
    aligned.append(img)

out = cv2.VideoWriter('aligned.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (cols, rows))
for i in range(len(aligned)):
    out.write(aligned[i])
    
out.release()