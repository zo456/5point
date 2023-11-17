import os 
import cv2
import numpy as np
import dlib

from natsort import natsorted

PREDICTOR_PATH = r"shape_predictor_5_face_landmarks.dat"

vid_path = os.listdir('./input/')[0]

predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()

def FrameCap(path):
    if not os.path.exists(f'./frames'):
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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = detector(gray, 1)
    for result in face:
        x = result.left()
        y = result.top()
        x1 = result.right()
        y1 = result.bottom()
        #cv2.rectangle(img, (x,y), (x1,y1), (0,255,255), 2)
    rect = dlib.rectangle(int(x), int(y), int(x1), int(y1))
    landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rect).parts()])
    return landmarks, img

FrameCap('./input' + os.sep + vid_path)
frame_list = natsorted(os.listdir("./frames/"))

base = fivePoints('./frames' + os.sep + frame_list[0])[0]
outer_base = np.float32(base[[0, 2, 4]])
inner_base = np.float32(base[[1, 3, 4]])

if not os.path.exists('./aligned_outer/'):
    os.makedirs('aligned_outer')
if not os.path.exists('./aligned_inner/'):
    os.makedirs('aligned_inner')
for item in frame_list:
    try:
        landmarks = fivePoints('./frames/' + os.sep + item)
        img = landmarks[1]
        landmarks = landmarks[0]
        outer = np.float32(landmarks[[0, 2, 4]])
        inner = np.float32(landmarks[[1, 3, 4]])
        M_out = cv2.getAffineTransform(outer, outer_base)
        M_in = cv2.getAffineTransform(inner, inner_base)
        rows, cols, ch = img.shape
        outer_img = cv2.warpAffine(img, M_out, (cols, rows))
        inner_img = cv2.warpAffine(img, M_in, (cols, rows))
        cv2.imwrite('./aligned_outer' + os.sep + item, outer_img)
        cv2.imwrite('./aligned_inner' + os.sep + item, inner_img)
    except:
        print(item)

out_align = []
for filename in natsort.natsorted(os.listdir('./aligned_outer/')):
    img = cv2.imread('./aligned_outer' + os.sep + filename)
    out_align.append(img)

out = cv2.VideoWriter('aligned_outer.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (cols, rows))
for i in range(len(out_align)):
    out.write(out_align[i])
    
out.release()

in_align = []
for filename in natsort.natsorted(os.listdir('./aligned_inner/')):
    img = cv2.imread('./aligned_inner' + os.sep + filename)
    in_align.append(img)

out = cv2.VideoWriter('aligned_inner.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (cols, rows))
for i in range(len(in_align)):
    out.write(in_align[i])
    
out.release()