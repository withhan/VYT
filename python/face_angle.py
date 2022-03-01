import cv2
import mediapipe as mp
import math
import time
import collections

import pickle
import copy

import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

from trainning.utility.custom_face_mash_tesselation import FACEMESH_TESSELATION as custom_tesslation
# option
places_of_decimals = 1
filter_rate = 5
image_save = True
image_save_path = './image/'
# var
standard_face_size = [None, None]
standard_solid = []
standard_face_size_part = [None, None]  # for eyes and lips
standard_eyes = []
standard_mouth = []

offset = [0, 0, 0]
direction_offset = [0]  # pitch offset
last_time = 0
sense_time = 1/20
sense_que = [collections.deque() for i in range(3)]

base_keypoint = [33, 263, 175]

def get_point_length(p1, p2):
    result = []
    p1 = results.multi_face_landmarks[0].landmark[p1]
    p2 = results.multi_face_landmarks[0].landmark[p2]
    for i in ['x', 'y', 'z']:
        result.append(abs(getattr(p1, i) - getattr(p2, i)))
    return result

def get_diagonal_length(p1, p2, d3=False):
    buffer = get_point_length(p1, p2)
    return sum(math.pow(i, 2) for i in (buffer if d3 else buffer[:2]))

# get_face_solid
def get_face_solid(element_point=base_keypoint):
    result = [0, 0, 0, 0]
    buffer = results.multi_face_landmarks[0].landmark
    for n, i in enumerate(['x', 'y']):
        result[n+2] = abs(getattr(buffer[element_point[0]], i) - getattr(buffer[element_point[1]], i))
        result[0] += pow(result[n+2], 2)
    for i in ['x', 'y']:
        result[1] += pow(abs((getattr(buffer[element_point[0]], i) + getattr(buffer[element_point[1]], i))/2
                             - getattr(buffer[element_point[2]], i)), 2)
    return result

def get_face_size():
    return [results_detect.detections[0].location_data.relative_bounding_box.width,
                                      results_detect.detections[0].location_data.relative_bounding_box.height]

def get_joint_angle(sx, sy, l):
    result = []
    result.append(math.degrees(math.asin(sy/l)))
    result.append(math.degrees(math.asin(
        sx/math.sqrt(math.pow(l, 2)-math.pow(sy, 2))
    )))
    return result

def get_joint2_angle(distance_set):
    return (math.atan2(distance_set[2], distance_set[0]), math.atan2(distance_set[2], distance_set[1]))

def get_angle(t, c):
    return math.acos(t/c)

def get_face_angle(target, standard, target_size, standard_size):
    size = max(target_size[0]/standard_size[0], target_size[1]/standard_size[1])
    buffer = [
        (math.pow(target[2], 2) / (standard[0]*size - math.pow(target[3], 2))),
        (math.pow(target[3], 2) / (standard[1]*size - math.pow(target[2], 2))),
              ]
    face_vertical = [0, 0, 0]
    buffer_landmark = results.multi_face_landmarks[0].landmark
    for n, i in enumerate(['x', 'y', 'z']):
        face_vertical[n] = abs(
            (getattr(buffer_landmark[base_keypoint[0]], i) + getattr(buffer_landmark[base_keypoint[1]], i)) / 2
            - getattr(buffer_landmark[base_keypoint[2]], i)
        )
    return (

    math.acos(buffer[0]) if 0 < buffer[0] < 1 else 0,  # right-left
    get_joint2_angle(face_vertical[:3])[1],  # up-down
    math.atan2(target[3], target[2])  # rotate
    )

def get_face_pitch(element_point=base_keypoint):
    face_vertical = [0, 0, 0]
    buffer = results.multi_face_landmarks[0].landmark
    for n, i in enumerate(['x', 'y', 'z']):
        face_vertical[n] = abs(
            (getattr(buffer[element_point[0]], i) + getattr(buffer[element_point[1]], i)) / 2
            - getattr(buffer[element_point[2]], i)
        )
    return get_joint2_angle(face_vertical[:3])[1]

def get_point_angle(p1, p2):
    buffer = get_point_length(p1, p2)
    return get_joint2_angle(buffer)

def get_face_direction():
    element_point = base_keypoint
    return (
        True if (results.multi_face_landmarks[0].landmark[element_point[0]].z >
                 results.multi_face_landmarks[0].landmark[element_point[1]].z) else False,
        True if (sum(results.multi_face_landmarks[0].landmark[element_point[j]].z for j in range(2))/2 >
                 results.multi_face_landmarks[0].landmark[element_point[2]].z + direction_offset[0]) else False,
        True if (results.multi_face_landmarks[0].landmark[element_point[0]].y >
                 results.multi_face_landmarks[0].landmark[element_point[1]].y) else False,
    )

def get_part_length(target, standard, target_size, standard_size):
    size = max(target_size[0] / standard_size[0], target_size[1] / standard_size[1])
    buffer = [0 for i in range(4)]
    for i in range(2):
        for j in range(2):
            buffer[i*2+j] = target[i][j]/standard[i][j]*size
    return buffer

def get_eyes():
    return(get_diagonal_length(374, 386, True),  # left eye
    get_diagonal_length(145, 159, True))  # right eye

def get_mouth():
    return(get_diagonal_length(0, 14, True),  # height
    get_diagonal_length(61, 291, True))  # width

def que_filter(i, v=None, n=1):
    global sense_que
    sense_que[i].append(v)
    while len(sense_que[i]) > n:
        sense_que[i].popleft()
    return sum(sense_que[i])/len(sense_que[i])

onece = True

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(1)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh, mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        if onece:
            onece = False
            iheight, iwidth, ch = image.shape
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image_save:
            image_original = copy.deepcopy(image)
        results = face_mesh.process(image)
        results_detect = face_detection.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=custom_tesslation,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .DrawingSpec(color=(255, 255, 128), thickness=1))
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_iris_connections_style())
            if results_detect.detections and time.time()-last_time>sense_time:
                last_time = time.time()
                # image export work for emotion detact
                if image_save:
                    face = results_detect.detections[0].location_data.relative_bounding_box
                    (x, y, w, h) = face.xmin, face.ymin, face.width, face.height
                    x = x * iwidth
                    w = w * iwidth
                    y = y * iheight
                    h = h * iheight
                    roi = image_original[round(y):round(y + h), round(x):round(x + w)]
                    try:
                        with open(image_save_path + '/fdata.pickle', "wb") as fw:
                            pickle.dump(cv2.resize(roi, (48, 48)), fw)
                    except FileExistsError:
                        pass
                    except cv2.error:
                        pass
                # image export work end
                sense_result = [0 for i in range(7)]
                if standard_face_size_part[0]:
                    buffer = [get_face_size(), get_eyes(), get_mouth()]
                    for n, i in enumerate(get_part_length([get_eyes(), get_mouth()], [standard_eyes, standard_mouth],
                                             get_face_size(), standard_face_size_part)):
                        sense_result[3+n] = i
                if standard_face_size[0]:
                    cal_angle = get_face_angle(get_face_solid(), standard_solid, get_face_size(), standard_face_size)
                    size = max(get_face_size()[0] / standard_face_size[0], get_face_size()[1] / standard_face_size[1])

                temp = get_point_angle(base_keypoint[0], base_keypoint[1])[0]
                dir = get_face_direction()
                sense_result[0] = (1 if dir[0] else -1) * (cal_angle[0] if (
                        standard_face_size[0] and abs(1-size) < 0.15) else temp)
                sense_result[1] = (1 if not dir[1] else -1) * get_face_pitch()
                face_solid = get_face_solid()
                sense_result[2] = (1 if not dir[2] else -1) * math.atan2(face_solid[3], face_solid[2])
                # value filtering
                for i in range(3):
                    sense_result[i] = que_filter(i, math.degrees(sense_result[i]), filter_rate)

                # 증폭설정
                sense_result[0] = sense_result[0]  # *0.7
                sense_result[1] = sense_result[1]  # *0.4
                # 증폭설정 끝

                for i in range(4):
                    sense_result[3+i] = sense_result[3+i]*100
                # send string
                send_string = 'A#'+'/'.join(map(str, map((lambda a: int(round(a, places_of_decimals)*10)), sense_result)))
                print(send_string)
                sock.sendto(send_string.encode(), ('192.168.0.8', 8003))
        else:
            for i in sense_que:
                i.clear()
        cv2.imshow('face angle [filter: {}]'.format(filter_rate), cv2.flip(image, 1))

        req = cv2.waitKey(5) & 0xFF
        if req == 27:
            break
        elif req == 13:
            if results_detect.detections and results.multi_face_landmarks:
                print(results_detect.detections[0].location_data.relative_bounding_box.width)
                print(results_detect.detections[0].location_data.relative_bounding_box.height)
                standard_face_size = get_face_size()

                standard_solid = get_face_solid()[:2]

                direction_offset[0] = (
                        sum(results.multi_face_landmarks[0].landmark[base_keypoint[j]].z for j in range(2)) / 2 -
                results.multi_face_landmarks[0].landmark[base_keypoint[2]].z)
            else:
                print('init fail: face was not detecting')

        elif req == 32:  #space - face part init
            if results_detect.detections and results.multi_face_landmarks:
                standard_face_size_part = get_face_size()
                standard_eyes = get_eyes()
                standard_mouth = get_mouth()
            else:
                print('init fail: face was not detecting')
        elif req == 127:  # DEL - precise mode clear
            standard_face_size = [None, None]
        elif req == ord('u'):
            filter_rate += 1
        elif req == ord('d'):
            filter_rate -= 1
cap.release()
