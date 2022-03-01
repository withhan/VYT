import socket
import time

# send speed control
last_time = 0
send_delay = 1/20

# socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# set server ip address and port
sock.bind(('192.168.0.8', 8003))

# data
sensor_addr = [None, None]
client_addr = None

msg_box = [0 for i in range(12)]
fix_box = [None, None, None]
resent_buffer = [0 for i in range(5)]

auto_emotion = -1
emotion_value_box = [1000 for i in range(5)]
emotion_fix = [-1 for i in range(5)]
combo_step_up = 45
combo_step_down = 50

# 클라이언트에서 처리가능한 표정 번호로 변환
emotion_convert = {'0': 1, '1': 4, '2': 0, '3': 3, '4': 3, '5': 3, '6': -1}

while True:
    data, addr = sock.recvfrom(1024)
    data = data.decode()
    if len(data.split('#')) > 1:
        if data.split('#')[0] == 'A':
            msg_box[:7] = data.split('#')[1].split('/')
            if any(fix_box):
                for n, i in enumerate(fix_box):
                    if i:
                        msg_box[n] = i
        elif data.split('#')[0] == 'C':
            temp = data.split('#')[1]
            # {'기쁨': 0, '당황': 1, '분노': 2, '불안': 3, '상처': 4, '슬픔': 5, '중립': 6}
            # 표정 모듈의 출력값을 위의 번호에 맞추어야함
            auto_emotion = emotion_convert[temp]
        elif data.split('#')[0] == 'B':
            temp = data.split('#')[1].split('/')
            if any(temp[:3]):
                for n, i in enumerate(temp[:3]):
                    if i != '':
                        fix_box[n] = i
                        msg_box[n] = i
                    else:
                        fix_box[n] = None
            else:
                fix_box = [None, None, None]
            emotion_fix = list(map(int,map((lambda a: 0 if a == '' else a), temp[3:8])))
            emotion_value_box = list(map(int,map((lambda a: 0 if a == '' else a), temp[8:13])))
        else:
            print('error')
        for i in range(5):
            if emotion_fix[i] != -1:
                if msg_box[7 + i] < emotion_fix[i]:
                    msg_box[7 + i] += combo_step_up
                    if msg_box[7 + i] > emotion_fix[i]:
                        msg_box[7 + i] = emotion_fix[i]
                else:
                    msg_box[7 + i] -= combo_step_down
                    if msg_box[7 + i] < emotion_fix[i]:
                        msg_box[7 + i] = emotion_fix[i]
            elif auto_emotion == i and emotion_value_box[i] != 0:  # 상승
                msg_box[7+i] += combo_step_up
                if msg_box[7+i] > emotion_value_box[i]:
                    msg_box[7 + i] = emotion_value_box[i]
            else:
                if msg_box[7+i] > 0:
                    msg_box[7+i] -= combo_step_down
                    if msg_box[7 + i] < 0:
                        msg_box[7 + i] = 0
    else:
        client_addr = addr
        print('client connected')
    send_data = '/'.join(map(str, map((lambda a: a if a != '' else 0), msg_box)))
    print(send_data)
    if client_addr and time.time() - last_time > send_delay:
        sock.sendto(send_data.encode(), client_addr)
        last_time = time.time()
