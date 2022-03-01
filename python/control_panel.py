import tkinter
import socket

places_of_decimals = 1

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

window = tkinter.Tk()

window.title('control')

def change(self=None):
    msg_data = ['' for i in range(13)]
    for n, i in enumerate(var_bool):
        if i.get():
            msg_data[n] = int(round(var_set[n].get(), places_of_decimals) * 10)
        elif 3 <= n and n < 8:
            msg_data[n] = -1
    msg = 'B#'+'/'.join(map(str, msg_data))
    print(msg)
    sock.sendto(msg.encode(), ('192.168.0.8', 8003))

def emotion_all_activate():
    global var_bool
    if auto_emotion_activate.get():
        for i in range(8, 13):
            var_bool[i].set(1)
    else:
        for i in range(8, 13):
            var_bool[i].set(0)
    change()

var_set = []
for i in range(13):
    var_set.append(tkinter.IntVar())

var_bool = []
for i in range(13):
    var_bool.append(tkinter.IntVar())

for i in range(5):
    var_set[8+i].set(100)
    var_bool[8+i].set(True)

auto_emotion_activate = tkinter.BooleanVar(value=True)

face_box = []
face_scale = []
for n, i in enumerate(['x', 'y', 'z']):
    face_box.append(tkinter.Checkbutton(window, text=i, variable=var_bool[n], command=change, width=10))
    face_scale.append(tkinter.Scale(window, variable=var_set[n], orient='horizontal', from_=-90, to=90, command=change,
                                    repeatdelay=300, repeatinterval=400, length=200, width=25))

tkinter.Scale()

emotion_box = []
emotion_scale = []
for n, i in enumerate(['Angry', 'Fun', 'Joy', 'Sorrow', 'Surprised']):
    emotion_box.append(tkinter.Checkbutton(window, text=i, variable=var_bool[3+n], command=change, width=10))
    emotion_scale.append(tkinter.Scale(window, variable=var_set[3+n], orient='horizontal', command=change,
                                    repeatdelay=300, repeatinterval=400, length=200, width=25))
auto_emotion_box = []
auto_emotion_scale = []
for n, i in enumerate(['Angry', 'Fun', 'Joy', 'Sorrow', 'Surprised']):
    auto_emotion_box.append(tkinter.Checkbutton(window, text=i, variable=var_bool[8+n], command=change, width=10))
    auto_emotion_scale.append(tkinter.Scale(window, variable=var_set[8+n], orient='horizontal', command=change,
                                    repeatdelay=300, repeatinterval=400, length=200, width=25))

for i in range(3):
    face_box[i].grid(row=i, column=0)
    face_scale[i].grid(row=i, column=1)

for i in range(5):
    emotion_box[i].grid(row=3+i, column=0)
    emotion_scale[i].grid(row=3+i, column=1)

for i in range(5):
    auto_emotion_box[i].grid(row=8+i, column=0)
    auto_emotion_scale[i].grid(row=8+i, column=1)

auto_emotion_activate_btn = tkinter.Checkbutton(
    window, text="auto activate", variable=auto_emotion_activate, command=emotion_all_activate, width=10)
auto_emotion_activate_btn.grid(row=13, column=1)

window.mainloop()
