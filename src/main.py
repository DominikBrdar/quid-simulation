
#region IMPORT
import tkinter as tk
from tkinter import ttk

import multiprocessing
from enum import IntEnum
import random
from threading import Timer
import heapq
from copy import deepcopy
import math

from multiprocessing.managers import BaseManager
import uuid
import sys
import os
#endregion


#region GLOBALS

PRINT_DEBUG = False


# listOfQuids = list()
# listOfQuids = multiprocessing.Manager().list()
listOfNeighbours = list()


# INITIAL VALUES ENTERED FROM TKINTER
MAX_QUIDS = multiprocessing.Value("i",200)
MAX_ITER = multiprocessing.Value("i",200)

R_NUM = multiprocessing.Value("i",10)
G_NUM = multiprocessing.Value("i",10)
B_NUM = multiprocessing.Value("i",10)
Y_NUM = multiprocessing.Value("i",10)

ARR_X = multiprocessing.Value("f",200.0)
ARR_Y = multiprocessing.Value("f",200.0)

SIM_SPEED = multiprocessing.Value("i", 100)
TEMPERATURE = multiprocessing.Value("i", 10)



DATA_ENTERED = False
waitOnEnd = False
done = False

phkvadrant1 = 0.0
ph1count = 0
phkvadrant2 = 0.0
ph2count = 0
phkvadrant3 = 0.0
ph3count = 0
phkvadrant4 = 0.0
ph4count = 0
phtotal = 0.0

ph1 = None
ph2 = None
ph3 = None
ph4 = None
phT = None


iter_c = None
max_iter = None
quid_c = None
max_quid = None

end_msg = None

#endregion


#region GRAPHIC



def quit(root: tk.Tk):
    os._exit(0)
    # global paused
    # paused = 1
    # SIM_SPEED.value = 0
    # root.destroy()
    # sys.exit()


def update_speed(val: int):
    SIM_SPEED.value = val

def update_temp(val: int):
    TEMPERATURE.value = val

saved_speed = 0
paused = 0
def play_gl() -> int:
    global paused, saved_speed
    if paused:
        paused = 0
        SIM_SPEED.value = saved_speed
        tick_upr_fun()
        return saved_speed
    else:
        return False

def pause_gl():
    global paused, saved_speed
    if not paused:
        paused = 1
        saved_speed = SIM_SPEED.value
        SIM_SPEED.value = int(1e20)
        return True
    else:
        return False

def is_paused():
    global paused
    return True if paused == 1 else False


def _create_circle(self, x, y, r, **kwargs):
    return self.create_oval(x-r, y-r, x+r, y+r, **kwargs)
tk.Canvas.create_circle = _create_circle


def _create_circle_arc(self, x, y, r, **kwargs):
    if "start" in kwargs and "end" in kwargs:
        kwargs["extent"] = kwargs["end"] - kwargs["start"]
        del kwargs["end"]
    return self.create_arc(x-r, y-r, x+r, y+r, **kwargs)
tk.Canvas.create_circle_arc = _create_circle_arc


def canvas_test_circles(canvas: tk.Canvas):
    canvas.create_circle(100, 120, 50, fill="blue", outline="#DDD", width=4)
    canvas.create_circle_arc(100, 120, 48, fill="green", outline="", start=45, end=140)
    canvas.create_circle_arc(100, 120, 48, fill="green", outline="", start=275, end=305)
    canvas.create_circle_arc(100, 120, 45, style="arc", outline="white", width=6, start=270 - 25, end=270 + 25)
    canvas.create_circle(150, 40, 20, fill="#BBB", outline="")


def waitonend(state):
    global waitOnEnd
    if state == 1:
        waitOnEnd = True
    else:
        waitOnEnd = False


def main_window():
    def increase_speed():
        pz = is_paused()
        if not pz:
            value = int(lbl_value["text"])
            if value < 10000:
                lbl_value["text"] = f"{value + 100}"
                update_speed(value + 100)

    def decrease_speed():
        pz = is_paused()
        if not pz:
            value = int(lbl_value["text"])
            if value > 100:
                lbl_value["text"] = f"{value - 100}"
                update_speed(value - 100)
            if value <= 100:
                # pause()
                pass

    def increase_temp():
        pz = is_paused()
        if not pz:
            value = int(lbl_temp["text"])
            if value < 100:
                lbl_temp["text"] = f"{value + 1}"
                update_temp(value + 1)

    def decrease_temp():
        pz = is_paused()
        if not pz:
            value = int(lbl_temp["text"])
            if value > 0:
                lbl_temp["text"] = f"{value - 1}"
                update_temp(value - 1)

    def play():
        saved_speed = play_gl()
        if saved_speed:
            lbl_value["text"] = f"{saved_speed}"

    def pause():
        succ = pause_gl()
        if succ:
            lbl_value["text"] = f"{0}"




    window_main = tk.Tk()
    window_main.title("Quid's Life")
    window_main.resizable(False, False)

    window_main.rowconfigure(1, minsize=500, weight=1)
    window_main.columnconfigure(0, minsize=500, weight=1)



    graph_canvas = tk.Canvas(window_main, background="lightblue", bd=2, confine=True, relief="groove")
    graph_canvas.config(width=ARR_X.value, height=ARR_Y.value)





    fr_buttons = tk.Frame(window_main)
    btn_open = tk.Button(fr_buttons, text="Play", command=play)
    btn_save = tk.Button(fr_buttons, text="Pause", command=pause)
    btn_quit = tk.Button(fr_buttons, text="Quit", fg="red", command=lambda: quit(window_main))


    btn_open.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
    btn_save.grid(row=0, column=1, sticky="ew", padx=5)
    btn_quit.grid(row=0, column=2, sticky="ew", padx=5)

    fr_buttons.grid(row=0, column=0, sticky="ns")
    graph_canvas.grid(row=1, column=0, sticky="nsew")



    spd = tk.Frame(fr_buttons)

    lbl_result = tk.Label(master=spd, text="Sim period [ms]:")
    lbl_result.grid(row=0, column=0, sticky="nsew")

    spd.rowconfigure(0, minsize=50, weight=1)
    spd.columnconfigure('all', minsize=50, weight=1)

    btn_decrease = tk.Button(spd, text="-", width=3, command=decrease_speed)
    btn_decrease.grid(row=0, column=1, sticky="nsew")

    lbl_value = tk.Label(spd, text=str(SIM_SPEED.value))
    lbl_value.grid(row=0, column=2, padx=10)

    btn_increase = tk.Button(spd, text="+", width=3, command=increase_speed)
    btn_increase.grid(row=0, column=3, sticky="nsew")

    spd.grid(row=0, column=3, padx=40, sticky="ns")




    tmp = tk.Frame(fr_buttons)

    lbl_result = tk.Label(master=tmp, text="Temp [K]:")
    lbl_result.grid(row=0, column=0, sticky="nsew")

    tmp.rowconfigure(0, minsize=50, weight=1)
    tmp.columnconfigure('all', minsize=50, weight=1)

    btn_decrease = tk.Button(tmp, text="-", width=3, command=decrease_temp)
    btn_decrease.grid(row=0, column=1, sticky="nsew")

    lbl_temp = tk.Label(tmp, text=str(TEMPERATURE.value))
    lbl_temp.grid(row=0, column=2, padx=10)

    btn_increase = tk.Button(tmp, text="+", width=3, command=increase_temp)
    btn_increase.grid(row=0, column=3, sticky="nsew")

    tmp.grid(row=0, column=4, padx=40, sticky="ns")


    ph = tk.Frame(fr_buttons)

    global ph1, ph2, ph3, ph4, phT
    ph1 = tk.StringVar()
    ph2 = tk.StringVar()
    ph3 = tk.StringVar()
    ph4 = tk.StringVar()
    phT = tk.StringVar()

    ph1.set('7.0')
    l1 = tk.Label(ph, textvariable=ph1)
    l1_lab = tk.Label(master=ph, text="Ph 1. Kvd:")
    l1_lab.grid(row=0, column=0, padx=5, sticky="ns")
    l1.grid(row=0, column=1, padx=5, sticky="ns")

    ph2.set('8.0')
    l2 = tk.Label(ph, textvariable=ph2)
    l2_lab = tk.Label(master=ph, text="Ph 2. Kvd:")
    l2_lab.grid(row=1, column=0, padx=5, sticky="ns")
    l2.grid(row=1, column=1, padx=5, sticky="ns")

    ph3.set('9.0')
    l3 = tk.Label(ph, textvariable=ph3)
    l3_lab = tk.Label(master=ph, text="Ph 3. Kvd:")
    l3_lab.grid(row=2, column=0, padx=5, sticky="ns")
    l3.grid(row=2, column=1, padx=5, sticky="ns")

    ph4.set('10.0')
    l4 = tk.Label(ph, textvariable=ph4)
    l4_lab = tk.Label(master=ph, text="Ph 4. Kvd:")
    l4_lab.grid(row=3, column=0, padx=5, sticky="ns")
    l4.grid(row=3, column=1, padx=5, sticky="ns")

    phT.set('11.0')
    lT = tk.Label(ph, textvariable=phT)
    lT_lab = tk.Label(master=ph, text="Total Average Ph:")
    lT_lab.grid(row=4, column=0, padx=5, sticky="ns")
    lT.grid(row=4, column=1, padx=5, sticky="ns")

    ph.grid(row=1, column=0, padx=5, sticky="nsew")




    info = tk.Frame(fr_buttons)

    global iter_c, max_iter, quid_c, max_quid
    iter_c = tk.StringVar()
    max_iter = tk.StringVar()
    quid_c = tk.StringVar()
    max_quid = tk.StringVar()

    iter_c.set('7.0')
    lb1 = tk.Label(info, textvariable=iter_c)
    lb1_lab = tk.Label(master=info, text="Iteration count:")
    lb1_lab.grid(row=0, column=0, padx=5, sticky="ns")
    lb1.grid(row=0, column=1, padx=5, sticky="ns")

    max_iter.set('8.0')
    lb2 = tk.Label(info, textvariable=max_iter)
    lb2_lab = tk.Label(master=info, text="Max. iterations:")
    lb2_lab.grid(row=1, column=0, padx=5, sticky="ns")
    lb2.grid(row=1, column=1, padx=5, sticky="ns")

    quid_c.set('9.0')
    lb3 = tk.Label(info, textvariable=quid_c)
    lb3_lab = tk.Label(master=info, text="Quid count:")
    lb3_lab.grid(row=2, column=0, padx=5, sticky="ns")
    lb3.grid(row=2, column=1, padx=5, sticky="ns")

    max_quid.set('10.0')
    lb4 = tk.Label(info, textvariable=max_quid)
    lb4_lab = tk.Label(master=info, text="Max. quids:")
    lb4_lab.grid(row=3, column=0, padx=5, sticky="ns")
    lb4.grid(row=3, column=1, padx=5, sticky="ns")

    info.grid(row=1, column=1, padx=5, sticky="nsew")




    input_p = tk.Frame(fr_buttons)
    woe_var = tk.IntVar()
    waitOE = tk.Checkbutton(input_p, text="Pause on end", variable=woe_var, command=lambda: waitonend(woe_var.get())
                            ,height=5, width=20)
    waitOE.pack()
    input_p.grid(row=1, column=2, padx=5, sticky="nsew")



    fin_msg = tk.Frame(fr_buttons)

    global end_msg
    end_msg = tk.StringVar()

    end_msg.set('')
    lc1 = tk.Label(fin_msg, textvariable=end_msg)
    lc1_lab = tk.Label(master=fin_msg, text="End message:")
    lc1_lab.grid(row=0, column=0, padx=5, sticky="ns")
    lc1.grid(row=0, column=1, padx=5, sticky="ns")

    fin_msg.grid(row=1, column=3, padx=5, sticky="nsew")

    # window_main.mainloop()

    return window_main, graph_canvas


def start_window():
    def clicked(root: tk.Tk):
        global MAX_TICS, MAX_ITER, R_NUM, G_NUM, B_NUM, Y_NUM, ARR_X, ARR_Y, SIM_SPEED
        global DATA_ENTERED

        MAX_QUIDS.value = int(quids_s.get())
        MAX_ITER.value = int(iter_s.get())

        R_NUM.value = int(spinR.get())
        G_NUM.value = int(spinG.get())
        B_NUM.value = int(spinB.get())
        Y_NUM.value = int(spinY.get())

        ARR_X.value = float(x_s.get())
        ARR_Y.value = float(y_s.get())

        TEMPERATURE.value = int(t_s.get())

        root.destroy()
        DATA_ENTERED = True
        # main_window()



    window = tk.Tk()

    frame_cont = tk.Frame(window)

    frame_cont.rowconfigure(1, minsize=30, weight=1)
    frame_cont.columnconfigure(0, minsize=20, weight=1)

    global MAX_TICS, MAX_ITER, R_NUM, G_NUM, B_NUM, Y_NUM, ARR_X, ARR_Y, SIM_SPEED

    limits = tk.Frame(frame_cont)
    limits_lab = tk.Label(master=limits, text="Maximums:")
    quids_lab = tk.Label(master=limits, text="Quids:")
    var = tk.DoubleVar(value=MAX_QUIDS.value)  # initial value
    quids_s = tk.Spinbox(limits, from_=0, to=1000, width=6, textvariable=var)
    iter_lab = tk.Label(master=limits, text="Iters:")
    var = tk.DoubleVar(value=MAX_ITER.value)
    iter_s = tk.Spinbox(limits, from_=0, to=1000, width=6, textvariable=var)
    limits_lab.grid(row=0, column=0, sticky="nsew")
    quids_lab.grid(row=0, column=1, sticky="nsew")
    quids_s.grid(row=0, column=2, sticky="ew", padx=5, pady=5)
    iter_lab.grid(row=0, column=3, sticky="nsew")
    iter_s.grid(row=0, column=4, sticky="ew", padx=5, pady=5)
    limits.grid(row=0, column=0, sticky="nsw")

    rgby_nums = tk.Frame(frame_cont)
    rgby_lab = tk.Label(master=rgby_nums, text="Number of Quids:")
    r_lab = tk.Label(master=rgby_nums, text=" R:")
    var = tk.DoubleVar(value=R_NUM.value)
    spinR = tk.Spinbox(rgby_nums, from_=0, to=1000, width=6, textvariable=var)
    g_lab = tk.Label(master=rgby_nums, text=" G:")
    var = tk.DoubleVar(value=G_NUM.value)
    spinG = tk.Spinbox(rgby_nums, from_=0, to=1000, width=6, textvariable=var)
    b_lab = tk.Label(master=rgby_nums, text=" B:")
    var = tk.DoubleVar(value=B_NUM.value)
    spinB = tk.Spinbox(rgby_nums, from_=0, to=1000, width=6, textvariable=var)
    y_lab = tk.Label(master=rgby_nums, text=" Y:")
    var = tk.DoubleVar(value=Y_NUM.value)
    spinY = tk.Spinbox(rgby_nums, from_=0, to=1000, width=6, textvariable=var)
    rgby_lab.grid(row=0, column=0, sticky="nsew")
    r_lab.grid(row=0, column=1, sticky="nsew")
    spinR.grid(row=0, column=2, sticky="ew", padx=5, pady=5)
    g_lab.grid(row=0, column=3, sticky="nsew")
    spinG.grid(row=0, column=4, sticky="ew", padx=5, pady=5)
    b_lab.grid(row=0, column=5, sticky="nsew")
    spinB.grid(row=0, column=6, sticky="ew", padx=5, pady=5)
    y_lab.grid(row=0, column=7, sticky="nsew")
    spinY.grid(row=0, column=8, sticky="ew", padx=5, pady=5)
    rgby_nums.grid(row=1, column=0, sticky="nsw")

    xy_coord = tk.Frame(frame_cont)
    limits_lab = tk.Label(master=xy_coord, text="Area size:")
    x_lab = tk.Label(master=xy_coord, text="X:")
    var = tk.DoubleVar(value=ARR_X.value)
    x_s = tk.Spinbox(xy_coord, from_=0, to=1000, width=6, textvariable=var)
    y_lab = tk.Label(master=xy_coord, text="Y:")
    var = tk.DoubleVar(value=ARR_Y.value)
    y_s = tk.Spinbox(xy_coord, from_=0, to=1000, width=6, textvariable=var)
    limits_lab.grid(row=0, column=0, sticky="nsew")
    x_lab.grid(row=0, column=1, sticky="nsew")
    x_s.grid(row=0, column=2, sticky="ew", padx=5, pady=5)
    y_lab.grid(row=0, column=3, sticky="nsew")
    y_s.grid(row=0, column=4, sticky="ew", padx=5, pady=5)
    xy_coord.grid(row=2, column=0, sticky="nsw")

    temps = tk.Frame(frame_cont)
    temp_lab = tk.Label(master=temps, text="Initial temperature:")
    var = tk.DoubleVar(value=TEMPERATURE.value)
    t_s = tk.Spinbox(temps, from_=0, to=100, width=6, textvariable=var)
    temp_lab.grid(row=0, column=0, sticky="nsew")
    t_s.grid(row=0, column=2, sticky="ew", padx=5, pady=5)
    temps.grid(row=3, column=0, sticky="nsw")

    fr_buttons = tk.Frame(frame_cont)
    btn1 = tk.Button(fr_buttons, text="Ok", command=lambda: clicked(window))
    btn2 = tk.Button(fr_buttons, text="Cancel", command=lambda: quit(window))
    btn1.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
    btn2.grid(row=0, column=1, sticky="ew", padx=5)
    fr_buttons.grid(row=5, column=0, pady=50, sticky="s")

    frame_cont.pack()



    window.title('Hello Python')
    window.geometry("600x200+10+10")
    window.mainloop()

#endregion


#region ZUGI
class Variable_e(IntEnum):
    SIM_SPEED = 1
    TEMPERATURE = 2


class Type_e(IntEnum):
    RED = 1
    BLUE = 2
    GREEN = 3
    YELLOW = 4


def get_color_amount(clr: Type_e, redQuids, greenQuids, blueQuids, yellowQuids):
    if clr == Type_e.RED:
        return redQuids
    elif clr == Type_e.GREEN:
        return greenQuids
    elif clr == Type_e.BLUE:
        return blueQuids
    elif clr == Type_e.YELLOW:
        return yellowQuids

class Quid:
    def __init__(self, x, y, l_type: Type_e = Type_e.RED):
        self.l_type = l_type
        self.x = x
        self.y = y
        self.lifetime = 0
        self.size = 2
        if l_type == Type_e.RED:
            self.pH = 4
        if l_type == Type_e.BLUE:
            self.pH = 12
        if l_type == Type_e.GREEN:
            self.pH = 9
        if l_type == Type_e.YELLOW:
            self.pH = 5
        self.uuid = uuid.uuid4()
        self.move_factor = random.random()
        self.move_x_factor = round(random.uniform(-self.move_factor, self.move_factor), 4)
        self.move_y_factor = round(random.uniform(-self.move_factor, self.move_factor), 4)

    def grow(self):
        if self.size < 5:
            self.size = self.size + 1
        else:
            quid_x = random.randint(0, ARR_X.value)
            quid_y = random.randint(0, ARR_Y.value)
            newQuid = Quid(x=self.x+10, y=self.y+10, l_type=self.l_type)
            listOfQuids.append(newQuid)


    def move(self):
        self.x = self.x + self.move_x_factor * TEMPERATURE.value
        self.y = self.y + self.move_y_factor * TEMPERATURE.value

    def get_color_code(self):
        if self.l_type == Type_e.RED:
            return 'red'
        elif self.l_type == Type_e.BLUE:
            return 'blue'
        elif self.l_type == Type_e.GREEN:
            return 'green'
        elif self.l_type == Type_e.YELLOW:
            return 'yellow'

    def die(self):
        listOfQuids.remove(self)
        del(self)


class Neighbours:
    def __init__(self,quid1,quid2,distance):
        self.quid1 = quid1
        self.quid2 = quid2
        self.distance = distance

    def interaction(self):
        if self.quid1.l_type == self.quid2.l_type:
            return Quid(x=(self.quid1.x + self.quid2.x) // 2, y=(self.quid2.y + self.quid1.y) // 2, l_type=self.quid1.l_type)
        if (self.quid1.l_type == Type_e.RED and self.quid2.l_type == Type_e.GREEN) or (self.quid2.l_type == Type_e.RED and self.quid1.l_type == Type_e.GREEN):
            return None
        if (self.quid1.l_type == Type_e.BLUE and self.quid2.l_type == Type_e.YELLOW) or (self.quid1.l_type == Type_e.YELLOW and self.quid2.l_type == Type_e.BLUE):
            return None
        else:
            return 1

    def __lt__(self, other):
        return True if self.distance > other.distance else False


def creation(listOfQuids, redQuids, greenQuids, blueQuids, yellowQuids, ARR_X, ARR_Y):
    for clr in Type_e:
        for i in range(0, get_color_amount(clr, redQuids, greenQuids, blueQuids, yellowQuids)):
            quid_x = random.randint(0, ARR_X)
            quid_y = random.randint(0, ARR_Y)
            newQuid = Quid(x=quid_x, y=quid_y, l_type=clr)
            listOfQuids.append(newQuid)

    return listOfQuids


event_tick_UPR = multiprocessing.Event()
draw_tick_UPR = multiprocessing.Event()

iter_counter = 0

def tick_upr_fun():
    global event_tick_UPR, iter_counter
    # event_tick_UPR.set()    # logic
    draw_tick_UPR.set()     # draw

    if PRINT_DEBUG:
        print("tick")

    if paused:
        return

    iter_counter += 1

    # create timer for controlling UPR
    # -> timer calls function which signals with events to unblock thread
    if iter_counter < MAX_ITER.value:
        Timer((SIM_SPEED.value/1000), tick_upr_fun).start()   # /1000 for s -> ms
    else:
        global done
        done = True

def calculateDistance(x1,y1,x2,y2):
    temp = ((abs(x2-x1))**2) + ((abs(y2-y1))**2)
    temp = round(math.sqrt(temp), 4)
    return temp



class ControlLoop():
    def __init__(self, listOfQuids, R_NUM, G_NUM, B_NUM, Y_NUM, ARR_X, ARR_Y):
        self.listOfQuids = listOfQuids
        self.R_NUM = R_NUM
        self.G_NUM = G_NUM
        self.B_NUM = B_NUM
        self.Y_NUM = Y_NUM
        self.ARR_X = ARR_X
        self.ARR_Y = ARR_Y

    def run(self):
        R_NUM = self.R_NUM
        G_NUM = self.G_NUM
        B_NUM = self.B_NUM
        Y_NUM = self.Y_NUM
        ARR_X = self.ARR_X
        ARR_Y = self.ARR_Y


        if PRINT_DEBUG:
            print("calc")

        for tmp_quid in self.listOfQuids:
            tmp_quid.move()
            if tmp_quid.x < 0 or tmp_quid.x > ARR_X or tmp_quid.y < 0 or tmp_quid.y > ARR_Y:
                self.listOfQuids.remove(tmp_quid)
                if PRINT_DEBUG:
                    print("A QUID HAS ESCAPED")
                del tmp_quid
            else:
                tmp_quid.lifetime = tmp_quid.lifetime + 1
                if tmp_quid.lifetime % 4 == 0:
                    tmp_quid.grow()
                if tmp_quid.lifetime % 20 == 0:
                    tmp_quid.die()


        listOfNeighbours = []

        # CUDA PART main part of logic that need to be run on CUDA
        for elem1 in self.listOfQuids:
            minDistance = ARR_X ** 2 + ARR_Y ** 2
            neighbour_cand = None
            for elem2 in self.listOfQuids:
                if elem1.uuid == elem2.uuid:
                    continue
                else:
                    distance = calculateDistance(elem1.x, elem1.y, elem2.x, elem2.y)
                    assert distance < (ARR_X ** 2 + ARR_Y ** 2)
                    if distance < minDistance:
                        neighbour_cand = elem2
                        minDistance = distance
            if neighbour_cand:
                neighbour = Neighbours(elem1, neighbour_cand, distance)
                listOfNeighbours.append(neighbour)
            if PRINT_DEBUG:
                print("new neighbour created and the distance is for elem " + str(distance))

        # CUDA PART 2
        global phtotal, phkvadrant1, phkvadrant2, phkvadrant3, phkvadrant4, ph1count, ph2count, ph3count, ph4count
        phkvadrant1 = 0.0
        phkvadrant2 = 0.0
        phkvadrant3 = 0.0
        phkvadrant4 = 0.0
        ph1count = 0.0
        ph2count = 0.0
        ph3count = 0.0
        ph4count = 0.0
        phtotal = 0.0

        for quid in self.listOfQuids:
            if (ARR_X/2) < quid.x < ARR_X and (ARR_Y/2) < quid.y < ARR_Y:
                phkvadrant1 = phkvadrant1 + quid.pH
                ph1count = ph1count + 1
            elif 0.0 < quid.x < (ARR_X/2) and (ARR_Y/2) < quid.y < ARR_Y:
                phkvadrant2 = phkvadrant2 + quid.pH
                ph2count = ph2count + 1
            elif 0.0 < quid.x < (ARR_X/2) and 0.0 < quid.y < (ARR_Y/2):
                phkvadrant3 = phkvadrant3 + quid.pH
                ph3count = ph3count + 1
            else:
                phkvadrant4 = phkvadrant4 + quid.pH
                ph4count = ph4count + 1
        if int(ph1count) != 0:
            phkvadrant1 = round((phkvadrant1 / ph1count), 4)
        else:
            phkvadrant1 = round(7.0, 4)
        if int(ph2count) != 0:
            phkvadrant2 = round((phkvadrant2 / ph2count), 4)
        else:
            phkvadrant2 = round(7.0, 4)
        if int(ph3count) != 0:
            phkvadrant3 = round((phkvadrant3 / ph3count), 4)
        else:
            phkvadrant3 = round(7.0, 4)
        if int(ph4count) != 0:
            phkvadrant4 = round((phkvadrant4 / ph4count), 4)
        else:
            phkvadrant4 = round(7.0, 4)
        phtotal = phkvadrant3 + phkvadrant4 + phkvadrant2 + phkvadrant1
        phtotal = round((phtotal / 4), 4)



        listOfNeighbours2 = deepcopy(listOfNeighbours)
        heapq.heapify(listOfNeighbours2)
        while len(listOfNeighbours2) > 0:
            temp = heapq.heappop(listOfNeighbours2)
            if temp.distance <= 5.0:
                interactionType = temp.interaction()
                if isinstance(interactionType, Quid):
                    listOfNeighbours.append(interactionType)
                    if PRINT_DEBUG:
                        print("THEY HAD SEX")
                elif interactionType == 1:
                    if temp.quid1 in listOfNeighbours:
                        listOfNeighbours.remove(temp.quid1)
                    if temp.quid2 in listOfNeighbours:
                        listOfNeighbours.remove(temp.quid2)
                    if PRINT_DEBUG:
                        print("THEY HAVE DESTROYED THEMSELVES")

#endregion


if __name__ == '__main__':
    # PARAM INPUT
    start_window()

    main_win, main_canvas = main_window()
    while not DATA_ENTERED:
        pass

    # CONTROL CODE
    tick_upr_fun()

    listOfQuids = creation(list(), R_NUM.value, G_NUM.value, B_NUM.value, Y_NUM.value, ARR_X.value, ARR_Y.value)
    controlLoop = ControlLoop(listOfQuids, R_NUM.value, G_NUM.value, B_NUM.value, Y_NUM.value, ARR_X.value, ARR_Y.value)
    # controlLoop.start()


    # DRAWING CODE
    # canvas_test_circles(main_canvas)  # works

    max_quid.set(str(MAX_QUIDS.value))
    max_iter.set(str(MAX_ITER.value))

    i = 0.0
    LOOP_ACTIVE = True
    while LOOP_ACTIVE and 0 < len(listOfQuids) < MAX_QUIDS.value:
        i += 1
        if PRINT_DEBUG:
            print("draw")
        draw_tick_UPR.clear()

        controlLoop.run()

        for q in listOfQuids:
            main_canvas.create_circle(q.x+5, q.y+5, q.size+5, fill=q.get_color_code())

        while paused:
            main_win.update()

        quid_c.set(str(len(listOfQuids)))
        iter_c.set(str(iter_counter))

        ph1.set(str(phkvadrant1))
        ph2.set(str(phkvadrant2))
        ph3.set(str(phkvadrant3))
        ph4.set(str(phkvadrant4))
        phT.set(str(phtotal))

        main_win.update()

        if done == True:
            break

        if main_canvas:
            main_canvas.delete("all")  # works
            main_canvas.create_rectangle(0, 0, ARR_X.value, ARR_Y.value, outline="black", fill=main_canvas["background"])

        draw_tick_UPR.wait()


    if done == True:    # reached maximum number of iterations
        print(f"Reached maximum number of iterations ({MAX_ITER.value})")
        end_msg.set(f"Reached maximum number of iterations ({MAX_ITER.value})")
        main_win.update()
    elif len(listOfQuids):
        print(f"Reached maximum number of Quids ({MAX_QUIDS.value})")
        end_msg.set(f"Reached maximum number of Quids ({MAX_QUIDS.value})")
        main_win.update()
    else:
        print("No more Quids left")
        end_msg.set("No more Quids left")
        main_win.update()


    if waitOnEnd:
        while 1:
            main_win.update()
    else:
        print("Exiting")
        os._exit(0)


    os._exit(0)
