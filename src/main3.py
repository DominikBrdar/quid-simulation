#region IMPORT
import tkinter as tk
from tkinter import ttk

import multiprocessing
from enum import IntEnum
from threading import Timer
from functools import wraps
from timeit import default_timer as timer
import uuid
import os

import numpy as np


# pip install wheel
# -> got to Python Packages (bottom of screen), search for 'cupy-cuda115' and click install (right side of screen)

USECUDA = 0
#import cupy as cp


""" memo
# pip install --upgrade setuptools
# install https://aka.ms/vs/17/release/vc_redist.x64.exe
# pip install cupy  # takes FOREVER (>30 mins)
# pip install cupy-cuda115  #prebuilt binary for CUDA v11.5 -> PyCharm doesn't see it
"""







#endregion

#region CUDA-TEST

CUDATEST = 1

def cuda_fun():
    # added to path: C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.28.29333\bin\Hostx64\x64

    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule

    import numpy
    a = numpy.random.randn(4, 4)

    a = a.astype(numpy.float32)

    a_gpu = cuda.mem_alloc(a.nbytes)

    cuda.memcpy_htod(a_gpu, a)

    mod = SourceModule("""
      __global__ void doublify(float *a)
      {
        int idx = threadIdx.x + threadIdx.y*4;
        a[idx] *= 2;
      }
      """)

    func = mod.get_function("doublify")
    func(a_gpu, block=(4, 4, 1))

    a_doubled = numpy.empty_like(a)
    cuda.memcpy_dtoh(a_doubled, a_gpu)
    print(a_doubled)
    print(a)


#endregion


#region GLOBALS

PRINT_DEBUG = False
LOG_ITER_TIMES = True
LOG_EVENTS = False
LOGFILE = "iter_times.log"
#LOGFILE = "quid_count.log"
NEXT = -1
LAST = 0


# INITIAL VALUES ENTERED FROM TKINTER
MAX_QUIDS = multiprocessing.Value("i",2000)
MAX_ITER = multiprocessing.Value("i",2000)

R_NUM = multiprocessing.Value("i",100)
G_NUM = multiprocessing.Value("i",100)
B_NUM = multiprocessing.Value("i",100)
Y_NUM = multiprocessing.Value("i",100)

ARR_X = multiprocessing.Value("f",600.0)
ARR_Y = multiprocessing.Value("f",400.0)

SIM_SPEED = multiprocessing.Value("i", 100)
TEMPERATURE = multiprocessing.Value("i", 10)


DATA_ENTERED = False
waitOnEnd = False
centered = True
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

class Logger:
    def __init__(self, file):
        self.f = open(file, 'w')
        
    def log(self, output):
        self.f.write(output)
    
    def close(self):
        self.f.close()


#region GRAPHIC

def quit(root: tk.Tk):
    logger.close()
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

def cntrset(state):
    global centered
    if state == 1:
        centered = False
    else:
        centered = True

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
    fr_buttons_sub = tk.Frame(fr_buttons)
    btn_open = tk.Button(fr_buttons_sub, text="Play", command=play, height=1, width=8)
    btn_save = tk.Button(fr_buttons_sub, text="Pause", command=pause, height=1, width=8)
    btn_quit = tk.Button(fr_buttons_sub, text="Quit", fg="red", command=lambda: quit(window_main), height=1, width=8)


    btn_open.grid(row=0, column=0, padx=5, pady=5)
    btn_save.grid(row=0, column=1, padx=5)
    btn_quit.grid(row=0, column=2, padx=5)

    fr_buttons_sub.grid(row=0, column=0, padx=5)

    fr_buttons.grid(row=0, column=0, sticky="ns")
    global centered
    if centered:
        graph_canvas.grid(row=1, column=0, sticky="nsew")
    else:
        graph_canvas.grid(row=1, column=0)



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

    global iter_c, max_iter, quid_c, max_quid, msiter, maxfps
    iter_c = tk.StringVar()
    max_iter = tk.StringVar()
    quid_c = tk.StringVar()
    max_quid = tk.StringVar()
    msiter = tk.StringVar()
    maxfps = tk.StringVar()

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

    msiter.set('0.0')
    lb5 = tk.Label(info, textvariable=msiter)
    lb5_lab = tk.Label(master=info, text="s/iter:")
    lb5_lab.grid(row=4, column=0, padx=5, sticky="ns")
    lb5.grid(row=4, column=1, padx=5, sticky="ns")
    print(info)

    maxfps.set('0.0')
    lb6 = tk.Label(info, textvariable=maxfps)
    lb6_lab = tk.Label(master=info, text="-> Max. FPS:")
    lb6_lab.grid(row=4, column=2, padx=5, sticky="ns")
    lb6.grid(row=4, column=3, padx=5, sticky="ns")

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
    window.resizable(False, False)

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

    input_centre = tk.Frame(frame_cont)
    cntr_var = tk.IntVar()
    cntrOE = tk.Checkbutton(input_centre, text="Centre canvas", variable=cntr_var,
                            command=lambda: cntrset(cntr_var.get())
                            , height=5, width=20)
    cntrOE.pack()
    input_centre.grid(row=4, column=0, padx=5, sticky="w")

    fr_buttons = tk.Frame(frame_cont)
    btn1 = tk.Button(fr_buttons, text="Ok", command=lambda: clicked(window))
    btn2 = tk.Button(fr_buttons, text="Cancel", command=lambda: quit(window))
    btn1.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
    btn2.grid(row=0, column=1, sticky="ew", padx=5)
    fr_buttons.grid(row=5, column=0, pady=10, sticky="s")

    frame_cont.pack()


    window.title('Parameter input')
    window.geometry("600x250+10+10")
    window.mainloop()

#endregion

class Type_e(IntEnum):
    RED = 1
    BLUE = 2
    GREEN = 3
    YELLOW = 4

class Quid:
    def __init__(self, pos, l_type: Type_e = Type_e.RED):
        global NEXT
        self.l_type = l_type 
        self.pos = pos
        self.dir = np.random.uniform(-1,1,size=2)
        self.lifetime = 0
        self.size = 2
        if l_type == Type_e.RED:
            self.pH = 4
            self.type = np.array([1, 0])
        if l_type == Type_e.BLUE:
            self.pH = 12
            self.type = np.array([0, 1])
        if l_type == Type_e.GREEN:
            self.pH = 9
            self.type = np.array([-1, 0])
        if l_type == Type_e.YELLOW:
            self.pH = 5
            self.type = np.array([0, -1])
        self.uuid = uuid.uuid4()
        
        NEXT = (NEXT + 1) % MAX_QUIDS.value
        self.index = freeSlots[NEXT]    
        
        posx[self.index // x][self.index % x] = self.pos[0]
        dirx[self.index // x][self.index % x] = self.dir[0]
        
        
    def grow(self):
        global NEXT
        if self.size < 5:
            self.size = self.size + 1
        else:
            if LOG_EVENTS:
                logger.log("vegetational child\n")
            listOfQuids[freeSlots[NEXT]] = Quid(pos=self.pos - self.dir, l_type=self.l_type)

    def move(self):
       self.pos = self.pos + np.round(self.dir * TEMPERATURE.value)

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
        global LAST
        listOfQuids[self.index] = None
        freeSlots[LAST] = self.index
        LAST = (LAST + 1) % MAX_QUIDS.value
        posx[self.index // x][self.index % x] = -1
        dirx[self.index // x][self.index % x] = 0
        del(self)


def creation(redQuids, greenQuids, blueQuids, yellowQuids, ARR_X, ARR_Y):
    for clr in Type_e:
        for i in range(0, get_color_amount(clr, redQuids, greenQuids, blueQuids, yellowQuids)):
            quid_x = np.random.randint(0, ARR_X)
            quid_y = np.random.randint(0, ARR_Y)
            listOfQuids[freeSlots[NEXT]] = Quid(pos=[quid_x, quid_y], l_type=clr)

# može bolje
# kao dot produkt vektora:  
# isti => 1
# R(1,0), G(-1,0) => -1
# B(0,1), Y(0,-1) => -1
# ostale kombinacije daju 0
def interaction(quid1, quid2):
    if USECUDA:
        r = np.dot(quid1.type, quid2.type)
    else:
        r = np.dot(quid1.type, quid2.type)
    if r == 1:
        if PRINT_DEBUG:
            ("THEY HAD SEX ")
        return Quid(pos=(quid1.pos+quid2.pos)//2, l_type=quid1.l_type)
    
    elif r != 1:
        if LOG_EVENTS:
            logger.log("Quids of oposite types destroyed themselves")
        quid1.die()
        quid2.die()

def get_color_amount(clr: Type_e, redQuids, greenQuids, blueQuids, yellowQuids):
    if clr == Type_e.RED:
        return redQuids
    elif clr == Type_e.GREEN:
        return greenQuids
    elif clr == Type_e.BLUE:
        return blueQuids
    elif clr == Type_e.YELLOW:
        return yellowQuids


msiter = 0.0
maxfps = 0.0

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = timer()
        result = f(*args, **kw)
        te = timer()
        if PRINT_DEBUG:
            print(f"Iteration %d" % iter_counter, f"took: %2.8f sec" % (te-ts))
        if LOG_ITER_TIMES:
            logger.log(f"Iteration %d " % iter_counter + f"took: %2.8f sec\n" % (te-ts))
        global msiter
        msiter.set('{:.8f}'.format(round((te-ts), 8)))
        maxfps.set('{:4.4f}'.format(round(1/(te-ts), 4)))
        return result
    return wrap

# CUDA part1
x = np.gcd(int(np.ceil(np.sqrt(MAX_QUIDS.value))), int(np.ceil(MAX_QUIDS.value)))
y = int(MAX_QUIDS.value / x)
posx = cp.empty((x,y))
dirx = cp.empty((x,y))
def init_moving():
    k = 0
    for i in range(x):
        for j in range(y):
            if listOfQuids[k]:
                posx[i][j] = listOfQuids[k].pos[0]
            else:
                posx[i][j] = -1
            k += 1
        
    k = 0
    for i in range(x):
        for j in range(y):
            if listOfQuids[k]:
                dirx[i][j] = listOfQuids[k].dir[0]
            else:
                dirx[i][j] = 0
            k += 1
            


''' fix moving

in main prije run:
    posx = posx + dirx

in die():
posx[self.index / /x][self.index % x] = -1
dirx[self.index / /x][self.index % x] = 0

in create
posx[self.index / /x][self.index % x] = self.pos[0]
dirx[self.index / /x][self.index % x] = self.dir[1]
'''


class ControlLoop():
    def __init__(self, ARR_X, ARR_Y):
        self.ARR_X = ARR_X
        self.ARR_Y = ARR_Y

    @timing
    def run(self):
        ARR_X = self.ARR_X
        ARR_Y = self.ARR_Y

        global iter_counter
        iter_counter += 1
        
        if PRINT_DEBUG:
            print("calc")

        # CUDA 1
        #move_cuda()
        # paralelno zbroji
        # listu vektora pozicija i listu vektora smjerova gibanja
        # paralelno svima provjeri jesu li u intervalu (području promatranja)
        for i in range(MAX_QUIDS.value):
            tmp_quid = listOfQuids[i]
            if not tmp_quid: continue
            #tmp_quid.move()
            if tmp_quid.pos[0] < 0 or tmp_quid.pos[0] > ARR_X or tmp_quid.pos[1] < 0 or tmp_quid.pos[1] > ARR_Y:
                if LOG_EVENTS:
                    logger.log("A quid has escaped\n")
                tmp_quid.die()
                
            else:
                tmp_quid.lifetime = tmp_quid.lifetime + 1
                if tmp_quid.lifetime % 4 == 0:
                    tmp_quid.grow()
                if tmp_quid.lifetime % 20 == 0:
                    tmp_quid.die()
                    if LOG_EVENTS:
                        logger.log("quid died from old age \n")

        # CUDA 2 main part of logic that need to be run on CUDA
        for i in range(MAX_QUIDS.value):
            quid1 = listOfQuids[i]
            if not quid1: continue
            md = 5
            for j in range(MAX_QUIDS.value): # svaki sa svakim - dovoljno je proći trokut
                quid2 = listOfQuids[j]
                if not quid2: continue
                d = np.linalg.norm(quid1.pos - quid2.pos)
                if d < md: 
                    md = d
                    q2 = j
            if md < 5:
                interaction(quid1, listOfQuids[q2])

                # kad već računamo svaki sa svakim,
                # odredimo sve međusobne udaljenosti
                # a poslje oduzmemo koji su bliži od zadane granice
                # zašto ne bismo jednostavno odredili matricu susjedstva
          

        # CUDA PART 3
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

        for i in range(MAX_QUIDS.value):
            quid = listOfQuids[i]
            if not quid: continue
            if (ARR_X/2) < quid.pos[0] < ARR_X and (ARR_Y/2) < quid.pos[1] < ARR_Y:
                phkvadrant1 = phkvadrant1 + quid.pH
                ph1count = ph1count + 1
            elif 0.0 < quid.pos[0] < (ARR_X/2) and (ARR_Y/2) < quid.pos[1] < ARR_Y:
                phkvadrant2 = phkvadrant2 + quid.pH
                ph2count = ph2count + 1
            elif 0.0 < quid.pos[0] < (ARR_X/2) and 0.0 < quid.pos[1] < (ARR_Y/2):
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
                    


#region ZUGI
class Variable_e(IntEnum):
    SIM_SPEED = 1
    TEMPERATURE = 2

event_tick_UPR = multiprocessing.Event()
draw_tick_UPR = multiprocessing.Event()

iter_counter = 0


def tick_upr_fun():
    global event_tick_UPR
    # event_tick_UPR.set()    # logic
    draw_tick_UPR.set()     # draw

    if PRINT_DEBUG:
        print("tick")

    if paused:
        return

    # create timer for controlling UPR
    # -> timer calls function which signals with events to unblock thread
    if iter_counter < MAX_ITER.value:
        Timer((SIM_SPEED.value/1000), tick_upr_fun).start()# /1000 for s -> ms
    else:
        global done
        done = True

#endregion


listOfQuids = np.empty(MAX_QUIDS.value, dtype=Quid)
freeSlots = np.arange(MAX_QUIDS.value)
    
if __name__ == '__main__':
    if CUDATEST:
        cuda_fun()
        exit(0)

    logger = Logger(LOGFILE)
    
    # PARAM INPUT
    start_window()

    main_win, main_canvas = main_window()
    while not DATA_ENTERED:
        pass

    # CONTROL CODE
    tick_upr_fun()

    
    creation(R_NUM.value, G_NUM.value, B_NUM.value, Y_NUM.value, ARR_X.value, ARR_Y.value)
    controlLoop = ControlLoop(ARR_X.value, ARR_Y.value)


    # DRAWING CODE
    # canvas_test_circles(main_canvas)  # works

    max_quid.set(str(MAX_QUIDS.value))
    max_iter.set(str(MAX_ITER.value))

    LOOP_ACTIVE = True
    quid_counter = np.count_nonzero(listOfQuids)
    while LOOP_ACTIVE and 0 < quid_counter < MAX_QUIDS.value:
        if PRINT_DEBUG:
            print("draw")
        draw_tick_UPR.clear()

        posx = posx + dirx
        controlLoop.run()

        for i in range(MAX_QUIDS.value):
            q = listOfQuids[i]
            if q:
                main_canvas.create_circle(q.pos[posx[q.index // x][q.index % x]]+5, q.pos[1]+5, q.size+5, fill=q.get_color_code())

        while paused:
            main_win.update()

        quid_counter = np.count_nonzero(listOfQuids)
        quid_c.set(str(quid_counter))
        iter_c.set(str(iter_counter))

        ph1.set('{:.4f}'.format(round(phkvadrant1, 4)))
        ph2.set('{:.4f}'.format(round(phkvadrant2, 4)))
        ph3.set('{:.4f}'.format(round(phkvadrant3, 4)))
        ph4.set('{:.4f}'.format(round(phkvadrant4, 4)))
        phT.set('{:.4f}'.format(round(phtotal, 4)))

        main_win.update()

        if done == True:
            break

        if main_canvas:
            main_canvas.delete("all") # TODO debug on exit (X)
            if centered:
                main_canvas.create_rectangle(0, 0, ARR_X.value, ARR_Y.value, outline="black", fill=main_canvas["background"])

        draw_tick_UPR.wait()


    if done == True:    # reached maximum number of iterations
        print(f"Reached maximum number of iterations ({MAX_ITER.value})")
        end_msg.set(f"Reached maximum number of iterations ({MAX_ITER.value})")
        main_win.update()
    elif LAST == NEXT: # upitno dali radi
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
        logger.close()
        os._exit(0)

    logger.close()
    os._exit(0)
