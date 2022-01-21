# region IMPORT
from array import array
from lib2to3.pgen2.token import STAR
import tkinter as tk

import multiprocessing
from threading import Timer
from functools import wraps
from timeit import default_timer as timer
from tracemalloc import start
from turtle import pos
import uuid
import os

# import cupy as cp


USECUDA = 0
CUDATEST = 0


DEMOVER = 1

"""
1 - pravedno
2 - dominantni red jedino ostavi kompatibilni green
3 - red vs yellow (blue vs green se isto ne vole)
4 - yellow i blue se vole -> explosion
"""

""" memo
# pip install wheel
# -> got to Python Packages (bottom of screen), search for 'cupy-cuda115' and click install (right side of screen)

# pip install --upgrade setuptools
# install https://aka.ms/vs/17/release/vc_redist.x64.exe
# pip install cupy  # takes FOREVER (>30 mins)
# pip install cupy-cuda115  #prebuilt binary for CUDA v11.5 -> PyCharm doesn't see it
"""


# endregion

# region CUDA-TEST

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


# endregion


# region GLOBALS

PRINT_DEBUG = False
LOG_ITER_TIMES = True
LOG_EVENTS = False
LOGFILE = "iter_times.log"
# LOGFILE = "quid_count.log"


# INITIAL VALUES ENTERED FROM TKINTER
MAX_QUIDS = multiprocessing.Value("i", 2500)
MAX_ITER = multiprocessing.Value("i", 2000)

SIM_SPEED = multiprocessing.Value("i", 100)
TEMPERATURE = multiprocessing.Value("i", 10)

if DEMOVER == 1:
    R_NUM = multiprocessing.Value("i", 200)
    G_NUM = multiprocessing.Value("i", 200)
    B_NUM = multiprocessing.Value("i", 200)
    Y_NUM = multiprocessing.Value("i", 200)
    TEMPERATURE = multiprocessing.Value("i", 10)
elif DEMOVER == 2:
    R_NUM = multiprocessing.Value("i", 100)
    G_NUM = multiprocessing.Value("i", 20)
    B_NUM = multiprocessing.Value("i", 20)
    Y_NUM = multiprocessing.Value("i", 50)
elif DEMOVER == 3:
    R_NUM = multiprocessing.Value("i", 200)
    G_NUM = multiprocessing.Value("i", 0)
    B_NUM = multiprocessing.Value("i", 0)
    Y_NUM = multiprocessing.Value("i", 200)
elif DEMOVER == 4:
    R_NUM = multiprocessing.Value("i", 0)
    G_NUM = multiprocessing.Value("i", 0)
    B_NUM = multiprocessing.Value("i", 200)
    Y_NUM = multiprocessing.Value("i", 200)

ARR_X = multiprocessing.Value("f", 1200.0)
ARR_Y = multiprocessing.Value("f", 500.0)



DATA_ENTERED = False
waitOnEnd = False
centered = True
done = False
p = ''

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


# endregion

class Logger:
    def __init__(self, file):
        self.f = open(file, 'w')

    def log(self, output):
        self.f.write(output)

    def close(self):
        self.f.close()


# region GRAPHIC

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
    return self.create_oval(x - r, y - r, x + r, y + r, **kwargs)


tk.Canvas.create_circle = _create_circle


def _create_circle_arc(self, x, y, r, **kwargs):
    if "start" in kwargs and "end" in kwargs:
        kwargs["extent"] = kwargs["end"] - kwargs["start"]
        del kwargs["end"]
    return self.create_arc(x - r, y - r, x + r, y + r, **kwargs)


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

    ph1.set('4.0')
    l1 = tk.Label(ph, textvariable=ph1)
    l1_lab = tk.Label(master=ph, text="Ph 1. Kvd:")
    l1_lab.grid(row=0, column=0, padx=5, sticky="ns")
    l1.grid(row=0, column=1, padx=5, sticky="ns")

    ph2.set('6.0')
    l2 = tk.Label(ph, textvariable=ph2)
    l2_lab = tk.Label(master=ph, text="Ph 2. Kvd:")
    l2_lab.grid(row=1, column=0, padx=5, sticky="ns")
    l2.grid(row=1, column=1, padx=5, sticky="ns")

    ph3.set('9.0')
    l3 = tk.Label(ph, textvariable=ph3)
    l3_lab = tk.Label(master=ph, text="Ph 3. Kvd:")
    l3_lab.grid(row=2, column=0, padx=5, sticky="ns")
    l3.grid(row=2, column=1, padx=5, sticky="ns")

    ph4.set('11.0')
    l4 = tk.Label(ph, textvariable=ph4)
    l4_lab = tk.Label(master=ph, text="Ph 4. Kvd:")
    l4_lab.grid(row=3, column=0, padx=5, sticky="ns")
    l4.grid(row=3, column=1, padx=5, sticky="ns")

    phT.set('7.5')
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
                            , height=5, width=20)
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

    return window_main, graph_canvas, fr_buttons


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
    quids_s = tk.Spinbox(limits, from_=0, to=10000, width=6, textvariable=var)
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


# endregion

event_tick_UPR = multiprocessing.Event()
draw_tick_UPR = multiprocessing.Event()

iter_counter = 0


def tick_upr_fun():
    global event_tick_UPR
    # event_tick_UPR.set()    # logic
    draw_tick_UPR.set()  # draw

    if PRINT_DEBUG:
        print("tick")

    if paused:
        return

    # create timer for controlling UPR
    # -> timer calls function which signals with events to unblock thread
    if iter_counter < MAX_ITER.value:
        Timer((SIM_SPEED.value / 1000), tick_upr_fun).start()  # /1000 for s -> ms
    else:
        global done
        done = True


# endregion

if __name__ == '__main__':
    if CUDATEST:
        cuda_fun()
        exit(0)

    logger = Logger(LOGFILE)

    # PARAM INPUT
    start_window()

    main_win, main_canvas, fr_buttons = main_window()
    while not DATA_ENTERED:
        pass

    # CONTROL CODE
    tick_upr_fun()

    max_quid.set(str(MAX_QUIDS.value))
    max_iter.set(str(MAX_ITER.value))

    # hack
    if USECUDA:
        import cupy as np
    else:
        import numpy as np
    # - - - - - - - - - - -{ . )CUDA( . }- - - - - - - - - - - #
    time1 = timer()
    MAX = MAX_QUIDS.value
    START = R_NUM.value + G_NUM.value + B_NUM.value + Y_NUM.value
    X = ARR_X.value
    Y = ARR_Y.value
    T = TEMPERATURE.value
    center_coord = np.array([X / 2, Y / 2])
    colors_pH_dict = {  # this dictionary instructs how will interactions play out
        "red": 4,  # 2 quids whos sum(ph) % 2 == 0 will make ofspings
        "green": 6,  # 2 quids whos sum(ph) == 15  will kill themselfs
        "blue": 9,  # if they come close enough
        "yellow": 11
    }
    # depricated // kompliciranije nego što je potrebno
    # types = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

    # init data structures for cuda
    Q = [*range(START)]  # za praćenje broja quidova
    size_c = np.zeros(MAX, dtype=int)
    ph_c = np.zeros(MAX, dtype=int)
    i = 0
    for q in Q:
        size_c[q] = 1
        if i < R_NUM.value:
            ph_c[q] = colors_pH_dict['red']
        elif i < R_NUM.value + G_NUM.value:
            ph_c[q] = colors_pH_dict['green']
        elif i < R_NUM.value + G_NUM.value + B_NUM.value:
            ph_c[q] = colors_pH_dict['blue']
        elif i < START:
            ph_c[q] = colors_pH_dict['yellow']
        i += 1
    pos_c = np.zeros((MAX, 2), dtype=int)
    pos_c[:START, :1] = np.random.randint(1, X, size=(START, 1))
    pos_c[:START, 1:] = np.random.randint(Y, size=(START, 1))

    for i in range(MAX):
        if i not in Q:
            pos_c[i] = np.array([1000000, 1000000])

    dir_c = np.random.randint(-T, T, size=(MAX, 2), dtype=int)

    # type_c = np.zeros((MAX, 2), dtype=int)
    # type_c[:START, :] = np.resize(types, (START, 2))

    # umjesto udaljenosti, računamo zbroj x i y komponenata poveznice
    # uvijet udaljenost < 5 je ekvivalnentno: x+y < 7
    # ** sjeti se pitagore i jednakokračnog trokuta
    nearest = np.resize(7, MAX)


    def get_color_code(i):
        if ph_c[i] == colors_pH_dict['red']:
            return 'red'
        elif ph_c[i] == colors_pH_dict['green']:
            return 'green'
        elif ph_c[i] == colors_pH_dict['blue']:
            return 'blue'
        elif ph_c[i] == colors_pH_dict['yellow']:
            return 'yellow'


    quid_counter = len(Q)
    time2 = timer()
    if USECUDA:
        logger.log("Using CUDA with CuPy... ")
    else:
        logger.log("Using NumPy... ")
    logger.log(f"Initialization took: %2.8f sec \n" % (time2 - time1))

    while 0 < quid_counter < MAX_QUIDS.value:
        draw_tick_UPR.clear()

        iter_counter += 1
        t1 = timer()  # start timer

        # 1 CUDA iteration ...
        # calc nearest neighbour
        nearestx = np.abs(pos_c[:, 1:] - pos_c[:, 1:].transpose())
        nearesty = np.abs(pos_c[:, :1] - pos_c[:, :1].transpose())
        nearestMat = nearestx + nearesty
        for i in Q:
            nearestMat[i][i] = 1000000

        # this will also return neighbours that don't exist
        # non-existing neighbours will have distance == 0
        nearest = np.where(nearestMat < 7)

        # calculate interactin
        for q1 in Q:
            if q1 in nearest[0]:
                q2 = nearest[1][np.where(nearest[0] == q1)][0]
                if q1 < q2 and q2 in Q:
                    a = ph_c[q1] + ph_c[q2]
                    # depricated
                    # a = np.dot(type_c[q1], type_c[q2])
                    if a % 2 == 0:
                        Q.append((Q[-1] + 1) % MAX)
                        pos_c[Q[-1]] = (pos_c[q1] + pos_c[q2]) // 2
                        ph_c[Q[-1]] = ph_c[q1]
                    elif a == 15:
                        Q.remove(q1)
                        Q.remove(q2)

        # move
        pos_c += dir_c * (TEMPERATURE.value // 3)

        # check if it escaped
        for q in Q:
            if pos_c[q][0] < 0 or pos_c[q][0] > X or pos_c[q][1] > Y or pos_c[q][1] < 0:
                Q.remove(q)

        # grow
        for q in Q:
            size_c[q] += 1
            if size_c[q] == 12:
                Q.append((Q[-1] + 1) % MAX)
                # pos_c[Q[-1]] = np.array([np.random.randint(X), np.random.randint(Y)])
                pos_c[Q[-1]] = pos_c[q] + 2
                ph_c[Q[-1]] = ph_c[q]
            elif size_c[q] == 17:
                size_c[q] = 0
                Q.remove(q)

        # pH change
        phkvadrant1 = 0.0
        phkvadrant2 = 0.0
        phkvadrant3 = 0.0
        phkvadrant4 = 0.0
        ph1count = 0.0
        ph2count = 0.0
        ph3count = 0.0
        ph4count = 0.0
        phtotal = 0.0

        tmp_pos = pos_c - center_coord
        for q in Q:
            if tmp_pos[q][0] > 0:
                if tmp_pos[q][1] > 0:
                    phkvadrant1 = phkvadrant1 + ph_c[q]
                    ph1count += 1
                else:
                    phkvadrant4 = phkvadrant4 + ph_c[q]
                    ph4count += 1
            else:
                if tmp_pos[q][1] > 0:
                    phkvadrant2 = phkvadrant2 + ph_c[q]
                    ph2count += 1
                else:
                    phkvadrant3 = phkvadrant3 + ph_c[q]
                    ph3count += 1

        if int(ph1count) != 0:
            phkvadrant1 = phkvadrant1 / ph1count
        if int(ph2count) != 0:
            phkvadrant2 = phkvadrant2 / ph2count
        if int(ph3count) != 0:
            phkvadrant3 = phkvadrant3 / ph3count
        if int(ph4count) != 0:
            phkvadrant4 = phkvadrant4 / ph4count
        phtotal = (phkvadrant4 + phkvadrant3 + phkvadrant2 + phkvadrant1) / 4
        if USECUDA:
            np.cuda.Stream.null.synchronize()
        # - - - - - - - - - - - - - - - - - - - - - - - - - #

        t2 = timer()  # stop timer
        logger.log(f"Iteration %d " % iter_counter + f"took: %2.8f sec, " % (t2 - t1) + f"quid counter = %d\n" % len(Q))

        msiter.set('{:.8f}'.format(round((t2 - t1), 8)))
        maxfps.set('{:4.4f}'.format(round(1 / (t2 - t1), 4)))

        # draw circles
        for q in Q:
            main_canvas.create_circle(pos_c[q][0], pos_c[q][1], size_c[q], fill=get_color_code(q))

        while paused:
            main_win.update()

        quid_counter = len(Q)
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
            if 0 < quid_counter < MAX_QUIDS.value and not done and iter_counter < MAX_ITER.value:
                main_canvas.delete("all")
                if centered:
                    main_canvas.create_rectangle(0, 0, ARR_X.value, ARR_Y.value, outline="black",
                                                 fill=main_canvas["background"])

        draw_tick_UPR.wait()

    if done == True:  # reached maximum number of iterations
        p = f"Reached maximum number of iterations ({MAX_ITER.value})"
        print(p)
        end_msg.set(f"Reached maximum number of iterations ({MAX_ITER.value})")
        main_win.update()
    elif quid_counter >= MAX_QUIDS.value:
        p = f"Reached maximum number of Quids ({MAX_QUIDS.value})"
        print(p)
        end_msg.set(f"Reached maximum number of Quids ({MAX_QUIDS.value})")
        main_win.update()
    else:
        print("No more Quids left")
        end_msg.set("No more Quids left")
        main_win.update()

    if waitOnEnd:
        while 1:
            pass
            # fr_buttons.update()
            main_win.update()
    else:
        print("Exiting")
        logger.log("Simulation completed\n" + p)
        logger.close()
        os._exit(0)

    logger.close()
    os._exit(0)
