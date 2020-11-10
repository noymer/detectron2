import time
import numpy as np

time_arr = np.ndarray(0)

class Timer:
    def __init__(self):
        self.is_measuring = False
        self.start_time = None
        self.end_time = None

    def start(self):
        if not self.is_measuring:
            self.start_time = time.perf_counter()
            self.is_measuring = True 
    
    def stop(self):
        if self.is_measuring:
            self.end_time = time.perf_counter()
            self.is_measuring = False
    
    def getTime(self):
        return self.end_time - self.start_time
    
    def save(self):
        global time_arr
        time_arr = np.append(time_arr, self.getTime())

def reset():
    global time_arr
    time_arr = np.ndarray(0)

def get_result(print_out=True):
    global time_arr
    time_mean = time_arr.mean()
    time_min = time_arr.min()
    time_max = time_arr.max()
    if print_out:
        print("\n")
        print("min time:\t{}ms".format(time_min * 1000))
        print("mean time:\t{}ms".format(time_mean * 1000))
        print("max time:\t{}ms".format(time_max * 1000))
    
    return time_arr



