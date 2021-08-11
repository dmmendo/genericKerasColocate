import multiprocessing
import time
import numpy as np
import sys

from keras.models import load_model
from keras.backend import clear_session

def get_input_shape(model):
    raw_in_shape = model.layers[0].input_shape[0]
    in_shape = []
    for entry in raw_in_shape:
        if entry == None:
            in_shape.append(1)
        else:
            in_shape.append(entry)
    return in_shape


def increment_q_count(count,l,arg=1):
    l.acquire()
    count.value += arg
    l.release()

def decrement_q_count(count,l):
    l.acquire()
    if count.value == 0:
        print("ERROR: attempting to decrement 0")
    count.value -= 1
    l.release()

def producer(q_count,l,e,ready_e,filename,interval_t,max_count,dut_ready_e,other_cons_ready_e):
    f = open(filename,'r')
    local_count = 0

    #first, wait for consumer to be ready
    ready_e.wait()
    dut_ready_e.wait()
    other_cons_ready_e.wait()

    while local_count < max_count:
        line = f.readline()
        if len(line) == 0:
            f.close()
            f = open(filename,'r')
            line = f.readline()
        
        s = int(line.split('\n')[0])
        if s > 0:
            increment_q_count(q_count,l,s)
            e.set()
            local_count += s

        time.sleep(interval_t)
    f.close()

def consumer(q_count,l,e,ready_e,finish_e,model_file,max_count):
    model = load_model(model_file)
    x = np.ones(get_input_shape(model))

    model.predict(x)
   
    ready_e.set()
        
    local_count = 0
    while local_count < max_count:
        if q_count.value == 0:
            e.wait()
            e.clear()
            
        decrement_q_count(q_count,l)
        if q_count.value == 0:
            e.clear()

        local_count += 1

        model.predict(x)

    finish_e.set()

    print("local_count:",local_count)
    clear_session()

def measure_run(cons0_ready_e,cons1_ready_e,dut_ready_e,finish_pc0_e,finish_pc_1,model_file,out_file):
    model = load_model(model_file)
    x = np.ones(get_input_shape(model))

    model.predict(x)

    dut_ready_e.set()
    cons0_ready_e.wait()
    cons1_ready_e.wait()

    runtimes = []
    while finish_pc0_e.is_set() is False and finish_pc1_e.is_set() is False:
        start_t = time.perf_counter()
        model.predict(x)
        end_t = time.perf_counter()
        runtimes.append(end_t - start_t)

    clear_session()

    f = open(out_file,'a')
    f.write(str(runtimes[0]))
    for entry in runtimes[1:]:
        f.write(','+str(entry))
    f.write('\n')
    f.close()

if __name__ == '__main__':
    q_count_pc0 = multiprocessing.Value('i',0)
    l_pc0 = multiprocessing.Lock()
    e_pc0 = multiprocessing.Event()
    cons0_ready_e = multiprocessing.Event()
    finish_pc0_e = multiprocessing.Event()

    q_count_pc1 = multiprocessing.Value('i',0)
    l_pc1 = multiprocessing.Lock()
    e_pc1 = multiprocessing.Event()
    cons1_ready_e = multiprocessing.Event()
    finish_pc1_e = multiprocessing.Event()

    dut_ready_e = multiprocessing.Event()

    
    #example params
    """
    model_file = '/home/danielmendoza865/gen_models/vggstyle_cnn_v2_47/1'
    col_model0_file = '/home/danielmendoza865/gen_models/vggstyle_cnn_v2_47/1'
    col_model1_file = '/home/danielmendoza865/gen_models/vggstyle_cnn_v2_47/1'
    arrival_dist_filename = '/home/danielmendoza865/poisson_arrival/qrate-0.5_interval-2ms_tcount-100_0.poisson'
    out_file = 'test_results.txt'
    interval_t = 0.002
    max_count = 100
    """

    
    model_file = sys.argv[1]
    col_model0_file = sys.argv[2]
    col_model1_file = sys.argv[3]
    arrival_dist_filename = sys.argv[4]
    out_file = sys.argv[5]
    interval_t = float(sys.argv[6])
    max_count = int(sys.argv[7])
    
    p_prod0 = multiprocessing.Process(target=producer, args=(q_count_pc0,l_pc0,e_pc0,cons0_ready_e,arrival_dist_filename,interval_t,max_count,dut_ready_e,cons1_ready_e))
    p_cons0 = multiprocessing.Process(target=consumer, args=(q_count_pc0,l_pc0,e_pc0,cons0_ready_e,finish_pc0_e,col_model0_file,max_count))

    p_prod1 = multiprocessing.Process(target=producer, args=(q_count_pc1,l_pc1,e_pc1,cons1_ready_e,arrival_dist_filename,interval_t,max_count,dut_ready_e,cons0_ready_e))
    p_cons1 = multiprocessing.Process(target=consumer, args=(q_count_pc1,l_pc1,e_pc1,cons1_ready_e,finish_pc1_e,col_model1_file,max_count))

    p_dut = multiprocessing.Process(target=measure_run, args=(cons0_ready_e,cons1_ready_e,dut_ready_e,finish_pc0_e,finish_pc1_e,model_file,out_file))

    start_t = time.time()
    p_cons0.start()
    p_prod0.start()
    p_cons1.start()
    p_prod1.start()
    p_dut.start()
    p_cons0.join()
    p_prod0.join()
    p_cons1.join()
    p_prod1.join()
    p_dut.join()
    print(time.time() - start_t)
