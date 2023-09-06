import threading

semaphore_ev = threading.Semaphore(1)
semaphore_bld = threading.Semaphore(1)
semaphore_pv = threading.Semaphore(1)
semaphore_combine = threading.Semaphore(1)
semaphore_log = threading.Semaphore(1)