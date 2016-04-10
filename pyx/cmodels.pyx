import cython

cimport cmodels_defs as cpp
import numpy as np
cimport numpy as np
from pandas import DataFrame

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef class delay:
    cdef cpp.delay *thisptr
    def __cinit__(self, _param, _init_state = []):
        cdef np.ndarray[double, ndim = 1, mode = 'c'] param
        param = np.array(_param)
        cdef np.ndarray[double, ndim = 1, mode = 'c'] init_state = np.array(_init_state)
        cdef double* init_state_ptr
        if len(_init_state) == 0:
            init_state_ptr = NULL
        else:
            init_state_ptr = &init_state[0]
        self.thisptr = new cpp.delay(&param[0], init_state_ptr)
    def __dealloc__(self):
        del self.thisptr
    def run(self, _input_data):
        cdef np.ndarray[double, ndim = 1, mode = 'c'] q_in
        q_in  = np.array(_input_data[0])
        cdef np.ndarray[double, ndim = 1, mode = 'c'] q_out = np.empty_like(q_in)
        cdef double* input_data[1]
        cdef double* output_data[1]
        input_data[0] = &q_in[0]
        output_data[0] = &q_out[0]
        self.thisptr.run(input_data, output_data, q_in.size)
        return q_out
    def get_name(self):
        return 'delay'
    def get_ptr(self):
        return <unsigned long long>self.thisptr.get_ptr()
    def get_param_nb(self):
        return self.thisptr.get_param_nb()

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef class gr4h:
    cdef cpp.gr4h *thisptr
    def __cinit__(self, _param, _init_state = []):
        cdef np.ndarray[double, ndim = 1, mode = 'c'] param
        param = np.array(_param)
        cdef np.ndarray[double, ndim = 1, mode = 'c'] init_state = np.array(_init_state)
        cdef double* init_state_ptr
        if len(_init_state) == 0:
            init_state_ptr = NULL
        else:
            init_state_ptr = &init_state[0]
        self.thisptr = new cpp.gr4h(&param[0], init_state_ptr)
    def __dealloc__(self):
        del self.thisptr
    def run(self, _input_data):
        cdef np.ndarray[double, ndim = 1, mode = 'c'] p = np.array(_input_data[0])
        cdef np.ndarray[double, ndim = 1, mode = 'c'] e = np.array(_input_data[1])
        cdef np.ndarray[double, ndim = 1, mode = 'c'] q_out = np.empty_like(p)
        cdef double* input_data[2]
        cdef double* output_data[1]
        input_data[0] = &p[0]
        input_data[1] = &e[0]
        output_data[0] = &q_out[0]
        self.thisptr.run(input_data, output_data, p.size)
        return q_out
    def get_name(self):
        return 'gr4h'
    def get_ptr(self):
        return <unsigned long long>self.thisptr.get_ptr()
    def get_param_nb(self):
        return self.thisptr.get_param_nb()
    def get_state(self):
        cdef int state_nb = self.thisptr.get_state_nb()
        cdef np.ndarray[double, ndim = 1, mode = 'c'] state_copy = np.empty((state_nb))
        self.thisptr.copy_state(self.thisptr.get_state(), &state_copy[0])
        return state_copy
    def set_state(self, _state):
        cdef np.ndarray[double, ndim = 1, mode = 'c'] state = np.array(_state)
        self.thisptr.set_state(&state[0])
    def set_conf(self, _conf):
        self.thisptr.set_conf(_conf)
