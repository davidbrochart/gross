cdef extern from 'gr4h.hh':
    cdef cppclass gr4h:
        gr4h(double*, double*) except +
        void set_state(double*)
        void set_conf(int)
        void run(double**, double**, int)
        void* get_ptr()
        int get_state_nb()
        double* get_state()
        int get_param_nb()
        void copy_state(double* _src, double* _dst)

cdef extern from 'delay.hh':
    cdef cppclass delay:
        delay(double*, double*) except +
        void set_state(double*)
        void set_conf(int)
        void run(double**, double**, int)
        void* get_ptr()
        int get_state_nb()
        double* get_state()
        int get_param_nb()
        void copy_state(double* _src, double* _dst)
