#ifndef delay_hh
#define delay_hh

#include <stdlib.h>
#include "model.hh"

class delay:public model
{
    private:
        double d, dc;
        int floor_d;
        bool mem_pool_is_here;
        void* delay_mem_pool;

    public:
        delay(double *_param, double *_init_state, void* _mem_pool = NULL);
        ~delay();
        void clear_state(void *_state);
        void set_state(void *_state = NULL);
        void* get_state();
        void copy_input_data(void* _src, void* _dst);
        void run(void *_input_data, void *_output_data, int _step_nb = 1);
        model* get_new();       
};

#endif
