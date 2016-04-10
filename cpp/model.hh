#ifndef model_hh
#define model_hh

#include <string.h>
#include <cstddef>
#include <iostream>
#include <math.h>
#include <stdio.h>

using namespace std;

class model
{
    public:
        int param_nb;
        int param_byte_nb;
        int state_nb;
        int state_byte_nb;
        void* mem_pool;
        int mem_pool_i;
        int mem_pool_byte_nb;
        char* name;

        void *param;
        void *state;

        model(char* _name);
        virtual ~model() {};
        virtual void set_state(void *_state) {};
        virtual void clear_state(void *_state) {};
        virtual void* get_state() {return NULL;};
        virtual void* alloc_state();
        virtual void* assign_mem(int _byte_nb);
        virtual void dealloc_state(void* _state);
        virtual int get_state_byte_nb() {return state_byte_nb;};
        virtual void copy_state(void* _src, void* _dst);
        virtual void copy_input_data(void* _src, void* _dst) {};
        virtual void run(void *_input_data, void *_output_data, int _step_nb = 1) {};
        virtual int get_state_nb() {return state_nb;};
        virtual int get_param_nb() {return param_nb;};
        virtual void* get_ptr() {return this;};
        virtual void* get_new();
};

#endif
