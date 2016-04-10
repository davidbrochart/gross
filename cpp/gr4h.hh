#ifndef gr4h_hh
#define gr4h_hh

#include <stdlib.h>
#include "model.hh"

class gr4h:public model
{
    private:
        double sh1(double t);
        double sh2(double t);
        double uh1(double j);
        double uh2(double j);
        double x1, x2, x3, x4;
        double s, r, *pr_prev;
        double *uh1_tab, *uh2_tab;
        int l, m;
        bool mem_pool_is_here;
        void* gr4h_mem_pool;

    public:
        gr4h(double *_param, double *_init_state, void* _mem_pool = NULL);
        ~gr4h();
        void clear_state(void *_state);
        void set_state(void *_state = NULL);
        void set_conf(int _conf);
        void* get_state();
        void copy_input_data(void* _src, void* _dst);
        void run(void *_input_data, void *_output_data, int _step_nb = 1);
        model* get_new();       
};

#endif
