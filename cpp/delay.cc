#include "delay.hh"

delay::delay(double *_param, double *_init_state, void* _mem_pool):model("delay")
{
    param_nb = 1;
    d = _param[0];
    state_nb = 0;
    state_byte_nb = state_nb * sizeof(double);
    mem_pool_byte_nb = sizeof(double) * state_nb;
    if (_mem_pool == NULL)
    {
        mem_pool_is_here = true;
        delay_mem_pool = malloc(mem_pool_byte_nb);
    }
    else
    {
        mem_pool_is_here = false;
        delay_mem_pool = _mem_pool;
    }
    mem_pool = delay_mem_pool;
    state = assign_mem(state_byte_nb);
    for (int i = 0; i < state_nb; i++)
    {
        if (_init_state == NULL)
            ((double*)state)[i] = 0.;
        else
            ((double*)state)[i] = _init_state[i];
    }
    floor_d = (int)floor(d);
    dc = d - (double)floor_d;
    set_state(state);
    // parameter packing:
    //param_byte_nb = sizeof(double) * param_nb + sizeof(int) * 4 + sizeof(double) * (l + m);
    param_byte_nb = sizeof(double) * param_nb;
    param = malloc(param_byte_nb);
    void* this_ptr = param;
    *(double*)this_ptr = d;
    this_ptr = &((double*)this_ptr)[1];
    *(int*)this_ptr = state_nb;
    //this_ptr = &((int*)this_ptr)[1];
    this_ptr = &((double*)this_ptr)[1];
    *(int*)this_ptr = state_byte_nb;
}

delay::~delay()
{
    free(param);
    if (mem_pool_is_here)
        free(delay_mem_pool);
}

void delay::run(void *_input_data, void *_output_data, int _step_nb)
{
    double* q_in  = ((double**)_input_data)[0];
    double* q_out = ((double**)_output_data)[0];
    double q_prev = dc * q_in[_step_nb - 1 - floor_d];
    for (int step_i = _step_nb - 1; step_i >= 0; step_i --)
    {
        if ((step_i - floor_d) >= 0) {
            q_out[step_i] = (1 - dc) * q_in[step_i - floor_d] + q_prev;
            q_prev = dc * q_in[step_i - floor_d];
        }
        else
            q_out[step_i] = 0.;
    }
}

void delay::clear_state(void *_state)
{
    for (int i = 0; i < state_nb; i++)
        (&(((double*)state)[0]))[i] = 0.;
}

void delay::set_state(void *_state)
{
}

void* delay::get_state()
{
    return state;
}

void delay::copy_input_data(void* _src, void* _dst)
{
    ((double*)_dst)[0] = ((double*)_src)[0]; // Qin
}

model* delay::get_new()
{
    double params[1];
    params[0] = d;
    return new delay(params, NULL, NULL);
}
