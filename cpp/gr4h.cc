#include "gr4h.hh"

gr4h::gr4h(double *_param, double *_init_state, void* _mem_pool):model("gr4h")
{
    param_nb = 5;
    x1 = _param[0];
    x2 = _param[1];
    x3 = _param[2];
    x4 = _param[3];
    l = ((int)x4) + 1;
    m = ((int)(2. * x4)) + 1;
    state_nb = 2 + (int)(2. * x4);
    state_byte_nb = state_nb * sizeof(double);
    mem_pool_byte_nb = sizeof(double) * (state_nb + (int)(2. * x4) + l + m);
    if (_mem_pool == NULL)
    {
        mem_pool_is_here = true;
        gr4h_mem_pool = malloc(mem_pool_byte_nb);
    }
    else
    {
        mem_pool_is_here = false;
        gr4h_mem_pool = _mem_pool;
    }
    mem_pool = gr4h_mem_pool;
    state = assign_mem(state_byte_nb);
    for (int i = 0; i < state_nb; i++)
    {
        if (_init_state == NULL)
            ((double*)state)[i] = 0.;
        else
            ((double*)state)[i] = _init_state[i];
    }
    if ((int)(2. * x4) > 0)
        pr_prev = (double*)assign_mem(((int)(2. * x4)) * sizeof(double));
    set_state(state);
    uh1_tab = (double*)assign_mem(l * sizeof(double));
    uh2_tab = (double*)assign_mem(m * sizeof(double));
    for (int i = 0; i < m; i++)
    {
        if (i < l)
            uh1_tab[i] = uh1(i + 1);
        uh2_tab[i] = uh2(i + 1);
    }
    // parameter packing:
    //param_byte_nb = sizeof(double) * param_nb + sizeof(int) * 4 + sizeof(double) * (l + m);
    param_byte_nb = sizeof(double) * param_nb + sizeof(double) * 5 + sizeof(double) * (l + m);
    param = malloc(param_byte_nb);
    void* this_ptr = param;
    *(double*)this_ptr = x1;
    this_ptr = &((double*)this_ptr)[1];
    *(double*)this_ptr = x2;
    this_ptr = &((double*)this_ptr)[1];
    *(double*)this_ptr = x3;
    this_ptr = &((double*)this_ptr)[1];
    *(double*)this_ptr = x4;
    this_ptr = &((double*)this_ptr)[1];
    *(int*)this_ptr = state_nb;
    //this_ptr = &((int*)this_ptr)[1];
    this_ptr = &((double*)this_ptr)[1];
    *(int*)this_ptr = state_byte_nb;
    //this_ptr = &((int*)this_ptr)[1];
    this_ptr = &((double*)this_ptr)[1];
    *(int*)this_ptr = l;
    //this_ptr = &((int*)this_ptr)[1];
    this_ptr = &((double*)this_ptr)[1];
    *(int*)this_ptr = m;
    //this_ptr = &((int*)this_ptr)[1];
    this_ptr = &((double*)this_ptr)[1];
    for (int i = 0; i < l; i++)
    {
        *(double*)this_ptr = uh1_tab[i];
        this_ptr = &((double*)this_ptr)[1];
    }
    for (int i = 0; i < m; i++)
    {
        *(double*)this_ptr = uh2_tab[i];
        this_ptr = &((double*)this_ptr)[1];
    }
}

gr4h::~gr4h()
{
    free(param);
    if (mem_pool_is_here)
        free(gr4h_mem_pool);
}

double gr4h::sh1(double t)
{
    double res;
    if (t == 0.)
        res = 0.;
    else if (t < x4)
        res = pow(t / x4, 1.25);
    else
        res = 1.;
    return res;
}

double gr4h::sh2(double t)
{
    double res;
    if (t == 0.)
        res = 0.;
    else if (t < x4)
        res = 0.5 * pow(t / x4, 1.25);
    else if (t < 2. * x4)
        res = 1. - 0.5 * pow(2. - t / x4, 1.25);
    else
        res = 1.;
    return res;
}

double gr4h::uh1(double j)
{
    return sh1(j) - sh1(j - 1.);
}

double gr4h::uh2(double j)
{
    return sh2(j) - sh2(j - 1.);
}

void gr4h::run(void *_input_data, void *_output_data, int _step_nb)
{
    double* p = ((double**)_input_data)[0];
    double* e = ((double**)_input_data)[1];
    double* q_in  = ((double**)_input_data)[2];
    double* q_out = ((double**)_output_data)[0];
    for (int step_i = 0; step_i < _step_nb; step_i ++)
    {
        double tmp;
        // reservoir de production:
        double pn, ps;
        if (*p > *e)
        {
            pn = *p - *e;
            ps = x1 * (1. - (s / x1) * (s / x1)) * tanh(pn / x1) / (1. + (s / x1) * tanh(pn / x1));
            s += ps;
        }
        else if (*p < *e)
        {
            ps = 0.;
            pn = 0.;
            double en = *e - *p;
            double es = s * (2. - s / x1) * tanh(en / x1) / (1. + (1. - s / x1) * tanh(en / x1));
            tmp = s - es;
            if (tmp > 0.)
                s = tmp;
            else
                s = 0.;
        }
        else
        {
            pn = 0.;
            ps = 0.;
        }
        // percolation:
        tmp = s / (5.25 * x1);
        double perc = s * (1. - pow(1. + tmp * tmp * tmp * tmp, -1. / 4.));
        s -= perc;
        // hydrogrammes:
        double pr_0 = perc + pn - ps;
        double q9 = 0.;
        double q1 = 0.;
        for (int k = 0; k < m; k++)
        {
            double pr_k;
            if (k == 0)
                pr_k = pr_0;
            else
                pr_k = pr_prev[k - 1];
            if (k < l)
                q9 += uh1_tab[k] * pr_k;
            q1 += uh2_tab[k] * pr_k;
        }
        q9 *= 0.9;
        q1 *= 0.1;
        // echange souterrain:
        double f = x2 * pow(r / x3, 7. / 2.);
        // reservoir de routage:
        tmp = r + q9 + f;
        if (tmp > 0.)
            r = tmp;
        else
            r = 0.;
        tmp = r / x3;
        double qr = r * (1. - pow(1. + tmp * tmp * tmp * tmp,  -1. / 4.));
        r -= qr;
        tmp = q1 + f;
        double qd;
        if (tmp > 0.)
            qd = tmp;
        else
            qd = 0.;
        *q_out = qr + qd;
        for (int i = (int)(2. * x4) - 2; i >= 0; i--)
            pr_prev[i + 1] = pr_prev[i];
        if ((int)(2. * x4) > 0)
            pr_prev[0] = pr_0;
        p ++;
        e ++;
        q_in  ++;
        q_out ++;
    }
}

void gr4h::clear_state(void *_state)
{
    ((double*)state)[0] = 0.;
    ((double*)state)[1] = 0.;
    for (int i = 0; i < state_nb - 2; i++)
        (&(((double*)state)[2]))[i] = 0.;
}

void gr4h::set_state(void *_state)
{
    s = ((double*)_state)[0];
    r = ((double*)_state)[1];
    for (int i = 0; i < (int)(2. * x4); i++)
        pr_prev[i] = (&(((double*)_state)[2]))[i];
}

void gr4h::set_conf(int _conf)
{
    switch (_conf)
    {
        case 0:
            s = 0.;
            r = 0.;
            for (int i = 0; i < (int)(2. * x4); i++)
                pr_prev[i] = 0.;
            break;
        case 1:
            s = x1 / 2.;
            r = x3 / 2.;
            for (int i = 0; i < (int)(2. * x4); i++)
                pr_prev[i] = 0.;
            break;
        default:
            break;
    }
}

void* gr4h::get_state()
{
    ((double*)state)[0] = s;
    ((double*)state)[1] = r;
    for (int i = 0; i < (int)(2. * x4); i++)
        (&(((double*)state)[2]))[i] = pr_prev[i];
    return state;
}

void gr4h::copy_input_data(void* _src, void* _dst)
{
    ((double*)_dst)[0] = ((double*)_src)[0]; // P
    ((double*)_dst)[1] = ((double*)_src)[1]; // E
    ((double*)_dst)[2] = ((double*)_src)[2]; // Qin
}

model* gr4h::get_new()
{
    double params[4];
    params[0] = x1;
    params[1] = x2;
    params[2] = x3;
    params[3] = x4;
    return new gr4h(params, NULL, NULL);
}
