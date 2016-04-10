#include "model.hh"

model::model(char* _name)
{
    state_nb = 0;
    param_nb = 0;
    state_byte_nb = 0;
    mem_pool_i = 0;
    name = _name;
}

void* model::alloc_state()
{
    void *_state = malloc(get_state_byte_nb());
    clear_state(_state);
    return _state;
}

void* model::assign_mem(int _byte_nb)
{
    void* res = &((char*)mem_pool)[mem_pool_i];
    mem_pool_i += _byte_nb;
    return res;
}

void model::dealloc_state(void* _state)
{
    free(_state);
}

void model::copy_state(void* _src, void* _dst)
{
    memcpy(_dst, _src, state_byte_nb);
}

void* model::get_new()
{
    cerr << "get_new() not implemented!" << endl;
    exit(1);
}
