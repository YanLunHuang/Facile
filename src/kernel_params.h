#include "ap_fixed.h"
#include "parameters.h"

//how many consecutive sets of inputs to run over per kernel execution
#define BatchSize 16384
#define COMPRESSION 32

#define DATA_SIZE_IN N_INPUT_1_1 //18
#define DATA_SIZE_OUT N_LAYER_11 //1

typedef ap_fixed<16,6> data_t;

struct group_in{
    data_t layer[COMPRESSION];
};
struct group_out{
    data_t layer[COMPRESSION];
};