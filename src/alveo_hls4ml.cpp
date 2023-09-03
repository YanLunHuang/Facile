/**********
Copyright (c) 2018, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/

/*******************************************************************************
Description:
    HLS pragmas can be used to optimize the design : improve throughput, reduce latency and 
    device resource utilization of the resulting RTL code
    This is a wrapper to be used with an hls4ml project to enable proper handling by SDAccel
*******************************************************************************/
#include <iostream>
#include "kernel_params.h"
#include "ereg_v1.h"

extern "C" {

void alveo_hls4ml(
    const group_in *in, // Read-Only Vector
    group_out *out       // Output Result
    )
{
    #pragma HLS INTERFACE m_axi port=in  offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem1
    #pragma HLS INTERFACE s_axilite port=in   bundle=control
    #pragma HLS INTERFACE s_axilite port=out  bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    #pragma HLS data_pack variable=in
    #pragma HLS data_pack variable=out

    input_t in_buf[BatchSize][DATA_SIZE_IN];
    #pragma HLS ARRAY_PARTITION variable=in_buf complete dim=2

    layer11_t out_buf[BatchSize][DATA_SIZE_OUT];
    #pragma HLS ARRAY_PARTITION variable=out_buf complete dim=2

    #pragma HLS DATAFLOW
    //=============================================
    //Input
    //=============================================
    Loop1: for (int i = 0; i < BatchSize; i++) {
        #pragma HLS PIPELINE
        Loop1_1: for (int j = 0; j < DATA_SIZE_IN; j++) {
            #pragma HLS UNROLL
            in_buf[i][j] = in[i].layer[j];
        }
    }

    //=============================================
    //Start computation
    //=============================================
    Loop3: for (int i = 0; i < BatchSize; i++) {
        #pragma HLS DATAFLOW
        //if(i==0)std::cout<<"inf start"<<std::endl;
        ereg_v1(in_buf[i],out_buf[i]);
        //if(i==BatchSize-1)std::cout<<"inf end"<<std::endl;
    }

    //=============================================
    //Output
    //=============================================
    Loop4: for (int i = 0; i < BatchSize/COMPRESSION; i++) {
        #pragma HLS PIPELINE
        Loop4_1: for(int j = 0; j < COMPRESSION; j++) {
            Loop4_2: for(int k = 0; k < DATA_SIZE_OUT; k++) {
                #pragma HLS UNROLL
                data_t tmp_small = out_buf[i*COMPRESSION+j][k];
                out[i].layer[j] = tmp_small;
            }
        }
    }

}
}
