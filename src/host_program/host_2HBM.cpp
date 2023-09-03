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

#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
typedef std::chrono::high_resolution_clock Clock;

#include "xcl2.hpp"
#include <vector>
#include "kernel_params.h"

#define STRINGIFY2(var) #var
#define STRINGIFY(var) STRINGIFY2(var)

static uint64_t get_duration_ns(const cl::Event & events) {
    uint64_t duration = 0;
    //for (size_t i=0; i<events.size(); i++) {
        uint64_t start, end;
        events.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &start);
        events.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &end);
        duration += end - start;
    //}
    return duration;
}

int main(int argc, char** argv)
{
    std::cout<<"================host_2HBM.cpp================"<<std::endl;
    std::cout<<"Ping-Pong transmission"<<std::endl;

    int CU_num=4;
    int num_buf = 5;
    int event_num = 1000;
    std::string print_or_not = "true";
    std::string detailed_info = "false";
    cl_int err;
    std::string datadir = STRINGIFY(HLS4ML_DATA_DIR);
    std::string xclbinFilename = "";

    if (argc > 1) xclbinFilename = argv[1];
    if (argc > 2) CU_num = atoi(argv[2]);
    if (argc > 3) num_buf = atoi(argv[3]);
    if (argc > 4) event_num = atoi(argv[4]);
    if (argc > 5) print_or_not = argv[5];
    if (argc > 6) detailed_info = argv[6];
    if (argc > 7) datadir = argv[7];
    std::cout <<"CU number: "<<CU_num<<std::endl;
    std::cout <<"Buffer number: "<<num_buf<<std::endl;
    std::cout <<"Event number: " << event_num << " time(s) \n";
    std::cout <<"Choose to print out the prediction: " << print_or_not<<std::endl;
    std::cout <<"Choose to print out the detailed info: " << detailed_info<<std::endl;
    std::cout <<"using "<< datadir << " to get input/output data \n";
    std::cout <<"============================================="<<std::endl;

    if(CU_num>=5){
        std::cout << "Kernel number has to be lower than 4, exit!\n";
        exit(EXIT_FAILURE);
    }
//=====================================================
//Find device & Load xclbin file & Program device
//=====================================================

    // OPENCL HOST CODE AREA START
    // get_xil_devices() is a utility API which will find the xilinx
    // platforms and will return list of devices connected to Xilinx platform
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    cl::Context context(device);
    std::vector<cl::CommandQueue>vec_Queue;

    for(int N=0;N<CU_num;N++){
        cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
        vec_Queue.push_back(q);
    }

    std::string device_name = device.getInfo<CL_DEVICE_NAME>(); 
    std::cout << "Found Device=" << device_name.c_str() << std::endl;
    
    cl::Program::Binaries bins;
    // Load xclbin
    std::cout << "Loading: '" << xclbinFilename << "'\n";
    std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
    bin_file.seekg (0, bin_file.end);
    unsigned nb = bin_file.tellg();
    bin_file.seekg (0, bin_file.beg);
    char *buf = new char [nb];
    bin_file.read(buf, nb);
    // Create Program from Binary File
    bins.push_back({buf,nb});
    
    // Program the device
    bool valid_device = false;
    cl::Program program(context, {device}, bins, nullptr, &err);
    std::vector<cl::Kernel>vec_Kernel;
    if (err != CL_SUCCESS) {
        std::cout << "Failed to program device with xclbin file!\n";
    }else {
        std::cout <<"program successful!\n";
        
        for(int N=0;N<CU_num;N++){
            cl::Kernel alveo_hls4ml;
            for (int i=0;i<num_buf;i++){
                if(N==0)alveo_hls4ml = cl::Kernel(program,"alveo_hls4ml:{alveo_hls4ml_1}");
                else if(N==1)alveo_hls4ml = cl::Kernel(program,"alveo_hls4ml:{alveo_hls4ml_2}");
                else if(N==2)alveo_hls4ml = cl::Kernel(program,"alveo_hls4ml:{alveo_hls4ml_3}");
                else alveo_hls4ml = cl::Kernel(program,"alveo_hls4ml:{alveo_hls4ml_4}");
                vec_Kernel.push_back(alveo_hls4ml);
            }
        }
        valid_device = true;
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

//==================================
//Launch the input/output source 
//              &
//        set HBM banks
//==================================

    //input or output HBM number per CU
    const int HBM_num = 2;

    size_t vector_size_in_bytes = sizeof(group_in) * BatchSize;
    size_t vector_size_out_bytes = sizeof(group_out) * (BatchSize/COMPRESSION);

    std::vector<group_in,aligned_allocator<group_in>> source_in(BatchSize);
    std::vector<group_out,aligned_allocator<group_out>> source_hw_results(BatchSize/COMPRESSION);

    std::vector< std::vector<group_in,aligned_allocator<group_in>> >vec_in;
    std::vector< std::vector<group_out,aligned_allocator<group_out>> >vec_output;

    cl_mem_ext_ptr_t pipo_buffer_in[4*num_buf];
    cl_mem_ext_ptr_t pipo_buffer_output[4*num_buf];

    //Reset the input data
    int hbm_add = 0;
    for(int N=0;N<num_buf*CU_num;N++){
        if(N%num_buf==0 && N!=0)hbm_add += 4;

        vec_in.push_back(source_in);
        if(N%HBM_num==0)pipo_buffer_in[N].flags = (0+hbm_add) | XCL_MEM_TOPOLOGY;
        else pipo_buffer_in[N].flags = (1+hbm_add) | XCL_MEM_TOPOLOGY;

        if(N%HBM_num==0)std::cout << "hbm : " << (0+hbm_add)<<std::endl;
        else std::cout << "hbm : " << (1+hbm_add)<<std::endl;
        pipo_buffer_in[N].param = 0;
        pipo_buffer_in[N].obj   = vec_in[N].data();

        std::cout << "Reset the input data1 " << std::endl;
        for(int i0 = 0; i0 < BatchSize; i0++) {
            for(int i1 = 0; i1 < COMPRESSION; i1++) {
                vec_in[N][i0].layer[i1] = 1;
            }
        }
    }

    //Reset the output result
    hbm_add = 0;
    for(int N=0;N<num_buf*CU_num;N++){
        if(N%num_buf==0 && N!=0)hbm_add += 4;

        vec_output.push_back(source_hw_results);
        if(N%HBM_num==0)pipo_buffer_output[N].flags = (2+hbm_add) | XCL_MEM_TOPOLOGY;
        else pipo_buffer_output[N].flags = (3+hbm_add) | XCL_MEM_TOPOLOGY;

        if(N%HBM_num==0)std::cout << "hbm : " << (2+hbm_add)<<std::endl;
        else std::cout << "hbm : " << (3+hbm_add)<<std::endl;
        pipo_buffer_output[N].param = 0;
        pipo_buffer_output[N].obj   = vec_output[N].data();

        std::cout << "Reset the output result " << std::endl;
        for(int i0 = 0 ; i0 < BatchSize/COMPRESSION; i0++){
            for(int i1 = 0 ; i1 < COMPRESSION; i1++){
                vec_output[N][i0].layer[i1] = 1;
            }
        }
    }

//================================
//Create buffer & Set Argument
//================================

    // Allocate Buffer in Global Memory
    // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and 
    // Device-to-host communication
    std::vector<cl::Buffer>vec_in_buf;
    std::vector<cl::Buffer>vec_out_buf;

    for(int N=0;N<num_buf*CU_num;N++){
        cl::Buffer buffer_in   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
                vector_size_in_bytes, &pipo_buffer_in[N]);
        cl::Buffer buffer_output(context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX, 
                vector_size_out_bytes, &pipo_buffer_output[N]);

        vec_in_buf.push_back(buffer_in);
        vec_out_buf.push_back(buffer_output);
    }

    for(int N=0;N<num_buf*CU_num;N++){
        vec_Kernel[N].setArg(0, vec_in_buf[N]);
        vec_Kernel[N].setArg(1, vec_out_buf[N]);
    }

//=====================
//Input
//=====================

    // Load input data from text file
    std::ifstream fin(datadir+"/tb_input_features.dat");
    // Load predictions from text file
    std::ifstream fpr(datadir+"/tb_output_predictions.dat");
    // Open output file
    std::ofstream fout;
    fout.open("tb_output_data.log");
    
    std::string iline;
    std::string pline;
    
    int exp_times = 0;

    // Flag for success/failure of finding data files
    if (!(fin.is_open()) || !(fpr.is_open())) {
        std::cout << "Unable to open input/predictions file, using random input" << std::endl;
        exit(EXIT_FAILURE);
    }
    else std::cout <<"successfully open input and output file"<<std::endl;
    
    // Get inputs/predictions from files
    if(fin.is_open() && fpr.is_open()){
      while(std::getline(fin,iline) && std::getline(fpr,pline)) {
        
        std::cout << "Processing event " << exp_times << std::endl;
        fout << "Processing event " << exp_times << "\n";
        exp_times++;
        
        // Here is input.
        char* cstr=const_cast<char*>(iline.c_str());
        char* current;
        std::vector<float> in;
        current=strtok(cstr," ");
        while(current!=NULL){
            in.push_back(atof(current));
            current=strtok(NULL," ");
        }
        
        //Here is the corresponding output(correct one)
        cstr=const_cast<char*>(pline.c_str());
        std::vector<float> pr;
        current=strtok(cstr," ");
        while(current!=NULL){
            pr.push_back(atof(current));
            current=strtok(NULL," ");
        }
        //Send into buffer
        for(int N = 0; N < num_buf*CU_num; N++) {
          std::cout << "Send input data into buffer " << std::endl;
          for(int i0 = 0; i0 < BatchSize; i0++) {
              for(int i1 = 0; i1 < 18; i1++) {
                  vec_in[N][i0].layer[i1] = in[i0*18+i1];
              }
          }
        }

//========================
//Start to run on FPGA
//========================
        auto t1 = Clock::now();
        auto t2 = Clock::now();
        uint64_t total_time = 0;
        uint64_t duration = 0;

        std::vector<cl::Event> Write,Execution,Read;
        std::vector<std::vector<cl::Event>> vec_Write,vec_Execution,vec_Read;

        for(int N=0; N<CU_num; N++){
            vec_Write.push_back(Write);
            vec_Execution.push_back(Execution);
            vec_Read.push_back(Read);
        }

        std::cout << "start computation" << std::endl;
        for(int e=0;e<event_num;e++){
            t1 = Clock::now();
            for(int i=0; i<num_buf; i++){
                for(int N=0; N<CU_num; N++){
                    cl::Event Task_W,Task_K,Task_R;

                    vec_Queue[N].enqueueMigrateMemObjects({vec_in_buf[i+num_buf*N]},0/* 0 means from host*/,NULL,&Task_W);
                    vec_Write[N].push_back(Task_W);
                    vec_Queue[N].enqueueTask(vec_Kernel[i+num_buf*N],&(vec_Write[N]),&Task_K);
                    vec_Execution[N].push_back(Task_K);
                    vec_Queue[N].enqueueMigrateMemObjects({vec_out_buf[i+num_buf*N]},CL_MIGRATE_MEM_OBJECT_HOST,&(vec_Execution[N]),&Task_R);
                    vec_Read[N].push_back(Task_R);
                }
            }
            //Waiting for the finish of commandqueue
            for(int N=0;N<CU_num;N++){
                vec_Queue[N].finish();
            }

            t2 = Clock::now();

            duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count(); 
            total_time += duration;

            for(int N=0; N<CU_num; N++){
                vec_Write[N].clear();
                vec_Execution[N].clear();
                vec_Read[N].clear();
            }

//================================
//Check the computatation process
//================================
            if(detailed_info == "true"){
                fout << "Event: " << e << std::endl;
                fout << "FPGA time: " << duration << " ns" <<std::endl;
                fout << "current_total: " << total_time << " ns" << std::endl;


                fout << "prediction: " << std::endl; 
                for(int N=0;N<num_buf*CU_num;N++){
                    fout << "buffer"<<N<< ": \n";
                    for(int j=0;j<COMPRESSION;j++){
                        fout << vec_output[N][50].layer[j] << " "; 
                    }
                    fout << "\n";
                }
                fout << "\n"; 
                fout << "========================================="; 
                if(e!=event_num-1){
                    fout << "reset \n"; 
                    for(int N=0;N<num_buf*CU_num;N++){
                        for(int j=0;j<COMPRESSION;j++){
                            vec_output[N][50].layer[j] = 1; 
                        }
                    }
                    for(int N=0;N<num_buf*CU_num;N++){
                        fout << "buffer"<<N<<": \n"; 
                        for(int j=0;j<COMPRESSION;j++){
                            fout << vec_output[N][50].layer[j] << " "; 
                        }
                        fout << "\n";
                    }
                }
                fout << "\n";
            }
        }
        //print timing
        fout << "========================================= \n";
        std::cout << "FPGA average time: " << total_time/(event_num*num_buf*CU_num) << " ns/event" << std::endl;
        fout << "FPGA average time: " << total_time/(event_num*num_buf*CU_num) << " ns/event \n";

        std::cout << "FPGA average throughput: " << pow(10,9)/(total_time/(event_num*num_buf*CU_num)) << " events/s" << std::endl;
        fout << "FPGA average throughput: " << pow(10,9)/(total_time/(event_num*num_buf*CU_num)) << " events/s \n";
//=====================
//Output result
//=====================

        fout <<"Predictions:  \n";
        for(int i=0;i<BatchSize ;i++){
            fout << pr[i] << " ";
        }
        fout<<"\n";

        fout <<"Quantized predictions: \n";

        for(int N=0;N<CU_num;N++){
            for(int i=0;i<num_buf;i++){
                if(N==0)fout <<"CU1 buf"<<i<<": \n";
                else if(N==1)fout <<"CU2 buf"<<i<<": \n";
                else if(N==2)fout <<"CU3 buf"<<i<<": \n";
                else fout <<"CU4 buf"<<i<<": \n";
            
                for(int i=0;i<BatchSize/COMPRESSION ;i++){
                    for(int j=0;j<COMPRESSION ;j++){
                        fout << vec_output[N][i].layer[j] << " "; 
                    }
                }
                fout << "\n\n";
            }
        }
        std::cout<<"---- END EVENT "<<" ----"<<std::endl;
        std::cout<<"All result has been saved in tb_output_data.log"<<std::endl;

      }
    }

// OPENCL HOST CODE AREA END
    fin.close();
    fpr.close();
    fout.close();

    return EXIT_SUCCESS;
}
