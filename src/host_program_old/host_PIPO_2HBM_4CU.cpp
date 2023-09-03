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
    std::cout<<"=======host_PIPO_2HBM_4CU.cpp========"<<std::endl;
    std::cout<<"4CU"<<std::endl;
    std::cout<<"N normal buffers"<<std::endl;
    std::cout<<"Ping-Pong transmission"<<std::endl;
    std::cout<<"====================================="<<std::endl;
    int event_num = 100;
    int num_buf = 1;
    cl_int err;
    std::string datadir = STRINGIFY(HLS4ML_DATA_DIR);
    std::string xclbinFilename = "";
    std::string print_or_not = "true";
    if (argc > 1) xclbinFilename = argv[1];
    if (argc > 2) event_num = atoi(argv[2]);
    if (argc > 3) num_buf = atoi(argv[3]);
    if (argc > 4) print_or_not = argv[4];
    if (argc > 5) datadir = argv[5];
    std::cout << "Will run " << event_num << " time(s), using " << datadir << " to get input features and output predictions (tb_input_features.dat and tb_output_predictions.dat)" << std::endl;
    std::cout << "Choose to print out the prediction: " << print_or_not <<std::endl;
    std::cout << "Using " << datadir << " to get input features and output predictions (tb_input_features.dat and tb_output_predictions.dat)" << std::endl;

//=====================================================
//Find device & Load xclbin file & Program device
//=====================================================

    // OPENCL HOST CODE AREA START
    // get_xil_devices() is a utility API which will find the xilinx
    // platforms and will return list of devices connected to Xilinx platform
    cl::Kernel alveo_hls4ml_1,alveo_hls4ml_2,alveo_hls4ml_3,alveo_hls4ml_4;
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    cl::Context context(device);
    cl::CommandQueue q1(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
    cl::CommandQueue q2(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
    cl::CommandQueue q3(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
    cl::CommandQueue q4(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
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
    if (err != CL_SUCCESS) {
        std::cout << "Failed to program device with xclbin file!\n";
    }else {
        std::cout <<"program successful!\n";
        
        std::string cu_id = std::to_string(1);
        std::string krnl_name_full = "alveo_hls4ml";
        alveo_hls4ml_1 = cl::Kernel(program,"alveo_hls4ml:{alveo_hls4ml_1}");
        alveo_hls4ml_2 = cl::Kernel(program,"alveo_hls4ml:{alveo_hls4ml_2}");
        alveo_hls4ml_3 = cl::Kernel(program,"alveo_hls4ml:{alveo_hls4ml_3}");
        alveo_hls4ml_4 = cl::Kernel(program,"alveo_hls4ml:{alveo_hls4ml_4}");
        valid_device = true;
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

//=====================
//Create buffer
//=====================

    size_t vector_size_in_bytes = sizeof(group_in) * BatchSize;
    size_t vector_size_out_bytes = sizeof(group_out) * (BatchSize/COMPRESSION);

    std::vector<group_in,aligned_allocator<group_in>> source_in(BatchSize);
    std::vector<group_out,aligned_allocator<group_out>> source_hw_results(BatchSize/COMPRESSION);

    std::vector< std::vector<group_in,aligned_allocator<group_in>> >vec_in;
    std::vector< std::vector<group_out,aligned_allocator<group_out>> >vec_output;

    cl_mem_ext_ptr_t pipo_buffer_in[4*num_buf];
    cl_mem_ext_ptr_t pipo_buffer_output[4*num_buf];

    std::vector<cl::Buffer>vec_in_buf;
    std::vector<cl::Buffer>vec_out_buf;

    //Reset the input data
    int hbm_add = 0;
    for(int N=0;N<num_buf*4;N++){
        if(N%num_buf==0 && N!=0)hbm_add += 4;

        vec_in.push_back(source_in);
        if(N%2==0)pipo_buffer_in[N].flags = (0+hbm_add) | XCL_MEM_TOPOLOGY;
        else pipo_buffer_in[N].flags = (1+hbm_add) | XCL_MEM_TOPOLOGY;
        if(N%2==0)std::cout << "hbm : " << (0+hbm_add)<<std::endl;
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
    for(int N=0;N<num_buf*4;N++){
        if(N%num_buf==0 && N!=0)hbm_add += 4;

        vec_output.push_back(source_hw_results);
        if(N%2==0)pipo_buffer_output[N].flags = (2+hbm_add) | XCL_MEM_TOPOLOGY;
        else pipo_buffer_output[N].flags = (3+hbm_add) | XCL_MEM_TOPOLOGY;
        if(N%2==0)std::cout << "hbm : " << (2+hbm_add)<<std::endl;
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

    // Allocate Buffer in Global Memory
    // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and 
    // Device-to-host communication
    for(int N=0;N<num_buf*4;N++){
        cl::Buffer buffer_in   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
                vector_size_in_bytes, &pipo_buffer_in[N]);
        cl::Buffer buffer_output(context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX, 
                vector_size_out_bytes, &pipo_buffer_output[N]);

        vec_in_buf.push_back(buffer_in);
        vec_out_buf.push_back(buffer_output);
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
    fout.open("tb_output_data.dat");
    
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
        for(int N = 0; N < num_buf*4; N++) {
          std::cout << "Send into buffer " << std::endl;
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
    uint64_t current_total = 0;

    std::vector<cl::Event> Read1,Read2,Read3,Read4;
    std::vector<cl::Event> Kernel1,Kernel2,Kernel3,Kernel4;
    std::vector<cl::Event> Write1,Write2,Write3,Write4;
    cl::Event Task1_R,Task1_K,Task1_W;
    cl::Event Task2_R,Task2_K,Task2_W;
    cl::Event Task3_R,Task3_K,Task3_W;
    cl::Event Task4_R,Task4_K,Task4_W;

    std::cout << "start computation" << std::endl;
    for(int i=0;i<event_num;i++){
        t1 = Clock::now();
        for(int N=0; N<num_buf; N++){
            //vec_in_buf[0]~vec_in_buf[N-1] -> for CU1
            alveo_hls4ml_1.setArg(0, vec_in_buf[N]);
            alveo_hls4ml_1.setArg(1, vec_out_buf[N]);
            q1.enqueueMigrateMemObjects({vec_in_buf[N]},0/* 0 means from host*/,NULL,&Task1_R);
            Read1.push_back(Task1_R);
            q1.enqueueTask(alveo_hls4ml_1,&Read1,&Task1_K);
            Kernel1.push_back(Task1_K);
            q1.enqueueMigrateMemObjects({vec_out_buf[N]},CL_MIGRATE_MEM_OBJECT_HOST,&Kernel1,&Task1_W);
            Write1.push_back(Task1_W);

            //vec_in_buf[N]~vec_in_buf[2N-1] -> for CU2
            alveo_hls4ml_2.setArg(0, vec_in_buf[num_buf+N]);
            alveo_hls4ml_2.setArg(1, vec_out_buf[num_buf+N]);
            q2.enqueueMigrateMemObjects({vec_in_buf[num_buf+N]},0/* 0 means from host*/,NULL,&Task2_R);
            Read2.push_back(Task2_R);
            q2.enqueueTask(alveo_hls4ml_2,&Read2,&Task2_K);
            Kernel2.push_back(Task2_K);
            q2.enqueueMigrateMemObjects({vec_out_buf[num_buf+N]},CL_MIGRATE_MEM_OBJECT_HOST,&Kernel2,&Task2_W);
            Write2.push_back(Task2_W);

            //vec_in_buf[2N]~vec_in_buf[3N-1] -> for CU3
            alveo_hls4ml_3.setArg(0, vec_in_buf[2*num_buf+N]);
            alveo_hls4ml_3.setArg(1, vec_out_buf[2*num_buf+N]);
            q3.enqueueMigrateMemObjects({vec_in_buf[2*num_buf+N]},0/* 0 means from host*/,NULL,&Task3_R);
            Read3.push_back(Task3_R);
            q3.enqueueTask(alveo_hls4ml_3,&Read3,&Task3_K);
            Kernel3.push_back(Task3_K);
            q3.enqueueMigrateMemObjects({vec_out_buf[2*num_buf+N]},CL_MIGRATE_MEM_OBJECT_HOST,&Kernel3,&Task3_W);
            Write3.push_back(Task3_W);

            //vec_in_buf[3N]~vec_in_buf[4N-1] -> for CU4
            alveo_hls4ml_4.setArg(0, vec_in_buf[3*num_buf+N]);
            alveo_hls4ml_4.setArg(1, vec_out_buf[3*num_buf+N]);
            q4.enqueueMigrateMemObjects({vec_in_buf[3*num_buf+N]},0/* 0 means from host*/,NULL,&Task4_R);
            Read4.push_back(Task4_R);
            q4.enqueueTask(alveo_hls4ml_4,&Read4,&Task4_K);
            Kernel4.push_back(Task4_K);
            q4.enqueueMigrateMemObjects({vec_out_buf[3*num_buf+N]},CL_MIGRATE_MEM_OBJECT_HOST,&Kernel4,&Task4_W);
            Write4.push_back(Task4_W);
        }

        for(int N=0; N<num_buf; N++){
            Write1[N].wait();
            Write2[N].wait();
            Write3[N].wait();
            Write4[N].wait();
        }

        t2 = Clock::now();
        Read1.clear();
        Read2.clear();
        Read3.clear();
        Read4.clear();
        Kernel1.clear();
        Kernel2.clear();
        Kernel3.clear();
        Kernel4.clear();
        Write1.clear();
        Write2.clear();
        Write3.clear();
        Write4.clear();
        if(i!=0)current_total += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count(); 

        std::cout << "computation: " << i << std::endl;
        std::cout << "t2 - t1: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << " ns" <<std::endl;
        std::cout << "current_total: " << current_total << " ns" << std::endl;
        std::cout << "prediction: " << std::endl; 

//================================
//Check the computatation process
//================================

        for(int N=0;N<num_buf*4;N++){
            fout << " buffer"<<N<< "\n";
            for(int j=0;j<COMPRESSION;j++){
                fout << vec_output[N][50].layer[j] << " "; 
            }
            fout << "\n";
        }
        fout << "\n"; 
        fout << "================================="; 
        if(i!=event_num-1){
            fout << " reset \n"; 
            for(int N=0;N<num_buf*4;N++){
                for(int j=0;j<COMPRESSION;j++){
                    vec_output[N][50].layer[j] = 1; 
                }
            }
            for(int N=0;N<num_buf*4;N++){
                fout << " buffer"<<N<<" \n"; 
                for(int j=0;j<COMPRESSION;j++){
                    fout << vec_output[N][50].layer[j] << " "; 
                }
                fout << "\n";
            }
        }
    }
    //print timing
    std::cout << "=========================================" <<std::endl;
    std::cout << "FPGA average time: " << current_total/(event_num-1) << " ns" << std::endl;
    fout << "FPGA average time: " << current_total/(event_num-1) << " ns \n";
//=====================
//Output result
//=====================
/*
        std::cout<<"Predictions: \n";
        fout <<"Predictions:  \n";
        for(int i=0;i<OUT ;i++){
            std::cout << pr[i] << " ";
            fout << pr[i] << " ";
        }
        std::cout << std::endl;*/
        fout<<"\n";

        for(int N=0;N<num_buf*4;N++){
            if(N<num_buf)std::cout<<"Quantized predictions(1CU) buf"<<N<<": \n";
            else if(N>=num_buf && N<2*num_buf)std::cout<<"Quantized predictions(2CU) buf"<<N-num_buf<<": \n";
            else if(N>=2*num_buf && N<3*num_buf)std::cout<<"Quantized predictions(3CU) buf"<<N-2*num_buf<<": \n";
            else std::cout<<"Quantized predictions(4CU) buf"<<N-3*num_buf<<": \n";

            if(N<num_buf)fout <<"Quantized predictions(1CU) buf"<<N<<": \n";
            else if(N>=num_buf && N<2*num_buf)fout <<"Quantized predictions(2CU) buf"<<N-num_buf<<": \n";
            else if(N>=2*num_buf && N<3*num_buf)fout <<"Quantized predictions(3CU) buf"<<N-2*num_buf<<": \n";
            else fout <<"Quantized predictions(4CU) buf"<<N-3*num_buf<<": \n";
            for(int i=0;i<BatchSize/COMPRESSION ;i++){
                for(int j=0;j<COMPRESSION ;j++){
                    //std::cout << source_hw_results1[i].layer[j]<< " ";
                    fout << vec_output[N][i].layer[j] << " "; 
                }
            }
            fout << "\n\n";
        }
        fout << "\n\n";
        std::cout<<"---- END EVENT "<<" ----"<<std::endl;

      }
    }

// OPENCL HOST CODE AREA END
    fin.close();
    fpr.close();
    fout.close();

    return EXIT_SUCCESS;
}