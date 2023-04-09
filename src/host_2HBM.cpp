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
    std::cout<<"============host_2HBM.cpp============"<<std::endl;
    std::cout<<"Only 1CU"<<std::endl;
    std::cout<<"Use 2 HBM bank in ping-pong fashion"<<std::endl;
    std::cout<<"====================================="<<std::endl;
    int event_num=150;
    cl_int err;
    cl::Kernel alveo_hls4ml_1;
    cl::Kernel alveo_hls4ml_2;
    cl::Kernel alveo_hls4ml_3;
    cl::Kernel alveo_hls4ml_4;
    std::string datadir = STRINGIFY(HLS4ML_DATA_DIR);
    std::string xclbinFilename = "";
    if (argc > 1) xclbinFilename = argv[1];
    if (argc > 2) event_num = atoi(argv[2]);
    if (argc > 3) datadir = argv[3];
    std::cout << "Will run " << event_num << " time(s), using " << datadir << " to get input features and output predictions (tb_input_features.dat and tb_output_predictions.dat)" << std::endl;

    size_t vector_size_in_bytes = sizeof(group_in) * BatchSize;
    size_t vector_size_out_bytes = sizeof(group_out) * (BatchSize/COMPRESSION);
    // Allocate Memory in Host Memory
    // When creating a buffer with user pointer (CL_MEM_USE_HOST_PTR), under the hood user ptr 
    // is used if it is properly aligned. when not aligned, runtime had no choice but to create
    // its own host side buffer. So it is recommended to use this allocator if user wish to
    // create buffer using CL_MEM_USE_HOST_PTR to align user buffer to page boundary. It will 
    // ensure that user buffer is used when user create Buffer/Mem object with CL_MEM_USE_HOST_PTR 
    std::vector<group_in,aligned_allocator<group_in>> source_in1(BatchSize);
    std::vector<group_in,aligned_allocator<group_in>> source_in2(BatchSize);
    std::vector<group_in,aligned_allocator<group_in>> source_in3(BatchSize);
    std::vector<group_in,aligned_allocator<group_in>> source_in4(BatchSize);
    std::vector<group_out,aligned_allocator<group_out>> source_hw_results1(BatchSize/COMPRESSION);
    std::vector<group_out,aligned_allocator<group_out>> source_hw_results2(BatchSize/COMPRESSION);
    std::vector<group_out,aligned_allocator<group_out>> source_hw_results3(BatchSize/COMPRESSION);
    std::vector<group_out,aligned_allocator<group_out>> source_hw_results4(BatchSize/COMPRESSION);

    //Reset the input data
    for(int i = 0; i < BatchSize; i++) { 
        for(int j = 0; j < COMPRESSION; j++) { 
            source_in1[i].layer[j] = 0;
            source_in2[i].layer[j] = 0;
            source_in3[i].layer[j] = 0;
            source_in4[i].layer[j] = 0;
        }
    }

    //Reset the output result
    for(int i = 0 ; i < BatchSize/COMPRESSION ; i++){
        for(int j = 0; j < COMPRESSION; j++) { 
            source_hw_results1[i].layer[j] = 0;
            source_hw_results2[i].layer[j] = 0;
            source_hw_results3[i].layer[j] = 0;
            source_hw_results4[i].layer[j] = 0;
        }
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
        printf("Creating a kernel [%s] for CU %d\n", krnl_name_full.c_str(), 0);
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
    cl_mem_ext_ptr_t pipo_buffer_in1[2];
    cl_mem_ext_ptr_t pipo_buffer_output1[2];
    pipo_buffer_in1[0].flags = 0 | XCL_MEM_TOPOLOGY;
    pipo_buffer_in1[0].param = 0;
    pipo_buffer_in1[0].obj   = source_in1.data();
    pipo_buffer_in1[1].flags = 1 | XCL_MEM_TOPOLOGY;
    pipo_buffer_in1[1].param = 0;
    pipo_buffer_in1[1].obj   = source_in1.data();

    pipo_buffer_output1[0].flags = 2 | XCL_MEM_TOPOLOGY;
    pipo_buffer_output1[0].param = 0;
    pipo_buffer_output1[0].obj   = source_hw_results1.data();
    pipo_buffer_output1[1].flags = 3 | XCL_MEM_TOPOLOGY;
    pipo_buffer_output1[1].param = 0;
    pipo_buffer_output1[1].obj   = source_hw_results1.data();


    // Allocate Buffer in Global Memory
    // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and 
    // Device-to-host communication
    cl::Buffer buffer_in1[2];
    cl::Buffer buffer_output1[2];
    cl::Buffer buffer_in2,buffer_in3,buffer_in4;
    cl::Buffer buffer_output2,buffer_output3,buffer_output4;

    buffer_in1[0] = cl::Buffer(context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            vector_size_in_bytes, &pipo_buffer_in1[0]);
    buffer_in1[1] = cl::Buffer(context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
            vector_size_in_bytes, &pipo_buffer_in1[1]);
    buffer_output1[0] = cl::Buffer(context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX, 
            vector_size_out_bytes, &pipo_buffer_output1[0]);
    buffer_output1[1] = cl::Buffer(context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX, 
            vector_size_out_bytes, &pipo_buffer_output1[1]);

    buffer_in2 = cl::Buffer(context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            vector_size_in_bytes, source_in2.data());
    buffer_output2 = cl::Buffer(context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, 
            vector_size_out_bytes, source_hw_results2.data());

    buffer_in3 = cl::Buffer(context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            vector_size_in_bytes, source_in3.data());
    buffer_output3 = cl::Buffer(context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, 
            vector_size_out_bytes, source_hw_results3.data());

    buffer_in4 = cl::Buffer(context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
            vector_size_in_bytes, source_in4.data());
    buffer_output4 = cl::Buffer(context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, 
            vector_size_out_bytes, source_hw_results4.data());

    int narg1 = 0;
    alveo_hls4ml_1.setArg(narg1++, buffer_in1[0]);
    alveo_hls4ml_1.setArg(narg1++, buffer_output1[0]);
    q1.enqueueMigrateMemObjects({buffer_in1[0], buffer_in1[1], buffer_output1[0],buffer_output1[1]}, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED);
    //Specify size of sub-buffers
    size_t subbuf_size_in_bytes = vector_size_in_bytes/2;
    size_t subbuf_size_out_bytes = vector_size_out_bytes/2;
    //Declare sub-buffer regions to specify offset and size of sub-buffer   
    cl_buffer_region subbuf_in_info1[2];
    cl_buffer_region subbuf_out_info1[2];
    // Specify offset and size of sub-buffers 
    subbuf_in_info1[0] = {0, subbuf_size_in_bytes};
    subbuf_in_info1[1] = {subbuf_size_in_bytes, subbuf_size_in_bytes};
    subbuf_out_info1[0] = {0, subbuf_size_out_bytes};
    subbuf_out_info1[1] = {subbuf_size_out_bytes, subbuf_size_out_bytes};
    // Declare sub-buffers
    cl::Buffer subbuf_in1[2];
    cl::Buffer subbuf_out1[2];
    // Create sub-buffers from buffers based on sub-buffer regions
    subbuf_in1[0] = buffer_in1[0].createSubBuffer(CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &subbuf_in_info1[0]);
    subbuf_in1[1] = buffer_in1[1].createSubBuffer(CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &subbuf_in_info1[1]);
    subbuf_out1[0] = buffer_output1[0].createSubBuffer (CL_MEM_WRITE_ONLY,  CL_BUFFER_CREATE_TYPE_REGION, &subbuf_out_info1[0]);
    subbuf_out1[1] = buffer_output1[1].createSubBuffer (CL_MEM_WRITE_ONLY,  CL_BUFFER_CREATE_TYPE_REGION, &subbuf_out_info1[1]);


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
        for(int i = 0; i < BatchSize; i++) { 
            for(int j = 0; j < 18; j++) { 
                source_in1[i].layer[j] = (data_t)in[i*18+j];
                source_in2[i].layer[j] = (data_t)in[i*18+j]+1;
                source_in3[i].layer[j] = (data_t)in[i*18+j]+2;
                source_in4[i].layer[j] = (data_t)in[i*18+j]+3;
            }
        }
        //Reset the output result
        for(int i = 0 ; i < BatchSize/COMPRESSION ; i++){
            for(int j = 0; j < COMPRESSION; j++) { 
                source_hw_results1[i].layer[j] = 0;
                source_hw_results2[i].layer[j] = 0;
                source_hw_results3[i].layer[j] = 0;
                source_hw_results4[i].layer[j] = 0;
            }
        }

//========================
//Start to run on FPGA
//========================
    auto t1 = Clock::now();
    auto t2 = Clock::now();
    auto duration_total = 0;
    std::vector<cl::Event> Read1,Read2,Read3,Read4;
    std::vector<cl::Event> Kernel1,Kernel2,Kernel3,Kernel4;
    std::vector<cl::Event> Write1,Write2,Write3,Write4;
    cl::Event Task1_R,Task1_K,Task1_W;
    cl::Event Task2_R,Task2_K,Task2_W;
    cl::Event Task3_R,Task3_K,Task3_W;
    cl::Event Task4_R,Task4_K,Task4_W;
    for(int i=0;i<event_num;i++){

        t1 = Clock::now();

        alveo_hls4ml_1.setArg(0, subbuf_in1[0]);
        alveo_hls4ml_1.setArg(1, subbuf_out1[0]);
        q1.enqueueMigrateMemObjects({subbuf_in1[0]},0/* 0 means from host*/,NULL,&Task1_R);
        Read1.push_back(Task1_R);
        q1.enqueueNDRangeKernel(alveo_hls4ml_1,0,1,1,&Read1,&Task1_K);
        Kernel1.push_back(Task1_K);
        q1.enqueueMigrateMemObjects({subbuf_out1[0]},CL_MIGRATE_MEM_OBJECT_HOST,&Kernel1,&Task1_W);
        Write1.push_back(Task1_W);

        alveo_hls4ml_1.setArg(0, subbuf_in1[1]);
        alveo_hls4ml_1.setArg(1, subbuf_out1[1]);
        q1.enqueueMigrateMemObjects({subbuf_in1[1]},0/* 0 means from host*/,&Read1,&Task1_R);
        Read1.push_back(Task1_R);
        q1.enqueueNDRangeKernel(alveo_hls4ml_1,0,1,1,&Read1,&Task1_K);
        Kernel1.push_back(Task1_K);
        q1.enqueueMigrateMemObjects({subbuf_out1[1]},CL_MIGRATE_MEM_OBJECT_HOST,&Kernel1,&Task1_W);
        Write1.push_back(Task1_W);

        Write1[0].wait();
        Write1[1].wait();

        t2 = Clock::now();
        duration_total = duration_total+std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count(); 

        //print timing
        if(i==event_num-1){
        std::cout << "FPGA time(4CU): " << duration_total/event_num << " ns" << std::endl;
        fout << "FPGA time(4CU): " << duration_total/event_num << " ns \n";
        }
    }
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

        //std::cout<<"Quantized predictions: \n";
        fout <<"Quantized predictions: \n";
        //std::cout<<"Kernel1: \n";
        fout <<"Kernel1: \n";
        for(int i=0;i<BatchSize/COMPRESSION ;i++){
            for(int j=0;j<COMPRESSION ;j++){
                //std::cout << source_hw_results1[i].layer[j]<< " ";
                fout << source_hw_results1[i].layer[j] << " "; 
            }
        }
        //std::cout<<"\nKernel2: \n";
        fout <<"\n Kernel2: \n";
        for(int i=0;i<BatchSize/COMPRESSION ;i++){
            for(int j=0;j<COMPRESSION ;j++){
                //std::cout << source_hw_results2[i].layer[j]<< " ";
                fout << source_hw_results2[i].layer[j] << " "; 
            }
        }
        //std::cout<<"\nKernel3: \n";
        fout <<"\n Kernel3: \n";
        for(int i=0;i<BatchSize/COMPRESSION ;i++){
            for(int j=0;j<COMPRESSION ;j++){
                //std::cout << source_hw_results3[i].layer[j]<< " ";
                fout << source_hw_results3[i].layer[j] << " "; 
            }
        }

        //std::cout<<"\nKernel4: \n";
        fout <<"\n Kernel4: \n";
        for(int i=0;i<BatchSize/COMPRESSION ;i++){
            for(int j=0;j<COMPRESSION ;j++){
                //std::cout << source_hw_results3[i].layer[j]<< " ";
                fout << source_hw_results4[i].layer[j] << " "; 
            }
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
