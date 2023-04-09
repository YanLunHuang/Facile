# hls4ml on Alveo U50 (HLS C/C++ Kernel)
## Vitis version 2019.2
## Activate the tool 
```bash
source /opt/Xilinx/Vitis/2019.2/settings64.sh # Vitis
source /opt/xilinx/xrt/setup.sh # Vitis XRT
```
## Compile project
```bash
make cleanall # clean all of the related files
make check TARGET=sw_emu DEVICE=xilinx_u50_xdma_201920_1 all  # software emulation
make check TARGET=hw_emu DEVICE=xilinx_u50_xdma_201920_1 all  # hardware emulation
make TARGET=hw DEVICE=xilinx_u50_xdma_201920_1 all # build
```
## Run project
```bash
XCL_EMULATION_MODE=sw_emu ./host ./build_dir.sw_emu.xilinx_u50_xdma_201920_1/alveo_hls4ml.xclbin  # software emulation
XCL_EMULATION_MODE=hw_emu ./host ./build_dir.hw_emu.xilinx_u50_xdma_201920_1/alveo_hls4ml.xclbin  # hardware emulation
./host build_dir.hw.xilinx_u50_xdma_201920_1/alveo_hls4ml.xclbin  # run on U50
```
## Some detail
```bash
host.cpp:
There are 4 CUs.
Split input(output) in two part and take two times to compute.

host_2HBM_4CU.cpp:
There are 4 CUs.
Use 2 HBM bank in ping-pong fashion.

host_iteration.cpp:
There are only 1 CUs running.
Add an iteration variable to manipulate the iteration times.
Input(output) is splitted based on the iteration variable.

host_2HBM.cpp:
There are only 1 CUs running.
Use 2 HBM bank in ping-pong fashion.

host_normal.cpp:
There are 4 CUs.
Compute parallel.

```
## Learn from Vitis tutorial
```bash
https://github.com/Xilinx/Vitis-Tutorials/blob/2021.2/Hardware_Acceleration/Design_Tutorials/02-bloom/5_data-movement.md
https://github.com/Xilinx/Vitis-Tutorials/blob/2021.1/Hardware_Acceleration/Design_Tutorials/02-bloom/6_using-multiple-ddr.md
```
