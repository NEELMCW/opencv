#!/bin/bash -e
echo "Generate files for CL runtime..."
python parser_cl.py opencl_core < sources/cl.h
python parser_cl.py opencl_gl < sources/cl_gl.h

python parser_clamdblas.py < sources/clBLAS.h
python parser_clamdfft.py < sources/clFFT.h

echo "Generate files for CL runtime... Done"
