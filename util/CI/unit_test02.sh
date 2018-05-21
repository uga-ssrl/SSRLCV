#!/bin/bash

echo "running cube test..."



make -j4
./bin/2viewreprojection.x data/2view/cube_cameras.txt data/2view/cube_matches.txt

if[ $? -eq 0 ]
then
    echo OK
else
    echo FAIL
    exit 1
fi

#mv output.ply output_gpu.ply
./bin/2viewreprojection.x data/2view/cube_cameras.txt data/2view/cube_matches.txt 0

if[ $? -eq 0 ]
then
    echo OK
else
    echo FAIL
    exit 1
fi

#mv output.ply output_cpu.ply
diff output_cpu.ply output_gpu.ply
make clean
