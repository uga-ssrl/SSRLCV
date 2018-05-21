#!/bin/bash

echo "running cube test..."
make -j4
./bin/2viewreprojection.x data/2view/cube_cameras.txt data/2view/cube_matches.txt
mv output.ply output_gpu.ply
./bin/2viewreprojection.x data/2view/cube_cameras.txt data/2view/cube_matches.txt 0
mv output.ply output_cpu.ply
diff output_cpu.ply output_gpu.ply
make clean
