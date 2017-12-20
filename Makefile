all:
	nvcc -std=c++11 src/reprojection.cpp -o bin/reprojection.x -lcublas
	nvcc -c -I/usr/local/cuda/include src/example_linear_solver.cpp -o util/example_linear_solver.o
	g++ -fopenmp -o bin/example_linear_solver.x util/example_linear_solver.o -L/usr/local/cuda/lib64 -lcusolver -lcudart
clean:
	rm -f bin/*
	rm -f *.ply
	rm -f src/*~
	rm -f util/*~
	rm -f util/*.o
