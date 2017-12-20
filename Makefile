all:
	nvcc -std=c++11 src/reprojection.cpp -o bin/reprojection.x -lcublas
clean:
	rm -f bin/*
	rm -f *.ply
	rm -f src/*~
	rm -f util/*~
	rm -f util/*.o
