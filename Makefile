all:
	g++ -std=c++11 src/reprojection.cpp -o bin/reprojection.x
clean:
	rm -f bin/*
	rm -f *.ply
	
