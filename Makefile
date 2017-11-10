all:
	g++ src/reprojection.cpp -o bin/reprojection.x
clean:
	rm -f bin/*
	rm -f *.ply
	
