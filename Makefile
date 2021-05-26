all: 
	python setup.py build_ext --inplace
	#mv *so bin
test:
	gcc -Isrc -Isrc/2pt -Itree kdtree_build.c tree/kdtree.c -o kdtree_build
clean:
	rm -fv *so 
