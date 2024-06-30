all: 
	#python setup.py build_ext --inplace
	python setup.py develop --user
	#mv *so bin
test:
	gcc -Isrc -Isrc/2pt -Itree kdtree_build.c tree/kdtree.c -o kdtree_build
clean:
	rm -fv *so 
