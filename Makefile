all: 
	python setup.py build_ext --inplace
	#mv *so bin
clean:
	rm -fv *so 
