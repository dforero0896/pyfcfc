PYTHON ?= python3

all: build

# Editable install (requires a C compiler with OpenMP support, Cython, numpy)
build:
	$(PYTHON) -m pip install -e .

# Same, but with SIMD vectorisation (AVX/AVX2/AVX512 auto-detected).
# Implies -march=native: the build then only runs on compatible machines.
simd:
	PYFCFC_WITH_SIMD=1 $(PYTHON) -m pip install -e . --no-build-isolation

test:
	$(PYTHON) -m pytest tests -v

examples:
	@for script in examples/example_*.py; do \
		case $$script in *benchmark*) continue;; esac; \
		echo "=== Running $$script ==="; \
		$(PYTHON) $$script || exit 1; \
	done

benchmark:
	$(PYTHON) examples/example_benchmark.py

clean:
	rm -rf build dist ./*.egg-info .pytest_cache
	rm -f FCFC-main/src/fcfc/2pt/pyfcfc.c FCFC-main/src/fcfc/2pt_box/pyfcfc.c
	find pyfcfc -name "*.so" -delete
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +

.PHONY: all build simd test examples clean
