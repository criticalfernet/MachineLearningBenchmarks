CXX = g++
CXXFLAGS = -O3 -DNDEBUG -march=native -std=c++23 -I./eigen-5.0.0

SRCS = $(wildcard Algorithms/*/main.cpp)
BENCH_SRCS = $(wildcard Algorithms/*/bench.cpp)

TARGETS = $(patsubst Algorithms/%/main.cpp, build/%, $(SRCS))
BENCH_TARGETS = $(patsubst Algorithms/%/bench.cpp, build/%_bench, $(BENCH_SRCS))

.PRECIOUS: build/% build/%_bench

build/%: Algorithms/%/main.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $< -o $@

build/%_bench: Algorithms/%/bench.cpp Algorithms/%/main.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) Algorithms/$*/bench.cpp Algorithms/$*/main.cpp -o $@

run-%: build/%
	./$<

bench-%: build/%_bench
	./$<

clean:
	rm -rf build

.PHONY: clean
