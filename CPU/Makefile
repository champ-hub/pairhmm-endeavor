GCC=g++
GCCFLAGS=-O3 -fopenmp -mavx -mavx512f -mavx512dq -mavx512bw

pairhmm: pairHMMCPU.cpp
	$(GCC) $(GCCFLAGS) pairHMMCPU.cpp -o pairhmm -lz

clean:
	rm -f *.o core a.out pairhmm *~
