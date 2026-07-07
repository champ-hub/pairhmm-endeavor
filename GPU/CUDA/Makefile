NVCC=nvcc
NVCCFLAGS=-O3 -arch=native --ptxas-options=-v --fmad=true --expt-relaxed-constexpr

pairhmm: pairHMM.cu
	$(NVCC) $(NVCCFLAGS) pairHMM.cu -o pairhmm -lz

clean:
	rm -f *.o core a.out pairhmm *~
