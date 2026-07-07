HIPCC=hipcc
HIPCCFLAGS=-O3 -ffast-math  --offload-arch=native  -ffp-contract=fast

pairhmm: pairHMM.hip
	$(NVCC) $(NVCCFLAGS) pairHMM.hip -o pairhmm -lz

clean:
	rm -f *.o core a.out pairhmm *~
