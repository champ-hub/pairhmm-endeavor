#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <random>
#include <time.h>
#include <immintrin.h>
#include <omp.h>
#include "PFACPUBench.cpp"
#include "PFACPUDataset.cpp"
#include <zlib.h>

#define NSEC_IN_SEC 1000000000.0;

using namespace std;

struct Arguments{
    string inputfile = "";
    string outputfile = "";
    int precision = 16;
    int width = 256;
    bool benchmark = false;
    bool help = false;
    int batchsize = 16384;
};

void process256_float(unsigned int* hapCPU32, unsigned int* readCPU32, unsigned int* qCPU32, unsigned int* iCPU32, unsigned int* dCPU32, unsigned long long* rlenCPU, unsigned long long* hlenCPU, 
    unsigned int rslen[16], unsigned int haplen[16], __m256i* haplen8f, __m256i* rslen8f, char buf[16][7][16384], unsigned int* seq_size, unsigned int* seqCounts, int count, int batchsize, unsigned long long* rcount, unsigned long long* hcount, unsigned long long* gcups){
    
    unsigned int maxsizeH = 0, maxsizeR = 0, kernidx = 0;

    for(int k = 0; k < 8; k++){
        haplen[k] = (unsigned int)strlen(buf[k][0]);
        rslen[k] = (unsigned int)strlen(buf[k][1]);

        *gcups += (unsigned long long)haplen[k]*(unsigned long long)rslen[k];

        maxsizeH = max(maxsizeH, haplen[k]);
        maxsizeR = max(maxsizeR, rslen[k]);
    }

    rslen8f[(count/8) - 1] = _mm256_setr_epi32(rslen[0],rslen[1],rslen[2],rslen[3],rslen[4],rslen[5],rslen[6],rslen[7]);
    haplen8f[(count/8) - 1] = _mm256_setr_epi32(haplen[0],haplen[1],haplen[2],haplen[3],haplen[4],haplen[5],haplen[6],haplen[7]);

    if(maxsizeH > 1024){
        for(int j = 1; j <= 4; j++){
            if(maxsizeH <= 1024*pow(2,j)){
                kernidx = 31 + j;
                break;
            }
        }
    }
    else{
        for(int j = 0; j < 32; j++){
            if(maxsizeH <= 32*(j + 1)){
                kernidx = j;
                maxsizeH = 32*(j + 1);
                maxsizeR = 32*(j + 1);
                break;
            }
        }
    }

    seqCounts[kernidx*(batchsize/8 + 1) + seq_size[kernidx]/8] = (count/8) - 1;

    seq_size[kernidx] += 8;

    for(int c = 0; c < 8*maxsizeH; c++){
        hapCPU32[*hcount + c] = buf[c%8][0][c/8];
        buf[c%8][0][c/8] = 0;
    }

    for(int c = 0; c < 8*maxsizeR; c++){
        readCPU32[*rcount + c] = buf[c%8][1][c/8];
        qCPU32[*rcount + c] = buf[c%8][2][c/8] - 33;
        iCPU32[*rcount + c] = buf[c%8][3][c/8] - 33;
        dCPU32[*rcount + c] = buf[c%8][4][c/8] - 33;

        buf[c%8][1][c/8] = 0;
        buf[c%8][2][c/8] = 0;
        buf[c%8][3][c/8] = 0;
        buf[c%8][4][c/8] = 0;
    }

    *rcount += (unsigned long long)8*maxsizeR;
    *hcount += (unsigned long long)8*maxsizeH;

    rlenCPU[(count/8)] = *rcount;
    hlenCPU[(count/8)] = *hcount;
}

void process512_float(unsigned int* hapCPU32, unsigned int* readCPU32, unsigned int* qCPU32, unsigned int* iCPU32, unsigned int* dCPU32, unsigned long long* rlenCPU, unsigned long long* hlenCPU, 
    unsigned int rslen[16], unsigned int haplen[16], __m512i* haplen16, __m512i* rslen16, char buf[16][7][16384], unsigned int* seq_size, unsigned int* seqCounts, int count, int batchsize, unsigned long long* rcount, unsigned long long* hcount, unsigned long long* gcups){

    unsigned int maxsizeH = 0, maxsizeR = 0, kernidx = 0;

    for(int k = 0; k < 16; k++){
        haplen[k] = (unsigned int)strlen(buf[k][0]);
        rslen[k] = (unsigned int)strlen(buf[k][1]);

        *gcups += (unsigned long long)haplen[k]*(unsigned long long)rslen[k];

        maxsizeH = max(maxsizeH, haplen[k]);
        maxsizeR = max(maxsizeR, rslen[k]);
    }

    rslen16[(count/16) - 1] = _mm512_setr_epi32(rslen[0],rslen[1],rslen[2],rslen[3],rslen[4],rslen[5],rslen[6],rslen[7],rslen[8],rslen[9],rslen[10],rslen[11],rslen[12],rslen[13],rslen[14],rslen[15]);
    haplen16[(count/16) - 1] = _mm512_setr_epi32(haplen[0],haplen[1],haplen[2],haplen[3],haplen[4],haplen[5],haplen[6],haplen[7],haplen[8],haplen[9],haplen[10],haplen[11],haplen[12],haplen[13],haplen[14],haplen[15]);

    if(maxsizeH > 1024){
        for(int j = 1; j <= 4; j++){
            if(maxsizeH <= 1024*pow(2,j)){
                kernidx = 31 + j;
                break;
            }
        }
    }
    else{
        for(int j = 0; j < 32; j++){
            if(maxsizeH <= 32*(j + 1)){
                kernidx = j;
                maxsizeH = 32*(j + 1);
                maxsizeR = 32*(j + 1);
                break;
            }
        }
    }

    seqCounts[kernidx*(batchsize/16 + 1) + seq_size[kernidx]/16] = (count/16) - 1;

    seq_size[kernidx] += 16;

    for(int c = 0; c < 16*maxsizeH; c++){
        hapCPU32[*hcount + c] = buf[c%16][0][c/16];
        buf[c%16][0][c/16] = 0;
    }

    for(int c = 0; c < 16*maxsizeR; c++){
        readCPU32[*rcount + c] = buf[c%16][1][c/16];
        qCPU32[*rcount + c] = buf[c%16][2][c/16] - 33;
        iCPU32[*rcount + c] = buf[c%16][3][c/16] - 33;
        dCPU32[*rcount + c] = buf[c%16][4][c/16] - 33;

        buf[c%16][1][c/16] = 0;
        buf[c%16][2][c/16] = 0;
        buf[c%16][3][c/16] = 0;
        buf[c%16][4][c/16] = 0;
    }

    *rcount += (unsigned long long)16*maxsizeR;
    *hcount += (unsigned long long)16*maxsizeH;

    rlenCPU[(count/16)] = *rcount;
    hlenCPU[(count/16)] = *hcount;
}

void process256_double(unsigned long long* hapCPU, unsigned long long* readCPU, unsigned long long* qCPU, unsigned long long* iCPU, unsigned long long* dCPU, unsigned long long* rlenCPU, unsigned long long* hlenCPU, 
    unsigned int rslen[16], unsigned int haplen[16], __m256i* haplen4, __m256i* rslen4, char buf[16][7][16384], unsigned int* seq_size, unsigned int* seqCounts, int count, int batchsize, unsigned long long* rcount, unsigned long long* hcount, unsigned long long* gcups){

    unsigned int maxsizeH = 0, maxsizeR = 0, kernidx = 0;

    for(int k = 0; k < 4; k++){
        haplen[k] = (unsigned int)strlen(buf[k][0]);
        rslen[k] = (unsigned int)strlen(buf[k][1]);

        *gcups += (unsigned long long)haplen[k]*(unsigned long long)rslen[k];

        maxsizeH = max(maxsizeH, haplen[k]);
        maxsizeR = max(maxsizeR, rslen[k]);
    }

    rslen4[(count/4) - 1] = _mm256_setr_epi64x(rslen[0],rslen[1],rslen[2],rslen[3]);
    haplen4[(count/4) - 1] = _mm256_setr_epi64x(haplen[0],haplen[1],haplen[2],haplen[3]);

    if(maxsizeH > 1024){
        for(int j = 1; j <= 4; j++){
            if(maxsizeH <= 1024*pow(2,j)){
                kernidx = 31 + j;
                break;
            }
        }
    }
    else{
        for(int j = 0; j < 32; j++){
            if(maxsizeH <= 32*(j + 1)){
                kernidx = j;
                maxsizeH = 32*(j + 1);
                maxsizeR = 32*(j + 1);
                break;
            }
        }
    }

    seqCounts[kernidx*(batchsize/4 + 1) + seq_size[kernidx]/4] = (count/4) - 1;

    seq_size[kernidx] += 4;

    for(int c = 0; c < 4*maxsizeH; c++){
        hapCPU[*hcount + c] = buf[c%4][0][c/4];
        buf[c%4][0][c/4] = 0;
    }

    for(int c = 0; c < 4*maxsizeR; c++){
        readCPU[*rcount + c] = buf[c%4][1][c/4];
        qCPU[*rcount + c] = buf[c%4][2][c/4] - 33;
        iCPU[*rcount + c] = buf[c%4][3][c/4] - 33;
        dCPU[*rcount + c] = buf[c%4][4][c/4] - 33;

        buf[c%4][1][c/4] = 0;
        buf[c%4][2][c/4] = 0;
        buf[c%4][3][c/4] = 0;
        buf[c%4][4][c/4] = 0;
    }

    *rcount += (unsigned long long)4*maxsizeR;
    *hcount += (unsigned long long)4*maxsizeH;

    rlenCPU[(count/4)] = *rcount;
    hlenCPU[(count/4)] = *hcount;
}

void process512_double(unsigned long long* hapCPU, unsigned long long* readCPU, unsigned long long* qCPU, unsigned long long* iCPU, unsigned long long* dCPU, unsigned long long* rlenCPU, unsigned long long* hlenCPU, 
    unsigned int rslen[16], unsigned int haplen[16], __m512i* haplen8, __m512i* rslen8, char buf[16][7][16384], unsigned int* seq_size, unsigned int* seqCounts, int count, int batchsize, int* rcount, int* hcount, unsigned long long* gcups){

    unsigned int maxsizeH = 0, maxsizeR = 0, kernidx = 0;

    for(int k = 0; k < 8; k++){
        haplen[k] = (unsigned int)strlen(buf[k][0]);
        rslen[k] = (unsigned int)strlen(buf[k][1]);

        *gcups += (unsigned long long)haplen[k]*(unsigned long long)rslen[k];

        maxsizeH = max(maxsizeH, haplen[k]);
        maxsizeR = max(maxsizeR, rslen[k]);
    }

    rslen8[(count/8) - 1] = _mm512_setr_epi64(rslen[0],rslen[1],rslen[2],rslen[3],rslen[4],rslen[5],rslen[6],rslen[7]);
    haplen8[(count/8) - 1] = _mm512_setr_epi64(haplen[0],haplen[1],haplen[2],haplen[3],haplen[4],haplen[5],haplen[6],haplen[7]);

    if(maxsizeH > 1024){
        for(int j = 1; j <= 4; j++){
            if(maxsizeH <= 1024*pow(2,j)){
                kernidx = 31 + j;
                break;
            }
        }
    }
    else{
        for(int j = 0; j < 32; j++){
            if(maxsizeH <= 32*(j + 1)){
                kernidx = j;
                maxsizeH = 32*(j + 1);
                maxsizeR = 32*(j + 1);
                break;
            }
        }
    }

    seqCounts[kernidx*(batchsize/8 + 1) + seq_size[kernidx]/8] = (count/8) - 1;

    seq_size[kernidx] += 8;

    for(int c = 0; c < 8*maxsizeH; c++){
        hapCPU[*hcount + c] = buf[c%8][0][c/8];
        buf[c%8][0][c/8] = 0;
    }

    for(int c = 0; c < 8*maxsizeR; c++){
        readCPU[*rcount + c] = buf[c%8][1][c/8];
        qCPU[*rcount + c] = buf[c%8][2][c/8] - 33;
        iCPU[*rcount + c] = buf[c%8][3][c/8] - 33;
        dCPU[*rcount + c] = buf[c%8][4][c/8] - 33;

        buf[c%8][1][c/8] = 0;
        buf[c%8][2][c/8] = 0;
        buf[c%8][3][c/8] = 0;
        buf[c%8][4][c/8] = 0;
    }

    *rcount += (unsigned long long)8*maxsizeR;
    *hcount += (unsigned long long)8*maxsizeH;

    rlenCPU[(count/8)] = *rcount;
    hlenCPU[(count/8)] = *hcount;
}

void processDataset(string inputfile, string outputfile, int precision, int width, int batchsize){

    gzFile gzfile = gzopen(inputfile.c_str(), "rb");
    FILE* fpout = fopen(outputfile.c_str(), "w");

    int j = 1, idx = 0, modulo = 0;
    char buf[16][7][16384] ={{{0}}};
    char tmp[7*16384] = {0};
    char* pch;

    if(gzgets(gzfile, tmp, sizeof(tmp)) != Z_NULL) j = 0;

    unsigned long long gcups = 0, CPUsize = (unsigned long long)batchsize*(unsigned long long)1024;
    double throughput = 0;

    double *finalsum, *qual;
    unsigned long long *hapCPU, *readCPU, *qCPU, *iCPU, *dCPU;

    float *finalsum32, *qual32;
    unsigned int *hapCPU32, *readCPU32, *qCPU32, *iCPU32, *dCPU32;

    unsigned long long* hlenCPU;
    unsigned long long* rlenCPU;
    unsigned int* seqCounts;

    __m256i *haplen4, *rslen4, *haplen8f, *rslen8f;
    __m512i *haplen8, *rslen8, *haplen16, *rslen16;

    if(precision == 32){
        finalsum32 = (float*)aligned_alloc(32, batchsize*sizeof(float));
        hapCPU32 = (unsigned int*)aligned_alloc(32, CPUsize*sizeof(unsigned int));
        readCPU32 = (unsigned int*)aligned_alloc(32, CPUsize*sizeof(unsigned int));
        qCPU32 = (unsigned int*)aligned_alloc(32, CPUsize*sizeof(unsigned int));
        iCPU32 = (unsigned int*)aligned_alloc(32, CPUsize*sizeof(unsigned int));
        dCPU32 = (unsigned int*)aligned_alloc(32, CPUsize*sizeof(unsigned int));
        qual32 = (float*)aligned_alloc(32, 128*sizeof(float));

        for(int a = 0; a < 128; a++) qual32[a] = powf(10.0f, -a/(float)10.0f);

        if(width == 256){
            haplen8f = (__m256i*)aligned_alloc(32, (batchsize/8 + 1)*sizeof(__m256i));
            rslen8f = (__m256i*)aligned_alloc(32, (batchsize/8 + 1)*sizeof(__m256i));

            hlenCPU = (unsigned long long*)aligned_alloc(32, (batchsize/8 + 1)*sizeof(unsigned long long));
            rlenCPU = (unsigned long long*)aligned_alloc(32, (batchsize/8 + 1)*sizeof(unsigned long long));
            seqCounts = (unsigned int*)aligned_alloc(32, 36*(batchsize/8 + 1)*sizeof(unsigned int));
            modulo = 8;
        }
        else{
            haplen16 = (__m512i*)aligned_alloc(64, (batchsize/16 + 1)*sizeof(__m512i));
            rslen16 = (__m512i*)aligned_alloc(64, (batchsize/16 + 1)*sizeof(__m512i));

            hlenCPU = (unsigned long long*)aligned_alloc(32, (batchsize/16 + 1)*sizeof(unsigned long long));
            rlenCPU = (unsigned long long*)aligned_alloc(32, (batchsize/16 + 1)*sizeof(unsigned long long));
            seqCounts = (unsigned int*)aligned_alloc(32, 36*(batchsize/16 + 1)*sizeof(unsigned int));
            modulo = 16;
        }
    }
    else{
        finalsum = (double*)aligned_alloc(32, batchsize*sizeof(double));
        hapCPU = (unsigned long long*)aligned_alloc(32, CPUsize*sizeof(unsigned long long));
        readCPU = (unsigned long long*)aligned_alloc(32, CPUsize*sizeof(unsigned long long));
        qCPU = (unsigned long long*)aligned_alloc(32, CPUsize*sizeof(unsigned long long));
        iCPU = (unsigned long long*)aligned_alloc(32, CPUsize*sizeof(unsigned long long));
        dCPU = (unsigned long long*)aligned_alloc(32, CPUsize*sizeof(unsigned long long));
        qual = (double*)aligned_alloc(32, 128*sizeof(double));

        for(int a = 0; a < 128; a++) qual[a] = pow(10, -a/(double)10);

        if(width == 256){
            haplen4 = (__m256i*)aligned_alloc(32, (batchsize/4 + 1)*sizeof(__m256i));
            rslen4 = (__m256i*)aligned_alloc(32, (batchsize/4 + 1)*sizeof(__m256i));

            hlenCPU = (unsigned long long*)aligned_alloc(32, (batchsize/4 + 1)*sizeof(unsigned long long));
            rlenCPU = (unsigned long long*)aligned_alloc(32, (batchsize/4 + 1)*sizeof(unsigned long long));
            seqCounts = (unsigned int*)aligned_alloc(32, 36*(batchsize/4 + 1)*sizeof(unsigned int));
            modulo = 4;
        }
        else{
            haplen8 = (__m512i*)aligned_alloc(64, (batchsize/8 + 1)*sizeof(__m512i));
            rslen8 = (__m512i*)aligned_alloc(64, (batchsize/8 + 1)*sizeof(__m512i));

            hlenCPU = (unsigned long long*)aligned_alloc(32, (batchsize/8 + 1)*sizeof(unsigned long long));
            rlenCPU = (unsigned long long*)aligned_alloc(32, (batchsize/8 + 1)*sizeof(unsigned long long));
            seqCounts = (unsigned int*)aligned_alloc(32, 36*(batchsize/8 + 1)*sizeof(unsigned int));
            modulo = 8;
        }
    }

    float total_time = 0, part_time = 0;

    unsigned long long i = 0, hcount = 0, rcount = 0, count = 0;
    unsigned int rslen[16] = {0}, haplen[16] = {0};

    unsigned int* seq_size = (unsigned int*)aligned_alloc(32, 36*sizeof(unsigned int));

    rlenCPU[0] = rcount;
    hlenCPU[0] = hcount;

    while(gzgets(gzfile, tmp, sizeof(tmp)) != Z_NULL){
        pch = strtok(tmp," ");
        while (idx != 7)
        {
            strcpy(buf[count%modulo][idx], pch);
            pch = strtok(NULL, " ");
            idx++;
        }

        idx = 0;
        count++;

        if(precision == 32){
            if(width == 256 && count%modulo == 0) process256_float(hapCPU32, readCPU32, qCPU32, iCPU32, dCPU32, rlenCPU, hlenCPU, rslen, haplen, haplen8f, rslen8f, buf, seq_size, seqCounts, count, batchsize, &rcount, &hcount, &gcups);
            else if(width == 512 && count%modulo == 0) process512_float(hapCPU32, readCPU32, qCPU32, iCPU32, dCPU32, rlenCPU, hlenCPU, rslen, haplen, haplen16, rslen16, buf, seq_size, seqCounts, count, batchsize, &rcount, &hcount, &gcups);
            
        }
        else{
            if(width == 256 && count%modulo == 0) process256_double(hapCPU, readCPU, qCPU, iCPU, dCPU, rlenCPU, hlenCPU, rslen, haplen, haplen4, rslen4, buf, seq_size, seqCounts, count, batchsize, &rcount, &hcount, &gcups);
            //else if(width == 512 && count%modulo == 0) process512_double();
        }


        if(count%batchsize == 0){

            if(precision == 32){
                if(width == 256){
                    total_time = pairhmm_process_fp32(rlenCPU, hlenCPU, hapCPU32, readCPU32, rslen8f, haplen8f, qCPU32, iCPU32, dCPU32, qual32, finalsum32, seq_size, seqCounts, batchsize);
                }
                else{
                    total_time = pairhmm_process_fp32_avx512(rlenCPU, hlenCPU, hapCPU32, readCPU32, rslen16, haplen16, qCPU32, iCPU32, dCPU32, qual32, finalsum32, seq_size, seqCounts, batchsize);
                }

                part_time += total_time;

                printf("THROUGHPUT: %.8f; TOTAL TIME: %8f\n", gcups/(double)(total_time*pow(10,12)), part_time);
                

                //for(int a = 0; a < batchsize; a++) fprintf(fpout, "%.48f\n", log10f(finalsum32[a]) - log10f(ldexpf(1.f,120.f)));
            }
            else{
                if(width == 256){
                    total_time = pairhmm_process_fp64(rlenCPU, hlenCPU, hapCPU, readCPU, rslen4, haplen4, qCPU, iCPU, dCPU, qual, finalsum, seq_size, seqCounts, batchsize);
                }
                else{
                    //total_time = pairhmm_benchmark_fp64_avx512(rlenCPU, hlenCPU, hapCPU, readCPU, qCPU, iCPU, dCPU, qual, finalsum, maxsize, batchsize);
                }

                part_time += total_time;

                printf("THROUGHPUT: %.8f; TOTAL TIME: %8f\n", gcups/(double)(total_time*pow(10,12)), part_time);

                //for(int a = 0; a < batchsize; a++) fprintf(fpout, "%.48f\n", log10(finalsum[a]) - log10(ldexp(1.0,1020.0)));
            }

            gcups = 0;
            count = 0;
            rcount = 0;
            hcount = 0;

            for(int a = 0; a < 36; a++) seq_size[a] = 0;

            rlenCPU[0] = rcount;
            hlenCPU[0] = hcount;
        }

    }


    free(hlenCPU);
    free(rlenCPU);
    free(seqCounts);
    free(seq_size);

    if(precision == 32){
        free(finalsum32);
        free(hapCPU32);
        free(readCPU32);
        free(qCPU32);
        free(iCPU32);
        free(dCPU32);
        free(qual32);

        if(width == 256){
            free(haplen8f);
            free(rslen8f);
        }
        else{{
            free(haplen16);
            free(rslen16);
        }}
    }
    else{
        free(finalsum);
        free(hapCPU);
        free(readCPU);
        free(qCPU);
        free(iCPU);
        free(dCPU);
        free(qual);

        if(width == 256){
            free(haplen4);
            free(rslen4);
        }
        else{{
            free(haplen8);
            free(rslen8);
        }}
    }

    fclose(fpout);
}

void benchmarkPairHMM(int precision, int width){
    int lengths[13] = {32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072};
    unsigned int batchsize = 1536, lenidx = 11;

    while(1){
        if(lenidx == 13) break;

        printf("Batchsize: %d; Sequence Lengths: %d; Precision: %d-bit; SIMD Width: %d-bits\n", batchsize, lengths[lenidx], precision, width);

        int maxsize = 0;
        unsigned long long gcups = 0, CPUsize = (unsigned long long)batchsize*(unsigned long long)lengths[lenidx];
        double throughput = 0;

        int modulo = 0;

        double* finalsum;
        double* qual;
        unsigned long long* hapCPU; 
        unsigned long long* readCPU; 
        unsigned long long* qCPU;
        unsigned long long* iCPU;
        unsigned long long* dCPU;

        float* finalsum32;
        float* qual32;
        unsigned int* hapCPU32; 
        unsigned int* readCPU32;
        unsigned int* qCPU32;
        unsigned int* iCPU32;
        unsigned int* dCPU32; 

        if(precision == 32){
            finalsum32 = (float*)aligned_alloc(32, batchsize*sizeof(float));
            hapCPU32 = (unsigned int*)aligned_alloc(32, CPUsize*sizeof(unsigned int));
            readCPU32 = (unsigned int*)aligned_alloc(32, CPUsize*sizeof(unsigned int));
            qCPU32 = (unsigned int*)aligned_alloc(32, CPUsize*sizeof(unsigned int));
            iCPU32 = (unsigned int*)aligned_alloc(32, CPUsize*sizeof(unsigned int));
            dCPU32 = (unsigned int*)aligned_alloc(32, CPUsize*sizeof(unsigned int));
            qual32 = (float*)aligned_alloc(32, 128*sizeof(float));

            if(width == 256) modulo = 8;
            else modulo = 16;

            for(int a = 0; a < 128; a++) qual32[a] = powf(10.0f, -a/(float)10.0f);
        }
        else{
            finalsum = (double*)aligned_alloc(32, batchsize*sizeof(double));
            hapCPU = (unsigned long long*)aligned_alloc(32, CPUsize*sizeof(unsigned long long));
            readCPU = (unsigned long long*)aligned_alloc(32, CPUsize*sizeof(unsigned long long));
            qCPU = (unsigned long long*)aligned_alloc(32, CPUsize*sizeof(unsigned long long));
            iCPU = (unsigned long long*)aligned_alloc(32, CPUsize*sizeof(unsigned long long));
            dCPU = (unsigned long long*)aligned_alloc(32, CPUsize*sizeof(unsigned long long));
            qual = (double*)aligned_alloc(32, 128*sizeof(double));

            if(width == 256) modulo = 4;
            else modulo = 8;

            for(int a = 0; a < 128; a++) qual[a] = pow(10, -a/(double)10);
        }

        unsigned long long* hlenCPU = (unsigned long long*)aligned_alloc(32, (batchsize + 1)*sizeof(unsigned long long));
        unsigned long long* rlenCPU = (unsigned long long*)aligned_alloc(32, (batchsize + 1)*sizeof(unsigned long long));

        float total_time = 0;

        unsigned long long i = 0, hcount = 0, rcount = 0;

        for(int b = 0; b < batchsize/modulo; b++){
            gcups += (unsigned long long)lengths[lenidx]*(unsigned long long)lengths[lenidx]*(unsigned long long)modulo;

            if(precision == 32){
                for(int c = 0; c < modulo*lengths[lenidx]; c++){
                    hapCPU32[hcount + c] = (c/modulo)%4;
                    readCPU32[rcount + c] = (c/modulo)%4;
                    qCPU32[rcount + c] = 63 + (c/modulo)%4;
                    iCPU32[rcount + c] = 73;
                    dCPU32[rcount + c] = 73;
                }
            }
            else{
                for(int c = 0; c < modulo*lengths[lenidx]; c++){
                    hapCPU[hcount + c] = (c/modulo)%4;
                    readCPU[rcount + c] = (c/modulo)%4;
                    qCPU[rcount + c] = 63 + (c/modulo)%4;
                    iCPU[rcount + c] = 73;
                    dCPU[rcount + c] = 73;
                }
            }


            for(int c = 0; c < modulo; c++){
                rlenCPU[i%batchsize] = rcount;
                hlenCPU[i%batchsize] = hcount;
                rcount += (unsigned long long)lengths[lenidx];
                hcount += (unsigned long long)lengths[lenidx];
                i++;
            }
        }

        maxsize = lengths[lenidx];

        rlenCPU[batchsize] = rcount;
        hlenCPU[batchsize] = hcount;

        for(int a = 0; a < 6; a++){
            if(precision == 32){
                if(width == 256){
                    total_time = pairhmm_benchmark_fp32(rlenCPU, hlenCPU, hapCPU32, readCPU32, qCPU32, iCPU32, dCPU32, qual32, finalsum32, maxsize, batchsize);
                }
                else{
                    total_time = pairhmm_benchmark_fp32_avx512(rlenCPU, hlenCPU, hapCPU32, readCPU32, qCPU32, iCPU32, dCPU32, qual32, finalsum32, maxsize, batchsize);
                }
            }
            else{
                if(width == 256){
                    total_time = pairhmm_benchmark_fp64(rlenCPU, hlenCPU, hapCPU, readCPU, qCPU, iCPU, dCPU, qual, finalsum, maxsize, batchsize);
                }
                else{
                    total_time = pairhmm_benchmark_fp64_avx512(rlenCPU, hlenCPU, hapCPU, readCPU, qCPU, iCPU, dCPU, qual, finalsum, maxsize, batchsize);
                }
            }

            if(a > 0) throughput += gcups/(double)(total_time*pow(10,12));   
            printf("%.6f\n", gcups/(double)(total_time*pow(10,12))); 
        }

        gcups = 0;
        maxsize = 0;
        rcount = 0;
        hcount = 0;
        i = 0; 

        printf("Throughput: %.6f TCUPS\n", throughput/5);

        if(precision == 32){
            free(finalsum32);
            free(hapCPU32);
            free(readCPU32);
            free(qCPU32);
            free(iCPU32);
            free(dCPU32);
            free(qual32);
        }
        else{
            free(finalsum);
            free(hapCPU);
            free(readCPU);
            free(qCPU);
            free(iCPU);
            free(dCPU);
            free(qual);
        }

        free(hlenCPU);
        free(rlenCPU);
        
        lenidx++;
        if(lenidx%2 != 0) batchsize /= 2;
    }
}

int main(int argc, char **argv) {
    Arguments args;
    
    for(int x = 1; x < argc; x++){
        string argstring = argv[x];
        if(argstring == "--inputfile"){
            args.inputfile = argv[x+1];
            x++;
        }
        if(argstring == "--outputfile"){
            args.outputfile = argv[x+1];
            x++;
        }  
        if(argstring == "--precision"){
            args.precision = std::atoi(argv[x+1]);
            x++;
        }
        if(argstring == "--width"){
            args.width = std::atoi(argv[x+1]);
            x++;
        }        
        if(argstring == "--benchmark"){
            args.benchmark = true;
        }
        if(argstring == "--batchsize"){
            args.batchsize = std::atoi(argv[x+1]);
            x++;
        }
        if(argstring == "--help"){
            args.help = true;
        }
    }

    //CHECKING ARGUMENTS

    //HELP ARGUMENT
    if(args.help){
        std::cout << "LIST OF ARGUMENTS FOR TRIOSEQ" << "\n\n";
        std::cout << "--inputfile <path_to_file> (String argument. File should contain a list of sequence triplets);" << "\n";
        std::cout << "--outputfile <path_to_file> (String argument. File stores alignment scores);" << "\n";
        std::cout << "--precision <num_bits> (Integer argument. Available precisions: 32 (single), and 64 (double). Default: 32);" << "\n";
        std::cout << "--width <simd_width> (Integer argument. Available widths: 256 (AVX), and 512 (AVX-512). Default: 256);" << "\n";
        std::cout << "--batchsize <num_sequences> (Integer argument. Selects the number of sequence pairs to evaluate in a single kernel run. Default: 16384);" << "\n";
        std::cout << "--benchmark (If this flag is selected, runs benchmark to evaluate PairHMM-CPU's peak performance. Default: false)." << "\n";
    
        return 0;
    }

    //PRECISION ARGUMENT
    if(args.precision != 32 && args.precision != 64){
        std::cout << "Error in --precision: " << args.precision << "-bit precision is not supported! Available precisions: 32 (single) and 64 (double)." << "\n";
        return 0;
    }

    //WIDTH ARGUMENT
    if(args.width != 256 && args.width != 512){
        std::cout << "Error in --width: " << args.width << "-bit SIMD width is not supported! Available widths: 256 (AVX) and 512 (AVX-512)." << "\n";
        return 0;
    }

    //BATCHSIZE ARGUMENT
    if(args.batchsize <= 0){
        std::cout << "Error in --batchsize: " << args.batchsize << " is invalid! Batchsize must be > 0." << "\n";
        return 0;
    }

    //BENCHMARK ARGUMENT
    if(args.benchmark){
        benchmarkPairHMM(args.precision, args.width);

        return 0;
    }

    //INPUTFILE ARGUMENT
    if(args.inputfile == ""){
        std::cout << "Error in --inputfile: no input dataset was specified." << "\n";
        return 0;
    }

    //OUTPUTFILE ARGUMENT
    if(args.outputfile == ""){
        std::cout << "Error in --outputfile: no output dataset was specified." << "\n";
        return 0;
    }

    processDataset(args.inputfile, args.outputfile, args.precision, args.width, args.batchsize);

    return 0;

}
