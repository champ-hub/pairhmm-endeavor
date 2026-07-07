#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <random>
#include <cuda.h>
#include <time.h>
#include <cuda_runtime.h>
#include "PFABench.cu"
#include "PFADataset.cu"
#include <zlib.h>

using namespace std;

struct Arguments{
    string inputfile = "";
    string outputfile = "";
    int precision = 32;
    bool benchmark = false;
    bool help = false;
    int batchsize = 16384;
};


void processDataset(string inputfile, string outputfile, int precision, int batchsize)
{
    gzFile gzfile = gzopen(inputfile.c_str(), "rb");
    
    FILE* fpout = fopen(outputfile.c_str(), "w");
	int idx = 0, j = 0;
    char buf[7][16385];
    char tmp[7*16385];

    if(gzgets(gzfile, tmp, sizeof(tmp)) != Z_NULL) j = 0;

    //Memory Allocation
    float* finalsum32;
    double* finalsum;
    unsigned int* blockDist;
    unsigned long long* hlenGPU;
    unsigned long long* rlenGPU;

    if(precision == 32){
        if (cudaMalloc(&finalsum32, batchsize * sizeof(float)) != cudaSuccess) {
            printf("Batchsize too big for GPU memory!");
            return;
        }
    }
    else{
        if (cudaMalloc(&finalsum, batchsize * sizeof(double)) != cudaSuccess) {
            printf("Batchsize too big for GPU memory!");
            return;
        }
    }

    if (cudaMalloc(&blockDist, batchsize * 39 * sizeof(unsigned int)) != cudaSuccess) {
        printf("Batchsize too big for GPU memory!");

        if(precision == 32) cudaFree(finalsum32);
        else cudaFree(finalsum);

        return;
    }

    if (cudaMalloc(&hlenGPU, (batchsize + 1) * sizeof(unsigned long long)) != cudaSuccess) {
        printf("Batchsize too big for GPU memory!");

        if(precision == 32) cudaFree(finalsum32);
        else cudaFree(finalsum);

        cudaFree(blockDist);

        return;
    }

    if (cudaMalloc(&rlenGPU, (batchsize + 1) * sizeof(unsigned long long)) != cudaSuccess) {
        printf("Batchsize too big for GPU memory!");

        if(precision == 32) cudaFree(finalsum32);
        else cudaFree(finalsum);

        cudaFree(blockDist);
        cudaFree(hlenGPU);

        return;
    }

    unsigned int maxsize = 0, blockidx = 0, i = 0, gcupsidx = 0, kernidx = 0, rslen = 0, haplen = 0;
    unsigned long long gcups = 0, CPUsize = (unsigned long long)batchsize*(unsigned long long)16384;
    double gcups_avg = 0;

    float* finalsum_32;
    double* finalsum_;

    if(precision == 32) finalsum_32 = (float*)calloc(batchsize, sizeof(float));
    else finalsum_ = (double*)calloc(batchsize, sizeof(double));

    unsigned long long* hlenCPU = (unsigned long long*)calloc(batchsize + 1, sizeof(unsigned long long));
    unsigned long long* rlenCPU = (unsigned long long*)calloc(batchsize + 1, sizeof(unsigned long long));
    unsigned char* hapCPU = (unsigned char*)calloc(CPUsize, sizeof(unsigned char));
    unsigned char* readCPU = (unsigned char*)calloc(CPUsize, sizeof(unsigned char));
    unsigned char* qCPU = (unsigned char*)calloc(CPUsize, sizeof(unsigned char));
    unsigned char* iCPU = (unsigned char*)calloc(CPUsize, sizeof(unsigned char));
    unsigned char* dCPU = (unsigned char*)calloc(CPUsize, sizeof(unsigned char));
    unsigned int* blockCounts = (unsigned int*)calloc(39, sizeof(unsigned int));

    float total_time = 0, part_time = 0;
    idx = 0;

    unsigned long long hcount = 0, rcount = 0;

    cudaStream_t stream1;
    cudaStreamCreate(&stream1);

    char* pch;

    while(gzgets(gzfile, tmp, sizeof(tmp)) != Z_NULL){
        pch = strtok(tmp," ");
        while (idx != 7)
        {
            strcpy(buf[idx], pch);
            pch = strtok(NULL, " ");
            idx++;
        }

        idx = 0;

        if(i%10000000 == 0) std::cout << i << " SEQUENCES ANALYSED!" << std::endl;

        haplen = strlen(buf[0]);
        rslen = strlen(buf[1]);

        gcups += (unsigned long long)rslen*(unsigned long long)haplen;

        rlenCPU[i%batchsize] = rcount;
        hlenCPU[i%batchsize] = hcount;

        for(int c = 0; c < haplen; c++) hapCPU[hcount + c] = buf[0][c];

        for(int c = 0; c < rslen; c++){
            readCPU[rcount + c] = buf[1][c];
            qCPU[rcount + c] = buf[2][c] - 33;
            iCPU[rcount + c] = buf[3][c] - 33;
            dCPU[rcount + c] = buf[4][c] - 33;
        }

        rcount += rslen;
        hcount += haplen;

        maxsize = max(haplen,rslen);
        if(maxsize > 1024){
            for(int j = 1; j <= 7; j++){
                if(maxsize <= 1024*pow(2,j)){
                    kernidx = 31 + j;
                    break;
                }
            }
        }
        else{
            for(int j = 0; j < 32; j++){
                if(maxsize <= 32*(j + 1)){
                    kernidx = j;
                    break;
                }
            }
        }

        blockidx = i%batchsize;
        cudaMemcpyAsync(&blockDist[kernidx*batchsize + blockCounts[kernidx]],&blockidx, sizeof(unsigned int),cudaMemcpyHostToDevice, stream1);
        blockCounts[kernidx]++;
            
        i++;
        
        if(i%batchsize == 0){
            rlenCPU[batchsize] = rcount;
            hlenCPU[batchsize] = hcount;

            //for(int j = 0; j < 10; j++) printf("%d ", blockCounts[j]);
            //printf("\n");

            unsigned char* hapGPU;
            unsigned char* readGPU;
            unsigned char* qGPU;
            unsigned char* iGPU;
            unsigned char* dGPU;
            
            if (cudaMalloc(&hapGPU, hcount * sizeof(unsigned char)) != cudaSuccess) {
                printf("Batchsize too big for GPU memory!\n");

                if(precision == 32){
                    cudaFree(finalsum32);
                    free(finalsum_32);
                }
                else{
                    cudaFree(finalsum);
                    free(finalsum_);
                }

                cudaFree(hlenGPU);
                cudaFree(rlenGPU);
                cudaFree(blockDist);
                free(hlenCPU);
                free(rlenCPU);
                free(hapCPU);
                free(readCPU);
                free(qCPU);
                free(iCPU);
                free(dCPU);
                free(blockCounts);
                return;
            }

            if (cudaMalloc(&readGPU, rcount * sizeof(unsigned char)) != cudaSuccess) {
                printf("Batchsize too big for GPU memory!");

                if(precision == 32){
                    cudaFree(finalsum32);
                    free(finalsum_32);
                }
                else{
                    cudaFree(finalsum);
                    free(finalsum_);
                }

                cudaFree(hlenGPU);
                cudaFree(rlenGPU);
                cudaFree(hapGPU);
                cudaFree(blockDist);
                free(hlenCPU);
                free(rlenCPU);
                free(hapCPU);
                free(readCPU);
                free(qCPU);
                free(iCPU);
                free(dCPU);
                free(blockCounts);
                return;
            }

            if (cudaMalloc(&qGPU, rcount * sizeof(unsigned char)) != cudaSuccess) {
                printf("Batchsize too big for GPU memory!");

                if(precision == 32){
                    cudaFree(finalsum32);
                    free(finalsum_32);
                }
                else{
                    cudaFree(finalsum);
                    free(finalsum_);
                }

                cudaFree(hlenGPU);
                cudaFree(rlenGPU);
                cudaFree(hapGPU);
                cudaFree(readGPU);
                cudaFree(blockDist);
                free(hlenCPU);
                free(rlenCPU);
                free(hapCPU);
                free(readCPU);
                free(qCPU);
                free(iCPU);
                free(dCPU);
                free(blockCounts);
                return;
            }

            if (cudaMalloc(&iGPU, rcount * sizeof(unsigned char)) != cudaSuccess) {
                printf("Batchsize too big for GPU memory!");

                if(precision == 32){
                    cudaFree(finalsum32);
                    free(finalsum_32);
                }
                else{
                    cudaFree(finalsum);
                    free(finalsum_);
                }

                cudaFree(hlenGPU);
                cudaFree(rlenGPU);
                cudaFree(hapGPU);
                cudaFree(readGPU);
                cudaFree(blockDist);
                cudaFree(qGPU);
                free(hlenCPU);
                free(rlenCPU);
                free(hapCPU);
                free(readCPU);
                free(qCPU);
                free(iCPU);
                free(dCPU);
                free(blockCounts);
                return;
            }

            if (cudaMalloc(&dGPU, rcount * sizeof(unsigned char)) != cudaSuccess) {
                printf("Batchsize too big for GPU memory!");

                if(precision == 32){
                    cudaFree(finalsum32);
                    free(finalsum_32);
                }
                else{
                    cudaFree(finalsum);
                    free(finalsum_);
                }

                cudaFree(hlenGPU);
                cudaFree(rlenGPU);
                cudaFree(hapGPU);
                cudaFree(qGPU);
                cudaFree(iGPU);
                cudaFree(blockDist);
                free(hlenCPU);
                free(rlenCPU);
                free(hapCPU);
                free(readCPU);
                free(qCPU);
                free(iCPU);
                free(dCPU);
                free(blockCounts);
                return;
            }
            

            cudaMemcpy(hapGPU, hapCPU, sizeof(unsigned char)*hcount, cudaMemcpyHostToDevice);
            cudaMemcpy(readGPU, readCPU, sizeof(unsigned char)*rcount, cudaMemcpyHostToDevice);
            cudaMemcpy(qGPU, qCPU, sizeof(unsigned char)*rcount, cudaMemcpyHostToDevice);
            cudaMemcpy(iGPU, iCPU, sizeof(unsigned char)*rcount, cudaMemcpyHostToDevice);
            cudaMemcpy(dGPU, dCPU, sizeof(unsigned char)*rcount, cudaMemcpyHostToDevice);
            cudaMemcpy(rlenGPU, rlenCPU, sizeof(unsigned long long)*(batchsize + 1), cudaMemcpyHostToDevice);
            cudaMemcpy(hlenGPU, hlenCPU, sizeof(unsigned long long)*(batchsize + 1), cudaMemcpyHostToDevice);
            cudaStreamSynchronize(stream1);

            if(precision == 32) part_time = runKernelsFP32(rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, finalsum32, finalsum_32, blockDist, blockCounts, fpout, maxsize, batchsize);
            else if(precision == 64) part_time = runKernelsFP64(rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, finalsum, finalsum_, blockDist, blockCounts, fpout, maxsize, batchsize);

            for(int j = 0; j < 39; j++) blockCounts[j] = 0;

            total_time += part_time;
            gcups_avg += gcups/(double)(part_time*pow(10,9));
            gcupsidx++;
            gcups = 0;
            maxsize = 0;
            hcount = 0;
            rcount = 0;

            printf("TIME: %.8f SECONDS; TCUPS (AVERAGE): %.8f\n", total_time/1000, gcups_avg/(double)(gcupsidx));

            
            cudaFree(hapGPU);
            cudaFree(readGPU);
            cudaFree(qGPU);
            cudaFree(iGPU);
            cudaFree(dGPU); 
        }   
    }

    if(i%batchsize != 0){
        rlenCPU[i%batchsize] = rcount;
        hlenCPU[i%batchsize] = hcount;

        unsigned char* hapGPU;
        unsigned char* readGPU;
        unsigned char* qGPU;
        unsigned char* iGPU;
        unsigned char* dGPU;

        if (cudaMalloc(&hapGPU, hcount * sizeof(unsigned char)) != cudaSuccess) {
            printf("Batchsize too big for GPU memory!.");

            if(precision == 32){
                cudaFree(finalsum32);
                free(finalsum_32);
            }
            else{
                cudaFree(finalsum);
                free(finalsum_);
            }

            cudaFree(hlenGPU);
            cudaFree(rlenGPU);
            cudaFree(blockDist);
            free(hlenCPU);
            free(rlenCPU);
            free(hapCPU);
            free(readCPU);
            free(qCPU);
            free(iCPU);
            free(dCPU);
            free(blockCounts);
            return;
        }

        if (cudaMalloc(&readGPU, rcount * sizeof(unsigned char)) != cudaSuccess) {
            printf("Batchsize too big for GPU memory!");

            if(precision == 32){
                cudaFree(finalsum32);
                free(finalsum_32);
            }
            else{
                cudaFree(finalsum);
                free(finalsum_);
            }

            cudaFree(hlenGPU);
            cudaFree(rlenGPU);
            cudaFree(hapGPU);
            cudaFree(blockDist);
            free(hlenCPU);
            free(rlenCPU);
            free(hapCPU);
            free(readCPU);
            free(qCPU);
            free(iCPU);
            free(dCPU);
            free(blockCounts);
            return;
        }

        if (cudaMalloc(&qGPU, rcount * sizeof(unsigned char)) != cudaSuccess) {
            printf("Batchsize too big for GPU memory!");

            if(precision == 32){
                cudaFree(finalsum32);
                free(finalsum_32);
            }
            else{
                cudaFree(finalsum);
                free(finalsum_);
            }

            cudaFree(hlenGPU);
            cudaFree(rlenGPU);
            cudaFree(hapGPU);
            cudaFree(readGPU);
            cudaFree(blockDist);
            free(hlenCPU);
            free(rlenCPU);
            free(hapCPU);
            free(readCPU);
            free(qCPU);
            free(iCPU);
            free(dCPU);
            free(blockCounts);
            return;
        }

        if (cudaMalloc(&iGPU, rcount * sizeof(unsigned char)) != cudaSuccess) {
            printf("Batchsize too big for GPU memory!");

            if(precision == 32){
                cudaFree(finalsum32);
                free(finalsum_32);
            }
            else{
                cudaFree(finalsum);
                free(finalsum_);
            }

            cudaFree(hlenGPU);
            cudaFree(rlenGPU);
            cudaFree(hapGPU);
            cudaFree(readGPU);
            cudaFree(blockDist);
            cudaFree(qGPU);
            free(hlenCPU);
            free(rlenCPU);
            free(hapCPU);
            free(readCPU);
            free(qCPU);
            free(iCPU);
            free(dCPU);
            free(blockCounts);
            return;
        }

        if (cudaMalloc(&dGPU, rcount * sizeof(unsigned char)) != cudaSuccess) {
            printf("Batchsize too big for GPU memory!");

            if(precision == 32){
                cudaFree(finalsum32);
                free(finalsum_32);
            }
            else{
                cudaFree(finalsum);
                free(finalsum_);
            }

            cudaFree(hlenGPU);
            cudaFree(rlenGPU);
            cudaFree(hapGPU);
            cudaFree(qGPU);
            cudaFree(iGPU);
            cudaFree(blockDist);
            free(hlenCPU);
            free(rlenCPU);
            free(hapCPU);
            free(readCPU);
            free(qCPU);
            free(iCPU);
            free(dCPU);
            free(blockCounts);
            return;
        }

        cudaMemcpy(hapGPU, hapCPU, sizeof(unsigned char)*hcount, cudaMemcpyHostToDevice);
        cudaMemcpy(readGPU, readCPU, sizeof(unsigned char)*rcount, cudaMemcpyHostToDevice);
        cudaMemcpy(qGPU, qCPU, sizeof(unsigned char)*rcount, cudaMemcpyHostToDevice);
        cudaMemcpy(iGPU, iCPU, sizeof(unsigned char)*rcount, cudaMemcpyHostToDevice);
        cudaMemcpy(dGPU, dCPU, sizeof(unsigned char)*rcount, cudaMemcpyHostToDevice);
        cudaMemcpy(rlenGPU, rlenCPU, sizeof(unsigned long long)*(batchsize + 1), cudaMemcpyHostToDevice);
        cudaMemcpy(hlenGPU, hlenCPU, sizeof(unsigned long long)*(batchsize + 1), cudaMemcpyHostToDevice);
        cudaStreamSynchronize(stream1);

        if(precision == 32) part_time = runKernelsFP32(rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, finalsum32, finalsum_32, blockDist, blockCounts, fpout, maxsize, batchsize);
        else if(precision == 64) part_time = runKernelsFP64(rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, finalsum, finalsum_, blockDist, blockCounts, fpout, maxsize, batchsize);

        for(int j = 0; j < 39; j++) blockCounts[j] = 0;
        
        total_time += part_time;
        gcups_avg += gcups/(double)(part_time*pow(10,9));
        gcupsidx++;

        printf("TIME: %.8f SECONDS; TCUPS (AVERAGE): %.8f\n", total_time/1000, gcups_avg/(double)(gcupsidx));

        cudaFree(hapGPU);
        cudaFree(readGPU);
        cudaFree(qGPU);
        cudaFree(iGPU);
        cudaFree(dGPU); 
    }
    
    printf("TIME: %.8f SECONDS; TCUPS (AVERAGE): %.8f\n", total_time/1000, gcups_avg/(double)(gcupsidx));
    
    
    cudaFree(hlenGPU);
    cudaFree(rlenGPU);
    cudaFree(blockDist);

    if(precision == 32){
        cudaFree(finalsum32);
        free(finalsum_32);
    }
    else{
        cudaFree(finalsum);
        free(finalsum_);
    }

    free(hlenCPU);
    free(rlenCPU);
    free(hapCPU);
    free(readCPU);
    free(qCPU);
    free(iCPU);
    free(dCPU);
    free(blockCounts);

    fclose(fpout);
    gzclose(gzfile);
}



void benchmarkPairHMM(int precision){
    int lengths[13] = {32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072};
    unsigned int batchsize = 4194304, lenidx = 0;

    while(1){
        if(lenidx == 13) break;

        printf("Batchsize: %d; Sequence Lengths: %d; Precision: %d-bit\n", batchsize, lengths[lenidx], precision);

        float* finalsum32;
        double* finalsum;

        if(precision == 32){
            if (cudaMalloc(&finalsum32, batchsize * sizeof(float)) != cudaSuccess) {
                printf("Batchsize too big for GPU memory! Reducing batchsize by half.\n");
                batchsize /= 2;
                continue;
            }
        }
        else{
            if (cudaMalloc(&finalsum, batchsize * sizeof(double)) != cudaSuccess) {
                printf("Batchsize too big for GPU memory! Reducing batchsize by half.\n");
                batchsize /= 2;
                continue;
            }
        }

        unsigned long long* hlenGPU;
        unsigned long long* rlenGPU;
        if (cudaMalloc(&hlenGPU, (batchsize + 1) * sizeof(unsigned long long)) != cudaSuccess) {
            printf("Batchsize too big for GPU memory! Reducing batchsize by half.\n");
            batchsize /= 2;
            if(precision == 32) cudaFree(finalsum32);
            else cudaFree(finalsum);

            continue;
        }

        if (cudaMalloc(&rlenGPU, (batchsize + 1) * sizeof(unsigned long long)) != cudaSuccess) {
            printf("Batchsize too big for GPU memory! Reducing batchsize by half.\n");
            batchsize /= 2;
            if(precision == 32) cudaFree(finalsum32);
            else cudaFree(finalsum);

            cudaFree(hlenGPU);
            continue;
        }

        int maxsize = 0;
        unsigned long long gcups = 0, CPUsize = (unsigned long long)batchsize*(unsigned long long)lengths[lenidx];
        double throughput = 0;

        float* finalsum_32;
        double* finalsum_;

        if(precision == 32) finalsum_32 = (float*)calloc(batchsize, sizeof(float));
        else finalsum_ = (double*)calloc(batchsize, sizeof(double));

        unsigned long long* hlenCPU = (unsigned long long*)calloc(batchsize + 1, sizeof(unsigned long long));
        unsigned long long* rlenCPU = (unsigned long long*)calloc(batchsize + 1, sizeof(unsigned long long));
        unsigned char* hapCPU = (unsigned char*)calloc(CPUsize, sizeof(unsigned char));
        unsigned char* readCPU = (unsigned char*)calloc(CPUsize, sizeof(unsigned char));
        unsigned char* qCPU = (unsigned char*)calloc(CPUsize, sizeof(unsigned char));
        unsigned char* iCPU = (unsigned char*)calloc(CPUsize, sizeof(unsigned char));
        unsigned char* dCPU = (unsigned char*)calloc(CPUsize, sizeof(unsigned char));

        float total_time = 0;

        unsigned long long i = 0, hcount = 0, rcount = 0;
        unsigned char* hapGPU;
        unsigned char* readGPU;
        unsigned char* qGPU;
        unsigned char* iGPU;
        unsigned char* dGPU;

        for(int b = 0; b < batchsize; b++){
            gcups += (unsigned long long)lengths[lenidx]*(unsigned long long)lengths[lenidx];

            rlenCPU[i%batchsize] = rcount;
            hlenCPU[i%batchsize] = hcount;

            for(int c = 0; c < lengths[lenidx]; c++){
                hapCPU[hcount + c] = c%4;
                readCPU[rcount + c] = c%4;
                qCPU[hcount + c] = 30;
                iCPU[rcount + c] = 40;
                dCPU[hcount + c] = 40;
            }

            rcount += lengths[lenidx];
            hcount += lengths[lenidx];

            i++;
        }

        maxsize = lengths[lenidx];

        rlenCPU[batchsize] = rcount;
        hlenCPU[batchsize] = hcount;

        if (cudaMalloc(&hapGPU, hcount * sizeof(unsigned char)) != cudaSuccess) {
            printf("Batchsize too big for GPU memory! Reducing batchsize by half.\n");
            batchsize /= 2;
            if(precision == 32){
                cudaFree(finalsum32);
                free(finalsum_32);
            }
            else{
                cudaFree(finalsum);
                free(finalsum_);
            }

            cudaFree(hlenGPU);
            cudaFree(rlenGPU);
            free(hlenCPU);
            free(rlenCPU);
            free(hapCPU);
            free(readCPU);
            free(qCPU);
            free(iCPU);
            free(dCPU);
            continue;
        }

        if (cudaMalloc(&readGPU, rcount * sizeof(unsigned char)) != cudaSuccess) {
            printf("Batchsize too big for GPU memory! Reducing batchsize by half.\n");
            batchsize /= 2;
            if(precision == 32){
                cudaFree(finalsum32);
                free(finalsum_32);
            }
            else{
                cudaFree(finalsum);
                free(finalsum_);
            }

            cudaFree(hlenGPU);
            cudaFree(rlenGPU);
            cudaFree(hapGPU);
            free(hlenCPU);
            free(rlenCPU);
            free(hapCPU);
            free(readCPU);
            free(qCPU);
            free(iCPU);
            free(dCPU);
            continue;
        }

        if (cudaMalloc(&qGPU, rcount * sizeof(unsigned char)) != cudaSuccess) {
            printf("Batchsize too big for GPU memory! Reducing batchsize by half.\n");
            batchsize /= 2;
            if(precision == 32){
                cudaFree(finalsum32);
                free(finalsum_32);
            }
            else{
                cudaFree(finalsum);
                free(finalsum_);
            }

            cudaFree(hlenGPU);
            cudaFree(rlenGPU);
            cudaFree(hapGPU);
            cudaFree(readGPU);
            free(hlenCPU);
            free(rlenCPU);
            free(hapCPU);
            free(readCPU);
            free(qCPU);
            free(iCPU);
            free(dCPU);
            continue;
        }

        if (cudaMalloc(&iGPU, rcount * sizeof(unsigned char)) != cudaSuccess) {
            printf("Batchsize too big for GPU memory! Reducing batchsize by half.\n");
            batchsize /= 2;
            if(precision == 32){
                cudaFree(finalsum32);
                free(finalsum_32);
            }
            else{
                cudaFree(finalsum);
                free(finalsum_);
            }

            cudaFree(hlenGPU);
            cudaFree(rlenGPU);
            cudaFree(hapGPU);
            cudaFree(readGPU);
            cudaFree(qGPU);
            free(hlenCPU);
            free(rlenCPU);
            free(hapCPU);
            free(readCPU);
            free(qCPU);
            free(iCPU);
            free(dCPU);
            continue;
        }

        if (cudaMalloc(&dGPU, rcount * sizeof(unsigned char)) != cudaSuccess) {
            printf("Batchsize too big for GPU memory! Reducing batchsize by half.\n");
            batchsize /= 2;
            if(precision == 32){
                cudaFree(finalsum32);
                free(finalsum_32);
            }
            else{
                cudaFree(finalsum);
                free(finalsum_);
            }

            cudaFree(hlenGPU);
            cudaFree(rlenGPU);
            cudaFree(hapGPU);
            cudaFree(qGPU);
            cudaFree(iGPU);
            free(hlenCPU);
            free(rlenCPU);
            free(hapCPU);
            free(readCPU);
            free(qCPU);
            free(iCPU);
            free(dCPU);
            continue;
        }

        cudaMemcpy(hapGPU, hapCPU, sizeof(unsigned char)*hcount, cudaMemcpyHostToDevice);
        cudaMemcpy(readGPU, readCPU, sizeof(unsigned char)*rcount, cudaMemcpyHostToDevice);
        cudaMemcpy(qGPU, qCPU, sizeof(unsigned char)*rcount, cudaMemcpyHostToDevice);
        cudaMemcpy(iGPU, iCPU, sizeof(unsigned char)*rcount, cudaMemcpyHostToDevice);
        cudaMemcpy(dGPU, dCPU, sizeof(unsigned char)*rcount, cudaMemcpyHostToDevice);
        cudaMemcpy(rlenGPU, rlenCPU, sizeof(unsigned long long)*(batchsize + 1), cudaMemcpyHostToDevice);
        cudaMemcpy(hlenGPU, hlenCPU, sizeof(unsigned long long)*(batchsize + 1), cudaMemcpyHostToDevice);

        for(int a = 0; a < 6; a++){
            printf("%d\n",a);
                
            if(precision == 32) total_time = pairhmm_benchmark_fp32(rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, finalsum32, finalsum_32, maxsize, batchsize);
            else total_time = pairhmm_benchmark_fp64(rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, finalsum, finalsum_, maxsize, batchsize);

            if(a > 0) throughput += gcups/(double)(total_time*pow(10,9));
        }

        gcups = 0;
        maxsize = 0;
        rcount = 0;
        hcount = 0;

        printf("Throughput: %.6f TCUPS\n", throughput/5);

        cudaFree(hapGPU);
        cudaFree(readGPU);
        cudaFree(qGPU);
        cudaFree(iGPU);
        cudaFree(dGPU); 
        cudaFree(hlenGPU);
        cudaFree(rlenGPU);
        if(precision == 32){
            cudaFree(finalsum32);
            free(finalsum_32);
        }
        else{
            cudaFree(finalsum);
            free(finalsum_);
        }

        free(hlenCPU);
        free(rlenCPU);
        free(hapCPU);
        free(readCPU);
        free(qCPU);
        free(iCPU);
        free(dCPU);

        lenidx++;
        i = 0;
        if(lenidx%3 == 0) batchsize /= 2;
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
        std::cout << "--precision <num_bits> (Integer argument. Available precisions: 32 (single), and 64 (double). Default: 16);" << "\n";
        std::cout << "--batchsize <num_sequences> (Integer argument. Selects the number of sequence triplets to evaluate in a single kernel run. Default: 16384);" << "\n";
        std::cout << "--benchmark (If this flag is selected, runs benchmark to evaluate TrioSeq's peak performance. Default: false)." << "\n";
    
        return 0;
    }

    //PRECISION ARGUMENT
    if(args.precision != 64 && args.precision != 32){
        std::cout << "Error in --precision: " << args.precision << "-bit precision is not supported! Available precisions: 32 (single) and 64 (double)." << "\n";
        return 0;
    }

    //BATCHSIZE ARGUMENT
    if(args.batchsize <= 0){
        std::cout << "Error in --batchsize: " << args.precision << " is invalid! Batchsize must be > 0." << "\n";
        return 0;
    }

    //BENCHMARK ARGUMENT
    if(args.benchmark){
        benchmarkPairHMM(args.precision);

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

    processDataset(args.inputfile, args.outputfile, args.precision, args.batchsize);
    
    return 0;

}
