#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <random>
#include <time.h>
#include <hip/hip_runtime.h>
#include "PFABench.hip"
#include "PFADataset.hip"
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
	int max_ = 0, idx = 0, j = 0, _i = 0;
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
        if (hipMalloc(&finalsum32, batchsize * sizeof(float)) != hipSuccess) {
            printf("Batchsize too big for GPU memory!");
            return;
        }
    }
    else{
        if (hipMalloc(&finalsum, batchsize * sizeof(double)) != hipSuccess) {
            printf("Batchsize too big for GPU memory!");
            return;
        }
    }

    if (hipMalloc(&blockDist, batchsize * 36 * sizeof(unsigned int)) != hipSuccess) {
        printf("Batchsize too big for GPU memory!");

        if(precision == 32) hipFree(finalsum32);
        else hipFree(finalsum);

        return;
    }

    if (hipMalloc(&hlenGPU, (batchsize + 1) * sizeof(unsigned long long)) != hipSuccess) {
        printf("Batchsize too big for GPU memory!");

        if(precision == 32) hipFree(finalsum32);
        else hipFree(finalsum);

        hipFree(blockDist);

        return;
    }

    if (hipMalloc(&rlenGPU, (batchsize + 1) * sizeof(unsigned long long)) != hipSuccess) {
        printf("Batchsize too big for GPU memory!");

        if(precision == 32) hipFree(finalsum32);
        else hipFree(finalsum);

        hipFree(blockDist);
        hipFree(hlenGPU);

        return;
    }

    unsigned int maxsize = 0, blockidx = 0, i = 0, gcupsidx = 0, kernidx = 0, rslen = 0, haplen = 0;
    unsigned long long gcups = 0, CPUsize = (unsigned long long)batchsize*(unsigned long long)16384;
    double throughput = 0, gcups_avg = 0;

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
    unsigned int* blockCounts = (unsigned int*)calloc(36, sizeof(unsigned int));

    float total_time = 0, part_time = 0;
    idx = 0;

    unsigned long long hcount = 0, rcount = 0;
    unsigned char* hapGPU;
    unsigned char* readGPU;
    unsigned char* qGPU;
    unsigned char* iGPU;
    unsigned char* dGPU;

    hipStream_t stream1;
    hipStreamCreate(&stream1);

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
            for(int j = 1; j <= 4; j++){
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
        hipMemcpyAsync(&blockDist[kernidx*batchsize + blockCounts[kernidx]],&blockidx, sizeof(unsigned int),hipMemcpyHostToDevice, stream1);
        blockCounts[kernidx]++;
            
        i++;
        
        if(i%batchsize == 0){
            rlenCPU[batchsize] = rcount;
            hlenCPU[batchsize] = hcount;


            if (hipMalloc(&hapGPU, hcount * sizeof(unsigned char)) != hipSuccess) {
                printf("Batchsize too big for GPU memory!.");

                if(precision == 32){
                    hipFree(finalsum32);
                    free(finalsum_32);
                }
                else{
                    hipFree(finalsum);
                    free(finalsum_);
                }

                hipFree(hlenGPU);
                hipFree(rlenGPU);
                hipFree(blockDist);
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

            if (hipMalloc(&readGPU, rcount * sizeof(unsigned char)) != hipSuccess) {
                printf("Batchsize too big for GPU memory!");

                if(precision == 32){
                    hipFree(finalsum32);
                    free(finalsum_32);
                }
                else{
                    hipFree(finalsum);
                    free(finalsum_);
                }

                hipFree(hlenGPU);
                hipFree(rlenGPU);
                hipFree(hapGPU);
                hipFree(blockDist);
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

            if (hipMalloc(&qGPU, rcount * sizeof(unsigned char)) != hipSuccess) {
                printf("Batchsize too big for GPU memory!");

                if(precision == 32){
                    hipFree(finalsum32);
                    free(finalsum_32);
                }
                else{
                    hipFree(finalsum);
                    free(finalsum_);
                }

                hipFree(hlenGPU);
                hipFree(rlenGPU);
                hipFree(hapGPU);
                hipFree(readGPU);
                hipFree(blockDist);
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

            if (hipMalloc(&iGPU, rcount * sizeof(unsigned char)) != hipSuccess) {
                printf("Batchsize too big for GPU memory!");

                if(precision == 32){
                    hipFree(finalsum32);
                    free(finalsum_32);
                }
                else{
                    hipFree(finalsum);
                    free(finalsum_);
                }

                hipFree(hlenGPU);
                hipFree(rlenGPU);
                hipFree(hapGPU);
                hipFree(readGPU);
                hipFree(blockDist);
                hipFree(qGPU);
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

            if (hipMalloc(&dGPU, rcount * sizeof(unsigned char)) != hipSuccess) {
                printf("Batchsize too big for GPU memory!");

                if(precision == 32){
                    hipFree(finalsum32);
                    free(finalsum_32);
                }
                else{
                    hipFree(finalsum);
                    free(finalsum_);
                }

                hipFree(hlenGPU);
                hipFree(rlenGPU);
                hipFree(hapGPU);
                hipFree(qGPU);
                hipFree(iGPU);
                hipFree(blockDist);
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

            hipMemcpy(hapGPU, hapCPU, sizeof(unsigned char)*hcount, hipMemcpyHostToDevice);
            hipMemcpy(readGPU, readCPU, sizeof(unsigned char)*rcount, hipMemcpyHostToDevice);
            hipMemcpy(qGPU, qCPU, sizeof(unsigned char)*rcount, hipMemcpyHostToDevice);
            hipMemcpy(iGPU, iCPU, sizeof(unsigned char)*rcount, hipMemcpyHostToDevice);
            hipMemcpy(dGPU, dCPU, sizeof(unsigned char)*rcount, hipMemcpyHostToDevice);
            hipMemcpy(rlenGPU, rlenCPU, sizeof(unsigned long long)*(batchsize + 1), hipMemcpyHostToDevice);
            hipMemcpy(hlenGPU, hlenCPU, sizeof(unsigned long long)*(batchsize + 1), hipMemcpyHostToDevice);
            hipStreamSynchronize(stream1);

            if(precision == 32) part_time = runKernelsFP32(rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, finalsum32, finalsum_32, blockDist, blockCounts, fpout, maxsize, batchsize);
            else if(precision == 64) part_time = runKernelsFP64(rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, finalsum, finalsum_, blockDist, blockCounts, fpout, maxsize, batchsize);

            for(int j = 0; j < 36; j++) blockCounts[j] = 0;

            total_time += part_time;
            gcups_avg += gcups/(double)(part_time*pow(10,9));
            gcupsidx++;
            gcups = 0;
            maxsize = 0;
            hcount = 0;
            rcount = 0;

            printf("TIME: %.8f SECONDS; TCUPS (AVERAGE): %.8f\n", total_time/1000, gcups_avg/(double)(gcupsidx));

            hipFree(hapGPU);
            hipFree(readGPU);
            hipFree(qGPU);
            hipFree(iGPU);
            hipFree(dGPU); 
        }   
    }

    if(i%batchsize != 0){
        rlenCPU[i%batchsize] = rcount;
        hlenCPU[i%batchsize] = hcount;


        if (hipMalloc(&hapGPU, hcount * sizeof(unsigned char)) != hipSuccess) {
            printf("Batchsize too big for GPU memory!.");

            if(precision == 32){
                hipFree(finalsum32);
                free(finalsum_32);
            }
            else{
                hipFree(finalsum);
                free(finalsum_);
            }

            hipFree(hlenGPU);
            hipFree(rlenGPU);
            hipFree(blockDist);
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

        if (hipMalloc(&readGPU, rcount * sizeof(unsigned char)) != hipSuccess) {
            printf("Batchsize too big for GPU memory!");

            if(precision == 32){
                hipFree(finalsum32);
                free(finalsum_32);
            }
            else{
                hipFree(finalsum);
                free(finalsum_);
            }

            hipFree(hlenGPU);
            hipFree(rlenGPU);
            hipFree(hapGPU);
            hipFree(blockDist);
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

        if (hipMalloc(&qGPU, rcount * sizeof(unsigned char)) != hipSuccess) {
            printf("Batchsize too big for GPU memory!");

            if(precision == 32){
                hipFree(finalsum32);
                free(finalsum_32);
            }
            else{
                hipFree(finalsum);
                free(finalsum_);
            }

            hipFree(hlenGPU);
            hipFree(rlenGPU);
            hipFree(hapGPU);
            hipFree(readGPU);
            hipFree(blockDist);
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

        if (hipMalloc(&iGPU, rcount * sizeof(unsigned char)) != hipSuccess) {
            printf("Batchsize too big for GPU memory!");

            if(precision == 32){
                hipFree(finalsum32);
                free(finalsum_32);
            }
            else{
                hipFree(finalsum);
                free(finalsum_);
            }

            hipFree(hlenGPU);
            hipFree(rlenGPU);
            hipFree(hapGPU);
            hipFree(readGPU);
            hipFree(blockDist);
            hipFree(qGPU);
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

        if (hipMalloc(&dGPU, rcount * sizeof(unsigned char)) != hipSuccess) {
            printf("Batchsize too big for GPU memory!");

            if(precision == 32){
                hipFree(finalsum32);
                free(finalsum_32);
            }
            else{
                hipFree(finalsum);
                free(finalsum_);
            }

            hipFree(hlenGPU);
            hipFree(rlenGPU);
            hipFree(hapGPU);
            hipFree(qGPU);
            hipFree(iGPU);
            hipFree(blockDist);
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

        hipMemcpy(hapGPU, hapCPU, sizeof(unsigned char)*hcount, hipMemcpyHostToDevice);
        hipMemcpy(readGPU, readCPU, sizeof(unsigned char)*rcount, hipMemcpyHostToDevice);
        hipMemcpy(qGPU, qCPU, sizeof(unsigned char)*rcount, hipMemcpyHostToDevice);
        hipMemcpy(iGPU, iCPU, sizeof(unsigned char)*rcount, hipMemcpyHostToDevice);
        hipMemcpy(dGPU, dCPU, sizeof(unsigned char)*rcount, hipMemcpyHostToDevice);
        hipMemcpy(rlenGPU, rlenCPU, sizeof(unsigned long long)*(batchsize + 1), hipMemcpyHostToDevice);
        hipMemcpy(hlenGPU, hlenCPU, sizeof(unsigned long long)*(batchsize + 1), hipMemcpyHostToDevice);
        hipStreamSynchronize(stream1);

        if(precision == 32) part_time = runKernelsFP32(rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, finalsum32, finalsum_32, blockDist, blockCounts, fpout, maxsize, batchsize);
        else if(precision == 64) part_time = runKernelsFP64(rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, finalsum, finalsum_, blockDist, blockCounts, fpout, maxsize, batchsize);

        for(int j = 0; j < 36; j++) blockCounts[j] = 0;
        
        total_time += part_time;
        gcups_avg += gcups/(double)(part_time*pow(10,9));
        gcupsidx++;
    }
    
    printf("TIME: %.8f SECONDS; TCUPS (AVERAGE): %.8f\n", total_time/1000, gcups_avg/(double)(gcupsidx));
    
    hipFree(hapGPU);
    hipFree(readGPU);
    hipFree(qGPU);
    hipFree(iGPU);
    hipFree(dGPU); 
    hipFree(hlenGPU);
    hipFree(rlenGPU);
    hipFree(blockDist);

    if(precision == 32){
        hipFree(finalsum32);
        free(finalsum_32);
    }
    else{
        hipFree(finalsum);
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
    unsigned int batchsize = 4096, lenidx = 11;

    while(1){
        if(lenidx == 13) break;

        printf("Batchsize: %d; Sequence Lengths: %d; Precision: %d-bit\n", batchsize, lengths[lenidx], precision);

        float* finalsum32;
        double* finalsum;

        if(precision == 32){
            if (hipMalloc(&finalsum32, batchsize * sizeof(float)) != hipSuccess) {
                printf("Batchsize too big for GPU memory! Reducing batchsize by half.");
                batchsize /= 2;
                continue;
            }
        }
        else{
            if (hipMalloc(&finalsum, batchsize * sizeof(double)) != hipSuccess) {
                printf("Batchsize too big for GPU memory! Reducing batchsize by half.");
                batchsize /= 2;
                continue;
            }
        }

        unsigned long long* hlenGPU;
        unsigned long long* rlenGPU;
        if (hipMalloc(&hlenGPU, (batchsize + 1) * sizeof(unsigned long long)) != hipSuccess) {
            printf("Batchsize too big for GPU memory! Reducing batchsize by half.");
            batchsize /= 2;
            if(precision == 32) hipFree(finalsum32);
            else hipFree(finalsum);

            continue;
        }

        if (hipMalloc(&rlenGPU, (batchsize + 1) * sizeof(unsigned long long)) != hipSuccess) {
            printf("Batchsize too big for GPU memory! Reducing batchsize by half.");
            batchsize /= 2;
            if(precision == 32) hipFree(finalsum32);
            else hipFree(finalsum);

            hipFree(hlenGPU);
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
                qCPU[hcount + c] = 30 + c%10;
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

        if (hipMalloc(&hapGPU, hcount * sizeof(unsigned char)) != hipSuccess) {
            printf("Batchsize too big for GPU memory! Reducing batchsize by half.");
            batchsize /= 2;
            if(precision == 32){
                hipFree(finalsum32);
                free(finalsum_32);
            }
            else{
                hipFree(finalsum);
                free(finalsum_);
            }

            hipFree(hlenGPU);
            hipFree(rlenGPU);
            free(hlenCPU);
            free(rlenCPU);
            free(hapCPU);
            free(readCPU);
            free(qCPU);
            free(iCPU);
            free(dCPU);
            continue;
        }

        if (hipMalloc(&readGPU, rcount * sizeof(unsigned char)) != hipSuccess) {
            printf("Batchsize too big for GPU memory! Reducing batchsize by half.");
            batchsize /= 2;
            if(precision == 32){
                hipFree(finalsum32);
                free(finalsum_32);
            }
            else{
                hipFree(finalsum);
                free(finalsum_);
            }

            hipFree(hlenGPU);
            hipFree(rlenGPU);
            hipFree(hapGPU);
            free(hlenCPU);
            free(rlenCPU);
            free(hapCPU);
            free(readCPU);
            free(qCPU);
            free(iCPU);
            free(dCPU);
            continue;
        }

        if (hipMalloc(&qGPU, rcount * sizeof(unsigned char)) != hipSuccess) {
            printf("Batchsize too big for GPU memory! Reducing batchsize by half.");
            batchsize /= 2;
            if(precision == 32){
                hipFree(finalsum32);
                free(finalsum_32);
            }
            else{
                hipFree(finalsum);
                free(finalsum_);
            }

            hipFree(hlenGPU);
            hipFree(rlenGPU);
            hipFree(hapGPU);
            hipFree(readGPU);
            free(hlenCPU);
            free(rlenCPU);
            free(hapCPU);
            free(readCPU);
            free(qCPU);
            free(iCPU);
            free(dCPU);
            continue;
        }

        if (hipMalloc(&iGPU, rcount * sizeof(unsigned char)) != hipSuccess) {
            printf("Batchsize too big for GPU memory! Reducing batchsize by half.");
            batchsize /= 2;
            if(precision == 32){
                hipFree(finalsum32);
                free(finalsum_32);
            }
            else{
                hipFree(finalsum);
                free(finalsum_);
            }

            hipFree(hlenGPU);
            hipFree(rlenGPU);
            hipFree(hapGPU);
            hipFree(readGPU);
            hipFree(qGPU);
            free(hlenCPU);
            free(rlenCPU);
            free(hapCPU);
            free(readCPU);
            free(qCPU);
            free(iCPU);
            free(dCPU);
            continue;
        }

        if (hipMalloc(&dGPU, rcount * sizeof(unsigned char)) != hipSuccess) {
            printf("Batchsize too big for GPU memory! Reducing batchsize by half.");
            batchsize /= 2;
            if(precision == 32){
                hipFree(finalsum32);
                free(finalsum_32);
            }
            else{
                hipFree(finalsum);
                free(finalsum_);
            }

            hipFree(hlenGPU);
            hipFree(rlenGPU);
            hipFree(hapGPU);
            hipFree(qGPU);
            hipFree(iGPU);
            free(hlenCPU);
            free(rlenCPU);
            free(hapCPU);
            free(readCPU);
            free(qCPU);
            free(iCPU);
            free(dCPU);
            continue;
        }

        hipMemcpy(hapGPU, hapCPU, sizeof(unsigned char)*hcount, hipMemcpyHostToDevice);
        hipMemcpy(readGPU, readCPU, sizeof(unsigned char)*rcount, hipMemcpyHostToDevice);
        hipMemcpy(qGPU, qCPU, sizeof(unsigned char)*rcount, hipMemcpyHostToDevice);
        hipMemcpy(iGPU, iCPU, sizeof(unsigned char)*rcount, hipMemcpyHostToDevice);
        hipMemcpy(dGPU, dCPU, sizeof(unsigned char)*rcount, hipMemcpyHostToDevice);
        hipMemcpy(rlenGPU, rlenCPU, sizeof(unsigned long long)*(batchsize + 1), hipMemcpyHostToDevice);
        hipMemcpy(hlenGPU, hlenCPU, sizeof(unsigned long long)*(batchsize + 1), hipMemcpyHostToDevice);

        for(int a = 0; a < 6; a++){
            printf("%d\n",a);
                
            if(precision == 32) total_time = pairhmm_benchmark_fp32(rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, finalsum32, finalsum_32, gcups, maxsize, batchsize);
            else total_time = pairhmm_benchmark_fp64(rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, finalsum, finalsum_, gcups, maxsize, batchsize);

            if(a > 0) throughput += gcups/(double)(total_time*pow(10,9));
        }

        gcups = 0;
        maxsize = 0;
        rcount = 0;
        hcount = 0;

        printf("Throughput: %.6f TCUPS\n", throughput/5);

        hipFree(hapGPU);
        hipFree(readGPU);
        hipFree(qGPU);
        hipFree(iGPU);
        hipFree(dGPU); 
        hipFree(hlenGPU);
        hipFree(rlenGPU);
        if(precision == 32){
            hipFree(finalsum32);
            free(finalsum_32);
        }
        else{
            hipFree(finalsum);
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
        if(lenidx%2 == 0) batchsize /= 2;
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
