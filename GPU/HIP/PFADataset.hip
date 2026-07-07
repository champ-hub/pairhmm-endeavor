#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <random>
#include <time.h>
#include <hip/hip_runtime.h>

using namespace std;

//PAIRHMM FP64 FUNCTIONS
template <unsigned int N, unsigned int P> 
__global__ void pairHMM_register_fp64_subwarp_dataset(unsigned long long* rslen, unsigned long long* haplen, unsigned char* hap, unsigned char* rs, unsigned char* q, unsigned char* i, unsigned char* d, double* finalsum, unsigned int* blockDist, unsigned int seq, unsigned int blockCounts)
{
    __shared__ unsigned char _rs[64*N];
    __shared__ unsigned char _q[64*N];
    __shared__ unsigned char _i[64*N];
    __shared__ unsigned char _d[64*N];  
    
    const unsigned short subwarp = 64/P;
    const unsigned int laneID = threadIdx.x%subwarp, warpID = ((threadIdx.x/subwarp)%P);
    const unsigned int bidx = (P*blockIdx.x + warpID)%blockCounts;
    const unsigned int seqidx = blockDist[seq + bidx], idx = N*subwarp*warpID;

    double Mprev[N] = {0};
    double Iprev[N] = {0};
    unsigned char _hap[N] = {0};

    double M = 0.0; 
    double I = 0.0; 
    double y1 = 0.0; 
    double y2 = 0.0; 
    double y3 = 0.0; 
    double value = 0.0;
    double d0 = 0.0;
    double tmm = 0.0;
    unsigned char rs0;
    unsigned char q0;
    unsigned char in0;
    unsigned char d1;

    const unsigned long long rsize = -rslen[seqidx] + rslen[seqidx + 1];
    const unsigned long long hsize = -haplen[seqidx] + haplen[seqidx + 1];
    const unsigned long long ridx = rslen[seqidx];
    const unsigned long long hidx = haplen[seqidx];

    rs0 = rs[ridx];
    q0 = q[ridx];
    
    for(unsigned int h = 0; h < N; h++){
        _hap[h] = hap[hidx + N*laneID + h];
        _rs[idx + N*laneID + h] = rs[ridx + N*laneID + h];
        _q[idx + N*laneID + h]  = q[ridx + N*laneID + h];
        _i[idx + N*laneID + h]  = i[ridx + N*laneID + h];
        _d[idx + N*laneID + h]  = d[ridx + N*laneID + h];
        Mprev[h] = (_hap[h] == rs0) ?  (1 - qualfp64[q0 ]) : qualfp64[q0 ]*0.3333333333333333333;
        Mprev[h] *= 0.9*ldexp(1.0, 1020.0)/(double)(hsize); 
        value = value*0.1 + Mprev[h];  
    }

    y3 = __shfl_up(value, 1);
    if(laneID == 0) y3 = 0;

    y1 = __shfl_up(y3, 1);
    if (laneID >= 1) y3 = fma(y1,powDfp64[N],y3);
    y1 = __shfl_up(y3, 2);
    if (laneID >= 2) y3 = fma(y1,powDfp64[2*N],y3);
    y1 = __shfl_up(y3, 4);
    if (laneID >= 4) y3 = fma(y1,powDfp64[4*N],y3);
    y1 = __shfl_up(y3, 8);
    if (laneID >= 8) y3 = fma(y1,powDfp64[8*N],y3);
    y1 = __shfl_up(y3, 16);
    if (laneID >= 16) y3 = fma(y1,powDfp64[16*N],y3);
    y1 = __shfl_up(y3, 32);
    if (laneID >= 32) y3 = fma(y1,powDfp64[32*N],y3);

    for(int k = 1; k < rsize; k++){
        rs0 = _rs[idx + k];
        q0 = _q[idx + k];
        in0 = _i[idx + k];
        d0 = 0.9*qualfp64[_d[idx + k-1] ];
        d1 = _d[idx + k];
        tmm = 1 - qualfp64[in0 ] - qualfp64[d1 ];
        value = __shfl_up(Mprev[N-1], 1);
        y1 = __shfl_up(Iprev[N-1], 1);
        
        y2 = (_hap[0] == rs0) ?  (1 - qualfp64[q0]) : qualfp64[q0]*0.3333333333333333333;
        y2 *= (value*tmm + 0.9*y1 + d0*10*(y3 - value));
    
        if(laneID == 0) y2 = 0.0;
        
        M = y2;
        value = M;
        
        for(int h = 1; h < N; h++){
            y2 = (_hap[h] == rs0) ?  (1 - qualfp64[q0]) : qualfp64[q0]*0.3333333333333333333;
            y2 *= (Mprev[h-1]*tmm + 0.9*Iprev[h-1] + d0*y3);

            y3 = y3*0.1 + Mprev[h-1];

            Iprev[h-1] = Mprev[h-1]*qualfp64[in0 ] + Iprev[h-1]*0.1;
            Mprev[h-1] = M;

            M = y2;
            value = value*0.1 + M;        
        }       

        Iprev[N-1] = Mprev[N-1]*qualfp64[in0 ] + Iprev[N-1]*0.1;
        Mprev[N-1] = M;

        y3 = __shfl_up(value, 1);

        if(laneID == 0) y3 = 0;

        y1 = __shfl_up(y3, 1);
        if (laneID >= 1) y3 = fma(y1,powDfp64[N],y3);
        y1 = __shfl_up(y3, 2);
        if (laneID >= 2) y3 = fma(y1,powDfp64[2*N],y3);
        y1 = __shfl_up(y3, 4);
        if (laneID >= 4) y3 = fma(y1,powDfp64[4*N],y3);
        y1 = __shfl_up(y3, 8);
        if (laneID >= 8) y3 = fma(y1,powDfp64[8*N],y3);
        y1 = __shfl_up(y3, 16);
        if (laneID >= 16) y3 = fma(y1,powDfp64[16*N],y3);
        y1 = __shfl_up(y3, 32);
        if (laneID >= 32) y3 = fma(y1,powDfp64[32*N],y3);
    }

    y1 = 0;
    for(int h = 0; h < N; h++) 
        if(N*laneID + h < hsize) y1 += Mprev[h] + Iprev[h]; 

    y2 = __shfl_up(y1, 1);
    if (laneID >= 1) y1 += y2;
    y2 = __shfl_up(y1, 2);
    if (laneID >= 2) y1 += y2;
    y2 = __shfl_up(y1, 4);
    if (laneID >= 4) y1 += y2;
    y2 = __shfl_up(y1, 8);
    if (laneID >= 8) y1 += y2;
    y2 = __shfl_up(y1, 16);
    if (laneID >= 16) y1 += y2;
    y2 = __shfl_up(y1, 32);
    if (laneID >= 32) y1 += y2;

    if(laneID == subwarp - 1){
        if(blockIdx.x != blockDim.x - 1) finalsum[seqidx] = y1;
        else if(blockIdx.x == blockDim.x - 1 && bidx > P) finalsum[seqidx] = y1;
    }
}

template <unsigned int N, unsigned int P> 
__global__ void pairHMM_shared_fp64_dataset(unsigned long long* rslen, unsigned long long* haplen, unsigned char* hap, unsigned char* rs, unsigned char* q, unsigned char* i, unsigned char* d, double* finalsum, unsigned int* blockDist, unsigned int seq)
{
    __shared__ double _aux[P+1];
    __shared__ double _M[P+1];
    __shared__ double _I[P+1];

    _M[0] = _I[0] = _aux[0] = 0;
    
    const unsigned int laneID = threadIdx.x%64, tid = threadIdx.x;
    unsigned int seqidx = blockDist[seq + blockIdx.x];
    double Mprev[N] = {0};
    double Iprev[N] = {0};
    unsigned char _hap[N] = {0};

    double M = 0; 
    double I = 0; 
    double y1 = 0; 
    double y2 = 0; 
    double y3 = 0; 
    double value = 0;
    double d0 = 0;
    double tmm = 0;
    unsigned char rs0;
    unsigned char q0;
    unsigned char in;
    unsigned char d1;
    
    const unsigned long long rsize = -rslen[seqidx] + rslen[seqidx + 1];
    const unsigned long long hsize = -haplen[seqidx] + haplen[seqidx + 1];
    const unsigned long long ridx = rslen[seqidx];
    const unsigned long long hidx = haplen[seqidx];

    rs0 = rs[ridx];
    q0 = q[ridx];
    d1 = d[ridx];
    
    for(unsigned int h = 0; h < N; h++){
        _hap[h] = hap[hidx + N*tid + h];
        Mprev[h] = (_hap[h] == rs0) ?  (1 - qualfp64[q0]) : qualfp64[q0]*0.3333333333333333333333;
        Mprev[h] *= 0.9*ldexp(1.0, 1020.0)/(double)(hsize); 
        value = value*0.1 + Mprev[h];  
    }

    y3 = __shfl_up(value, 1);
    if(laneID == 0) y3 = 0;
    else if(laneID == 63){
        _M[1 + (tid/64)] = Mprev[N-1];
        _I[1 + (tid/64)] = Iprev[N-1];
    }

    

    y1 = __shfl_up(y3, 1);
    if (laneID >= 1) y3 = fma(y1,powDfp64[N],y3);
    y1 = __shfl_up(y3, 2);
    if (laneID >= 2) y3 = fma(y1,powDfp64[2*N],y3);
    y1 = __shfl_up(y3, 4);
    if (laneID >= 4) y3 = fma(y1,powDfp64[4*N],y3);
    y1 = __shfl_up(y3, 8);
    if (laneID >= 8) y3 = fma(y1,powDfp64[8*N],y3);
    y1 = __shfl_up(y3, 16);
    if (laneID >= 16) y3 = fma(y1,powDfp64[16*N],y3);

    if(laneID == 63) _aux[1 + (tid/64)] = y3*powDfp64[N] + value;
    __syncthreads();

    y3 += _aux[tid/64]*powDfp64[N*laneID];

    for(int k = 1; k < rsize; k++){
        rs0 = rs[ridx + k];
        q0 = q[ridx + k];
        in = i[ridx + k];
        d0 = 0.9*qualfp64[d1];
        d1 = d[ridx + k];
        tmm = 1 - qualfp64[in] - qualfp64[d1];
        value = __shfl_up(Mprev[N-1], 1);
        y1 = __shfl_up(Iprev[N-1], 1);
        
        y2 = (_hap[0] == rs0) ?  (1 - qualfp64[q0]) : qualfp64[q0]*0.3333333333333333333333;
        if(laneID == 0) y2 *= (_M[tid/64]*tmm + 0.9*(_I[tid/64] + d0*10*(y3 - _M[tid/64])));
        else y2 *= (value*tmm + 0.9*y1 + d0*10*(y3 - value));

        M = y2;
        I = Mprev[0]*qualfp64[in] + Iprev[0]*0.1;

        value = M;
        
        for(int h = 1; h < N; h++){
            y2 = (_hap[h] == rs0) ?  (1 - qualfp64[q0]) : qualfp64[q0]*0.3333333333333333333333;
            y2 *= (Mprev[h-1]*tmm + 0.9*Iprev[h-1] + d0*y3);

            y3 = fma(y3,0.1,Mprev[h-1]);

            Mprev[h-1] = M;
            Iprev[h-1] = I;

            M = y2;
            I = Mprev[h]*qualfp64[in] + Iprev[h]*0.1;  
            value = fma(value,0.1,M);        
        }       

        Mprev[N-1] = M;
        Iprev[N-1] = I; 

        y3 = __shfl_up(value, 1);

        __syncthreads();
        if(laneID == 0) y3 = 0;
        else if(laneID == 63){
            _M[1 + (tid/64)] = M;
            _I[1 + (tid/64)] = I;
        }

        y1 = __shfl_up(y3, 1);
        if (laneID >= 1) y3 = fma(y1,powDfp64[N],y3);
        y1 = __shfl_up(y3, 2);
        if (laneID >= 2) y3 = fma(y1,powDfp64[2*N],y3);
        y1 = __shfl_up(y3, 4);
        if (laneID >= 4) y3 = fma(y1,powDfp64[4*N],y3);
        y1 = __shfl_up(y3, 8);
        if (laneID >= 8) y3 = fma(y1,powDfp64[8*N],y3);
        y1 = __shfl_up(y3, 16);
        if (laneID >= 16) y3 = fma(y1,powDfp64[16*N],y3);

        if(laneID == 63) _aux[1 + (tid/64)] = y3*powDfp64[N] + value;
        __syncthreads();

        y3 += _aux[tid/64]*powDfp64[N*laneID];
    }
    

    y1 = 0;
    for(int h = 0; h < N; h++)
        if(N*tid + h < hsize) y1 += Mprev[h] + Iprev[h]; 
    
    y2 = __shfl_up(y1, 1);
    if (laneID >= 1) y1 += y2;
    y2 = __shfl_up(y1, 2);
    if (laneID >= 2) y1 += y2;
    y2 = __shfl_up(y1, 4);
    if (laneID >= 4) y1 += y2;
    y2 = __shfl_up(y1, 8);
    if (laneID >= 8) y1 += y2;
    y2 = __shfl_up(y1, 16);
    if (laneID >= 16) y1 += y2;
    y2 = __shfl_up(y1, 32);
    if (laneID >= 32) y1 += y2;

    if(laneID == 63) _aux[1 + (tid/64)] = y1;
    __syncthreads();

    if(tid < 64){
        if(tid < P) y1 = _aux[1 + tid];
        else y1 = 0;

        y2 = __shfl_up(y1, 1);
        if (laneID >= 1) y1 += y2;
        y2 = __shfl_up(y1, 2);
        if (laneID >= 2) y1 += y2;
        y2 = __shfl_up(y1, 4);
        if (laneID >= 4) y1 += y2;
        y2 = __shfl_up(y1, 8);
        if (laneID >= 8) y1 += y2;
        y2 = __shfl_up(y1, 16);
        if (laneID >= 16) y1 += y2;
        y2 = __shfl_up(y1, 32);
        if (laneID >= 32) y1 += y2;

        if(laneID == 63) finalsum[seqidx] = y1;
    }    
}

float runKernelsFP64(unsigned long long* rslen, unsigned long long* haplen, unsigned char* hap, unsigned char* rs, unsigned char* q, unsigned char* in, unsigned char* d, double* sumGPU, double* sumCPU, unsigned int* blockDist, unsigned int* blockCounts, FILE* fpout, int maxsize, unsigned int seq_size){
    float total_time = 0;
    double g_ctxd = log10(ldexp(1.0, 1020.0));
    double powvalues[128] = {0};
    
    for(int i = 0; i < 128; i++){
        powvalues[i] = pow(10.0, -(double)i/10.0);
    }

    hipMemcpyToSymbol(qualfp64, powvalues, sizeof(powvalues));

    double powd[2049] = {0};
    for(int i = 0; i < 2049; i++){
        powd[i] = pow(10.0, -(double)(i));
    }

    hipMemcpyToSymbol(powDfp64, powd, sizeof(powd));

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    if(blockCounts[0] != 0) pairHMM_register_fp64_subwarp_dataset<16,32><<<1 + blockCounts[0]/32,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 0, blockCounts[0]); //32
    if(blockCounts[1] != 0) pairHMM_register_fp64_subwarp_dataset<16,16><<<1 + blockCounts[1]/16,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, seq_size, blockCounts[1]); //64
    if(blockCounts[2] != 0) pairHMM_register_fp64_subwarp_dataset<16,10><<<1 + blockCounts[2]/10,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 2*seq_size, blockCounts[2]); //96
    if(blockCounts[3] != 0) pairHMM_register_fp64_subwarp_dataset<16,8><<<1 + blockCounts[3]/8,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 3*seq_size, blockCounts[3]); //128
    if(blockCounts[4] != 0) pairHMM_register_fp64_subwarp_dataset<16,6><<<1 + blockCounts[4]/6,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 4*seq_size, blockCounts[4]); //160
    if(blockCounts[5] != 0) pairHMM_register_fp64_subwarp_dataset<16,5><<<1 + blockCounts[5]/5,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 5*seq_size, blockCounts[5]); //192
    if(blockCounts[6] != 0) pairHMM_register_fp64_subwarp_dataset<16,4><<<1 + blockCounts[6]/4,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 6*seq_size, blockCounts[6]); //224
    if(blockCounts[7] != 0) pairHMM_register_fp64_subwarp_dataset<16,4><<<1 + blockCounts[7]/4,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 7*seq_size, blockCounts[7]); //256
    if(blockCounts[8] != 0) pairHMM_register_fp64_subwarp_dataset<16,3><<<1 + blockCounts[8]/3,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 8*seq_size, blockCounts[8]); //288
    if(blockCounts[9] != 0) pairHMM_register_fp64_subwarp_dataset<16,3><<<1 + blockCounts[9]/3,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 9*seq_size, blockCounts[9]); //320
    if(blockCounts[10] != 0) pairHMM_register_fp64_subwarp_dataset<16,2><<<1 + blockCounts[10]/2,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 10*seq_size, blockCounts[10]); //352
    if(blockCounts[11] != 0) pairHMM_register_fp64_subwarp_dataset<16,2><<<1 + blockCounts[11]/2,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 11*seq_size, blockCounts[11]); //384
    if(blockCounts[12] != 0) pairHMM_register_fp64_subwarp_dataset<16,2><<<1 + blockCounts[12]/2,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 12*seq_size, blockCounts[12]); //416
    if(blockCounts[13] != 0) pairHMM_register_fp64_subwarp_dataset<16,2><<<1 + blockCounts[13]/2,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 13*seq_size, blockCounts[13]); //448
    if(blockCounts[14] != 0) pairHMM_register_fp64_subwarp_dataset<16,2><<<1 + blockCounts[14]/2,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 14*seq_size, blockCounts[14]); //480
    if(blockCounts[15] != 0) pairHMM_register_fp64_subwarp_dataset<16,2><<<1 + blockCounts[15]/2,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 15*seq_size, blockCounts[15]); //512

    for(int i = 16; i < 32; i++){
        if(blockCounts[i] != 0) pairHMM_register_fp64_subwarp_dataset<16,1><<<blockCounts[i],64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, i*seq_size, blockCounts[i]);
    }

    if(blockCounts[32] != 0) pairHMM_shared_fp64_dataset<16,2><<<blockCounts[32],128>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 32*seq_size);
    if(blockCounts[33] != 0) pairHMM_shared_fp64_dataset<16,4><<<blockCounts[33],256>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 33*seq_size);
    if(blockCounts[34] != 0) pairHMM_shared_fp64_dataset<16,8><<<blockCounts[34],512>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 34*seq_size);
    if(blockCounts[35] != 0) pairHMM_shared_fp64_dataset<16,16><<<blockCounts[35],1024>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 35*seq_size);
    hipEventRecord(stop);
    hipMemcpy(sumCPU, sumGPU, seq_size*sizeof(double), hipMemcpyDeviceToHost);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&total_time, start, stop);

    //for(int j = 0; j < seq_size; j++) fprintf(fpout,"%.64f\n", log10(sumCPU[j]) - g_ctxd);

    return total_time;
}

//PAIRHMM FP32 FUNCTIONS
template <unsigned int N, unsigned int P> 
__global__ void pairHMM_register_fp32_subwarp_dataset(unsigned long long* rslen, unsigned long long* haplen, unsigned char* hap, unsigned char* rs, unsigned char* q, unsigned char* i, unsigned char* d, float* finalsum, unsigned int* blockDist, unsigned int seq, unsigned int blockCounts)
{
    __shared__ unsigned char _rs[64*N];
    __shared__ unsigned char _q[64*N];
    __shared__ unsigned char _i[64*N];
    __shared__ unsigned char _d[64*N];   
    
    const unsigned int subwarp = 64/P;
    const unsigned int laneID = threadIdx.x%subwarp, warpID = ((threadIdx.x/subwarp)%P);
    const unsigned int bidx = (P*blockIdx.x + warpID)%blockCounts;
    const unsigned int seqidx = blockDist[seq + bidx], idx = N*subwarp*warpID;

    float Mprev[N] = {0};
    float Iprev[N] = {0};
    unsigned char _hap[N] = {0};

    float M = 0.0f; 
    float I = 0.0f; 
    float y1 = 0.0f; 
    float y2 = 0.0f; 
    float y3 = 0.0f; 
    float value = 0.0f;
    float d0 = 0.0f;
    float tmm = 0.0f;
    unsigned char rs0;
    unsigned char q0;
    unsigned char in;
    unsigned char d1;

    const unsigned long long rsize = -rslen[seqidx] + rslen[seqidx + 1];
    const unsigned long long hsize = -haplen[seqidx] + haplen[seqidx + 1];
    const unsigned long long ridx = rslen[seqidx];
    const unsigned long long hidx = haplen[seqidx];

    rs0 = rs[ridx];
    q0 = q[ridx];
    
    for(unsigned int h = 0; h < N; h++){
        _hap[h] = hap[hidx + N*laneID + h];
        
        _rs[idx + N*laneID + h] = rs[ridx + N*laneID + h];
        _q[idx + N*laneID + h]  = q[ridx + N*laneID + h];
        _i[idx + N*laneID + h]  = i[ridx + N*laneID + h];
        _d[idx + N*laneID + h]  = d[ridx + N*laneID + h];
        
        Mprev[h] = (_hap[h] == rs0) ?  (1.0f - qualfp32[q0]) : qualfp32[q0]*0.3333333333333333333f;
        Mprev[h] *= 0.9f*ldexpf(1.f, 120.f)/(float)(hsize); 
        value = value*0.1f + Mprev[h];  
    }

    y3 = __shfl_up( value, 1);
    if(laneID == 0) y3 = 0.0f;

    y1 = __shfl_up( y3, 1);
    if (laneID >= 1) y3 += y1*powDfp32[N];
    y1 = __shfl_up(y3, 2);
    if (laneID >= 2) y3 += y1*powDfp32[2*N];

    for(int k = 1; k < rsize; k++){
        rs0 = _rs[idx + k];
        q0 = _q[idx + k];
        in = _i[idx + k];
        d0 = qualfp32[_d[idx + k-1]];
        d1 = _d[idx + k];
        tmm = 1.0f - qualfp32[in] - qualfp32[d1];
        value = __shfl_up( Mprev[N-1], 1);
        y1 = __shfl_up( Iprev[N-1], 1);
        
        y2 = (_hap[0] == rs0) ?  (1.0f - qualfp32[q0]) : qualfp32[q0]*0.3333333333333333333f;
        y2 *= (value*tmm + 0.9f*(y1 + d0*10.0f*(y3 - value)));
    
        if(laneID == 0) y2 = 0.0f;
        
        M = y2;
        I = Mprev[0]*qualfp32[in] + Iprev[0]*0.1f;

        value = M;
        
        for(int h = 1; h < N; h++){
            y2 = (_hap[h] == rs0) ? (1.0f - qualfp32[q0]) : qualfp32[q0]*0.3333333333333333333f;
            y2 *= (Mprev[h-1]*tmm + 0.9f*(Iprev[h-1] + d0*y3));

            y3 = y3*0.1f + Mprev[h-1];

            Mprev[h-1] = M;
            Iprev[h-1] = I;

            M = y2;
            I = Mprev[h]*qualfp32[in] + Iprev[h]*0.1f;  
            value = value*0.1f + M;        
        }       

        Mprev[N-1] = M;
        Iprev[N-1] = I; 

        y3 = __shfl_up( value, 1);

        if(laneID == 0) y3 = 0.0f;
        
        y1 = __shfl_up( y3, 1);
        if (laneID >= 1) y3 += y1*powDfp32[N];
        y1 = __shfl_up(y3, 2);
        if (laneID >= 2) y3 += y1*powDfp32[2*N];
    }

    y1 = 0.0f;
    for(int h = 0; h < N; h++)
        if(N*laneID + h < hsize) y1 += Mprev[h] + Iprev[h]; 

    y2 = __shfl_up( y1, 1);
    if (laneID >= 1) y1 += y2;
    y2 = __shfl_up( y1, 2);
    if (laneID >= 2) y1 += y2;
    y2 = __shfl_up( y1, 4);
    if (laneID >= 4) y1 += y2;
    y2 = __shfl_up( y1, 8);
    if (laneID >= 8) y1 += y2;
    y2 = __shfl_up( y1, 16);
    if (laneID >= 16) y1 += y2;
    y2 = __shfl_up( y1, 32);
    if (laneID >= 32) y1 += y2;

    if(laneID == subwarp - 1){
        if(blockIdx.x != blockDim.x - 1) finalsum[seqidx] = y1;
        else if(blockIdx.x == blockDim.x - 1 && bidx > P) finalsum[seqidx] = y1;
    }
}

template <unsigned int N, unsigned int P> 
__global__ void pairHMM_shared_fp32_dataset(unsigned long long* rslen, unsigned long long* haplen, unsigned char* hap, unsigned char* rs, unsigned char* q, unsigned char* i, unsigned char* d, float* finalsum, unsigned int* blockDist, unsigned int seq)
{
    __shared__ float _aux[P+1];
    __shared__ float _M[P+1];
    __shared__ float _I[P+1];

    _M[0] = _I[0] = _aux[0] = 0.0f;
    
    const unsigned int laneID = threadIdx.x%64, tid = threadIdx.x, seqidx = blockDist[seq + blockIdx.x];;
    float Mprev[N] = {0};
    float Iprev[N] = {0};
    unsigned char _hap[N] = {0};

    float M = 0.0f; 
    float I = 0.0f; 
    float y1 = 0.0f; 
    float y2 = 0.0f; 
    float y3 = 0.0f; 
    float value = 0.0f;
    float d0 = 0.0f;
    float tmm = 0.0f;
    unsigned char rs0;
    unsigned char q0;
    unsigned char in;
    unsigned char d1;

    const unsigned long long rsize = -rslen[seqidx] + rslen[seqidx + 1];
    const unsigned long long hsize = -haplen[seqidx] + haplen[seqidx + 1];
    const unsigned long long ridx = rslen[seqidx];
    const unsigned long long hidx = haplen[seqidx];

    rs0 = rs[ridx];
    q0 = q[ridx];
    d1 = d[ridx];
    
    for(unsigned int h = 0; h < N; h++){
        _hap[h] = hap[hidx + N*tid + h];
        Mprev[h] = (_hap[h] == rs0) ?  (1.0f - qualfp32[q0]) : qualfp32[q0]*0.3333333333333333333f;
        Mprev[h] *= 0.9f*ldexpf(1.f, 120.f)/(float)(hsize); 
        value = value*0.1f + Mprev[h];  
    }

    y3 = __shfl_up( value, 1);
    if(laneID == 0) y3 = 0.0f;
    else if(laneID == 63){
        _M[1 + (tid/64)] = Mprev[N-1];
        _I[1 + (tid/64)] = Iprev[N-1];
    }

    y1 = __shfl_up( y3, 1);
    if (laneID >= 1) y3 += y1*powDfp32[N];

    if(laneID == 63) _aux[1 + (tid/64)] = y3*powDfp32[N] + value;
    __syncthreads();

    y3 += _aux[tid/64]*powDfp32[laneID];

    for(int k = 1; k < rsize; k++){
        rs0 = rs[ridx + k];
        q0 = q[ridx + k];
        in = i[ridx + k];
        d0 = qualfp32[d1];
        d1 = d[ridx + k];
        tmm = 1.0f - qualfp32[in] - qualfp32[d1];
        value = __shfl_up( Mprev[N-1], 1);
        y1 = __shfl_up( Iprev[N-1], 1);
        
        y2 = (_hap[0] == rs0) ?  (1.0f - qualfp32[q0]) : qualfp32[q0]*0.3333333333333333333f;
        if(laneID == 0) y2 *= (_M[tid/64]*tmm + 0.9f*(_I[tid/64] + d0*10.0f*(y3 - _M[tid/64])));
        else y2 *= (value*tmm + 0.9f*(y1 + d0*10.0f*(y3 - value)));

        M = y2;
        I = Mprev[0]*qualfp32[in] + Iprev[0]*0.1f;

        value = M;
        
        for(int h = 1; h < N; h++){
            y2 = (_hap[h] == rs0) ?  (1.0f - qualfp32[q0]) : qualfp32[q0]*0.3333333333333333333f;
            y2 *= (Mprev[h-1]*tmm + 0.9f*(Iprev[h-1] + d0*y3));

            y3 = y3*0.1f + Mprev[h-1];

            Mprev[h-1] = M;
            Iprev[h-1] = I;

            M = y2;
            I = Mprev[h]*qualfp32[in] + Iprev[h]*0.1f;  
            value = value*0.1f + M;        
        }       

        Mprev[N-1] = M;
        Iprev[N-1] = I; 

        y3 = __shfl_up( value, 1);

        __syncthreads();
        if(laneID == 0) y3 = 0.0f;
        else if(laneID == 63){
            _M[1 + (tid/64)] = M;
            _I[1 + (tid/64)] = I;
        }

        y1 = __shfl_up( y3, 1);
        if (laneID >= 1) y3 += y1*powDfp32[N];

        if(laneID == 63) _aux[1 + (tid/64)] = y3*powDfp32[N] + value;
        __syncthreads();

        y3 += _aux[tid/64]*powDfp32[laneID];
    }
    

    y1 = 0.0f;
    for(int h = 0; h < N; h++)
        if(N*tid + h < hsize) y1 += Mprev[h] + Iprev[h]; 
    
    y2 = __shfl_up( y1, 1);
    if (laneID >= 1) y1 += y2;
    y2 = __shfl_up( y1, 2);
    if (laneID >= 2) y1 += y2;
    y2 = __shfl_up( y1, 4);
    if (laneID >= 4) y1 += y2;
    y2 = __shfl_up( y1, 8);
    if (laneID >= 8) y1 += y2;
    y2 = __shfl_up( y1, 16);
    if (laneID >= 16) y1 += y2;
    y2 = __shfl_up( y1, 32);
    if (laneID >= 32) y1 += y2;

    if(laneID == 63) _aux[1 + (tid/64)] = y1;
    __syncthreads();

    if(tid < 64){
        if(tid < P) y1 = _aux[1 + tid];
        else y1 = 0.0f;

        y2 = __shfl_up( y1, 1);
        if (laneID >= 1) y1 += y2;
        y2 = __shfl_up( y1, 2);
        if (laneID >= 2) y1 += y2;
        y2 = __shfl_up( y1, 4);
        if (laneID >= 4) y1 += y2;
        y2 = __shfl_up( y1, 8);
        if (laneID >= 8) y1 += y2;
        y2 = __shfl_up( y1, 16);
        if (laneID >= 16) y1 += y2;
        y2 = __shfl_up( y1, 32);
        if (laneID >= 32) y1 += y2;

        if(laneID == 63) finalsum[seqidx] = y1;
    }    
}

float runKernelsFP32(unsigned long long* rslen, unsigned long long* haplen, unsigned char* hap, unsigned char* rs, unsigned char* q, unsigned char* in, unsigned char* d, float* sumGPU, float* sumCPU, unsigned int* blockDist, unsigned int* blockCounts, FILE* fpout, int maxsize, unsigned int seq_size){
    float total_time = 0;
    float g_ctxf = log10f(ldexpf(1.f, 120.f));
    float powvalues[128] = {0};

    for(int i = 0; i < 128; i++){
        powvalues[i] = powf(10.0f, -(float)i/10.0f);
    }

    hipMemcpyToSymbol(qualfp32, powvalues, sizeof(powvalues));

    float powd[2049] = {0};

    for(int i = 0; i < 2049; i++){
        powd[i] = powf(10.0f, -(float)(i));
    }

    hipMemcpyToSymbol(powDfp32, powd, sizeof(powd));

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
        if(blockCounts[0] != 0) pairHMM_register_fp32_subwarp_dataset<16,32><<<1 + blockCounts[0]/32,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 0, blockCounts[0]); //32
    if(blockCounts[1] != 0) pairHMM_register_fp32_subwarp_dataset<16,16><<<1 + blockCounts[1]/16,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, seq_size, blockCounts[1]); //64
    if(blockCounts[2] != 0) pairHMM_register_fp32_subwarp_dataset<12,8><<<1 + blockCounts[2]/8,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 2*seq_size, blockCounts[2]); //96
    if(blockCounts[3] != 0) pairHMM_register_fp32_subwarp_dataset<16,8><<<1 + blockCounts[3]/8,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 3*seq_size, blockCounts[3]); //128
    if(blockCounts[4] != 0) pairHMM_register_fp32_subwarp_dataset<10,4><<<1 + blockCounts[4]/4,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 4*seq_size, blockCounts[4]); //160
    if(blockCounts[5] != 0) pairHMM_register_fp32_subwarp_dataset<12,4><<<1 + blockCounts[5]/4,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 5*seq_size, blockCounts[5]); //192
    if(blockCounts[6] != 0) pairHMM_register_fp32_subwarp_dataset<14,4><<<1 + blockCounts[6]/4,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 6*seq_size, blockCounts[6]); //224
    if(blockCounts[7] != 0) pairHMM_register_fp32_subwarp_dataset<16,4><<<1 + blockCounts[7]/4,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 7*seq_size, blockCounts[7]); //256
    if(blockCounts[8] != 0) pairHMM_register_fp32_subwarp_dataset<18,2><<<1 + blockCounts[8]/2,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 8*seq_size, blockCounts[8]); //288
    if(blockCounts[9] != 0) pairHMM_register_fp32_subwarp_dataset<10,2><<<1 + blockCounts[9]/2,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 9*seq_size, blockCounts[9]); //320
    if(blockCounts[10] != 0) pairHMM_register_fp32_subwarp_dataset<11,2><<<1 + blockCounts[10]/2,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 10*seq_size, blockCounts[10]); //352
    if(blockCounts[11] != 0) pairHMM_register_fp32_subwarp_dataset<12,2><<<1 + blockCounts[11]/2,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 11*seq_size, blockCounts[11]); //384
    if(blockCounts[12] != 0) pairHMM_register_fp32_subwarp_dataset<13,2><<<1 + blockCounts[12]/2,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 12*seq_size, blockCounts[12]); //416
    if(blockCounts[13] != 0) pairHMM_register_fp32_subwarp_dataset<14,2><<<1 + blockCounts[13]/2,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 13*seq_size, blockCounts[13]); //448
    if(blockCounts[14] != 0) pairHMM_register_fp32_subwarp_dataset<15,2><<<1 + blockCounts[14]/2,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 14*seq_size, blockCounts[14]); //480
    if(blockCounts[15] != 0) pairHMM_register_fp32_subwarp_dataset<16,2><<<1 + blockCounts[15]/2,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 15*seq_size, blockCounts[15]); //512

    if(blockCounts[16] != 0) pairHMM_register_fp32_subwarp_dataset<9,1><<<blockCounts[16],64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 16*seq_size, blockCounts[16]);
    if(blockCounts[17] != 0) pairHMM_register_fp32_subwarp_dataset<9,1><<<blockCounts[17],64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 17*seq_size, blockCounts[17]);
    if(blockCounts[18] != 0) pairHMM_register_fp32_subwarp_dataset<10,1><<<blockCounts[18],64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 18*seq_size, blockCounts[18]);
    if(blockCounts[19] != 0) pairHMM_register_fp32_subwarp_dataset<10,1><<<blockCounts[19],64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 19*seq_size, blockCounts[19]);
    if(blockCounts[20] != 0) pairHMM_register_fp32_subwarp_dataset<11,1><<<blockCounts[20],64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 20*seq_size, blockCounts[20]);
    if(blockCounts[21] != 0) pairHMM_register_fp32_subwarp_dataset<11,1><<<blockCounts[21],64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 21*seq_size, blockCounts[21]);
    if(blockCounts[22] != 0) pairHMM_register_fp32_subwarp_dataset<12,1><<<blockCounts[22],64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 22*seq_size, blockCounts[22]);
    if(blockCounts[23] != 0) pairHMM_register_fp32_subwarp_dataset<12,1><<<blockCounts[23],64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 23*seq_size, blockCounts[23]);
    if(blockCounts[24] != 0) pairHMM_register_fp32_subwarp_dataset<13,1><<<blockCounts[24],64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 24*seq_size, blockCounts[24]);
    if(blockCounts[25] != 0) pairHMM_register_fp32_subwarp_dataset<13,1><<<blockCounts[25],64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 25*seq_size, blockCounts[25]);
    if(blockCounts[26] != 0) pairHMM_register_fp32_subwarp_dataset<14,1><<<blockCounts[26],64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 26*seq_size, blockCounts[26]);
    if(blockCounts[27] != 0) pairHMM_register_fp32_subwarp_dataset<14,1><<<blockCounts[27],64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 27*seq_size, blockCounts[27]);
    if(blockCounts[28] != 0) pairHMM_register_fp32_subwarp_dataset<15,1><<<blockCounts[28],64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 28*seq_size, blockCounts[28]);
    if(blockCounts[29] != 0) pairHMM_register_fp32_subwarp_dataset<15,1><<<blockCounts[29],64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 29*seq_size, blockCounts[29]);
    if(blockCounts[30] != 0) pairHMM_register_fp32_subwarp_dataset<16,1><<<blockCounts[30],64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 30*seq_size, blockCounts[30]);
    if(blockCounts[31] != 0) pairHMM_register_fp32_subwarp_dataset<16,1><<<blockCounts[31],64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 31*seq_size, blockCounts[31]);

    if(blockCounts[32] != 0) pairHMM_shared_fp32_dataset<16,2><<<blockCounts[32],128>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 32*seq_size);
    if(blockCounts[33] != 0) pairHMM_shared_fp32_dataset<16,4><<<blockCounts[33],256>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 33*seq_size);
    if(blockCounts[34] != 0) pairHMM_shared_fp32_dataset<16,8><<<blockCounts[34],512>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 34*seq_size);
    if(blockCounts[35] != 0) pairHMM_shared_fp32_dataset<16,16><<<blockCounts[35],1024>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 35*seq_size);
    hipEventRecord(stop);
    hipMemcpy(sumCPU, sumGPU, seq_size*sizeof(float), hipMemcpyDeviceToHost);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&total_time, start, stop);

    return total_time;
}
