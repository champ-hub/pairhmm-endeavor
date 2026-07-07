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
__constant__ double qualfp64[128];
__constant__ double powDfp64[2049];

template <unsigned int N, unsigned int P> 
__global__ void pairHMM_register_fp64_subwarp(unsigned long long* rslen, unsigned long long* haplen, unsigned char* hap, unsigned char* rs, unsigned char* q, unsigned char* i, unsigned char* d, double* finalsum)
{
    __shared__ unsigned char _rs[64*N];
    __shared__ unsigned char _q[64*N];
    __shared__ unsigned char _i[64*N];
    __shared__ unsigned char _d[64*N];  
    
    const unsigned short subwarp = 64/P;
    const unsigned int laneID = threadIdx.x%subwarp, warpID = ((threadIdx.x/subwarp)%P);
    const unsigned int seqidx = P*blockIdx.x + warpID, idx = N*subwarp*warpID;

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
        Mprev[h] = (_hap[h] == rs0) ?  (1 - qualfp64[q0 - 33]) : qualfp64[q0 - 33]*0.3333333333333333333;
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
        d0 = 0.9*qualfp64[_d[idx + k-1] - 33];
        d1 = _d[idx + k];
        tmm = 1 - qualfp64[in0 - 33] - qualfp64[d1 - 33];
        value = __shfl_up(Mprev[N-1], 1);
        y1 = __shfl_up(Iprev[N-1], 1);
        
        y2 = (_hap[0] == rs0) ?  (1 - qualfp64[q0 - 33]) : qualfp64[q0 - 33]*0.3333333333333333333;
        y2 *= (value*tmm + 0.9*y1 + d0*10*(y3 - value));
    
        if(laneID == 0) y2 = 0.0;
        
        M = y2;
        value = M;
        
        for(int h = 1; h < N; h++){
            y2 = (_hap[h] == rs0) ?  (1 - qualfp64[q0 - 33]) : qualfp64[q0 - 33]*0.3333333333333333333;
            y2 *= (Mprev[h-1]*tmm + 0.9*Iprev[h-1] + d0*y3);

            y3 = y3*0.1 + Mprev[h-1];

            Iprev[h-1] = Mprev[h-1]*qualfp64[in0 - 33] + Iprev[h-1]*0.1;
            Mprev[h-1] = M;

            M = y2;
            value = value*0.1 + M;        
        }       

        Iprev[N-1] = Mprev[N-1]*qualfp64[in0 - 33] + Iprev[N-1]*0.1;
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

    if(laneID == subwarp - 1) finalsum[seqidx] = y1;
}

template <unsigned int N, unsigned int P> 
__global__ void pairHMM_shared_fp64(unsigned long long* rslen, unsigned long long* haplen, unsigned char* hap, unsigned char* rs, unsigned char* q, unsigned char* i, unsigned char* d, double* finalsum)
{
    __shared__ double _aux[P+1];
    __shared__ double _M[P+1];
    __shared__ double _I[P+1];

    _M[0] = _I[0] = _aux[0] = 0;
    
    const unsigned int laneID = threadIdx.x%64, tid = threadIdx.x;
    unsigned int seqidx = blockIdx.x;
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
        Mprev[h] = (_hap[h] == rs0) ?  (1 - qualfp64[q0 - 33]) : qualfp64[q0 - 33]*0.3333333333333333333333;
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
        d0 = 0.9*qualfp64[d1 - 33];
        d1 = d[ridx + k];
        tmm = 1 - qualfp64[in - 33] - qualfp64[d1 - 33];
        value = __shfl_up(Mprev[N-1], 1);
        y1 = __shfl_up(Iprev[N-1], 1);
        
        y2 = (_hap[0] == rs0) ?  (1 - qualfp64[q0 - 33]) : qualfp64[q0 - 33]*0.3333333333333333333333;
        if(laneID == 0) y2 *= (_M[tid/64]*tmm + 0.9*(_I[tid/64] + d0*10*(y3 - _M[tid/64])));
        else y2 *= (value*tmm + 0.9*y1 + d0*10*(y3 - value));

        M = y2;
        I = Mprev[0]*qualfp64[in - 33] + Iprev[0]*0.1;

        value = M;
        
        for(int h = 1; h < N; h++){
            y2 = (_hap[h] == rs0) ?  (1 - qualfp64[q0 - 33]) : qualfp64[q0 - 33]*0.3333333333333333333333;
            y2 *= (Mprev[h-1]*tmm + 0.9*Iprev[h-1] + d0*y3);

            y3 = fma(y3,0.1,Mprev[h-1]);

            Mprev[h-1] = M;
            Iprev[h-1] = I;

            M = y2;
            I = Mprev[h]*qualfp64[in - 33] + Iprev[h]*0.1;  
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

float pairhmm_benchmark_fp64(unsigned long long* rslen, unsigned long long* haplen, unsigned char* hap, unsigned char* rs, unsigned char* q, unsigned char* in, unsigned char* d, double* sumGPU, double* sumCPU, unsigned long long gcups, int maxsize, unsigned int seq_size){
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

    if(maxsize == 32){
        hipEventRecord(start);
        pairHMM_register_fp64_subwarp<16,32><<<seq_size/32,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU);
        hipEventRecord(stop);
    }
    else if(maxsize == 64){
        hipEventRecord(start);
        pairHMM_register_fp64_subwarp<16,16><<<seq_size/16,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU);
        hipEventRecord(stop);
    }
    else if(maxsize == 128){
        hipEventRecord(start);
        pairHMM_register_fp64_subwarp<16,8><<<seq_size/8,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU);
        hipEventRecord(stop);
    }
    else if(maxsize == 256){
        hipEventRecord(start);
        pairHMM_register_fp64_subwarp<16,4><<<seq_size/4,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU);
        hipEventRecord(stop);
    }
    else if(maxsize == 512){
        hipEventRecord(start);
        pairHMM_register_fp64_subwarp<16,2><<<seq_size/2,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU);
        hipEventRecord(stop);
    }
    else if(maxsize == 1024){
        hipEventRecord(start);
        pairHMM_register_fp64_subwarp<16,1><<<seq_size,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU);
        hipEventRecord(stop);
    }
    else if(maxsize == 2048){
        hipEventRecord(start);
        pairHMM_shared_fp64<16,2><<<seq_size,128>>>(rslen, haplen, hap, rs, q, in, d, sumGPU);
        hipEventRecord(stop);
    }
    else if(maxsize == 4096){
        hipEventRecord(start);
        pairHMM_shared_fp64<16,4><<<seq_size,256>>>(rslen, haplen, hap, rs, q, in, d, sumGPU);
        hipEventRecord(stop);
    }
    else if(maxsize == 8192){
        hipEventRecord(start);
        pairHMM_shared_fp64<16,8><<<seq_size,512>>>(rslen, haplen, hap, rs, q, in, d, sumGPU);
        hipEventRecord(stop);
    }
    else if(maxsize == 16384){
        hipEventRecord(start);
        pairHMM_shared_fp64<16,16><<<seq_size,1024>>>(rslen, haplen, hap, rs, q, in, d, sumGPU);
        hipEventRecord(stop);
    }
    else if(maxsize == 32768){
        hipEventRecord(start);
        pairHMM_shared_fp64<32,16><<<seq_size,1024>>>(rslen, haplen, hap, rs, q, in, d, sumGPU);
        hipEventRecord(stop);
    }
    else if(maxsize == 65536){
        hipEventRecord(start);
        pairHMM_shared_fp64<64,16><<<seq_size,1024>>>(rslen, haplen, hap, rs, q, in, d, sumGPU);
        hipEventRecord(stop);
    }
    else if(maxsize == 131072){
        hipEventRecord(start);
        pairHMM_shared_fp64<128,16><<<seq_size,1024>>>(rslen, haplen, hap, rs, q, in, d, sumGPU);
        hipEventRecord(stop);
    }

    hipMemcpy(sumCPU, sumGPU, seq_size*sizeof(double), hipMemcpyDeviceToHost);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&total_time, start, stop);

    //printf("%.8f\n", gcups/(double)(total_time*pow(10,6)));
    //for(int j = 0; j < 32; j++)
    //    printf("%.48f\n", log10(sumCPU[j]) - g_ctxd);

    return total_time;
}


//PAIRHMM FP32 FUNCTIONS
__constant__ float qualfp32[128];
__constant__ float powDfp32[2049];

template <unsigned int N, unsigned int P> 
__global__ void pairHMM_register_fp32_subwarp(unsigned long long* rslen, unsigned long long* haplen, unsigned char* hap, unsigned char* rs, unsigned char* q, unsigned char* i, unsigned char* d, float* finalsum)
{
    __shared__ unsigned char _rs[64*N];
    __shared__ unsigned char _q[64*N];
    __shared__ unsigned char _i[64*N];
    __shared__ unsigned char _d[64*N];   
    
    const unsigned int subwarp = 64/P;
    const unsigned int laneID = threadIdx.x%subwarp, warpID = ((threadIdx.x/subwarp)%P);
    const unsigned int seqidx = P*blockIdx.x + warpID, idx = N*subwarp*warpID;

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
        
        Mprev[h] = (_hap[h] == rs0) ?  (1.0f - qualfp32[q0 - 33]) : qualfp32[q0 - 33]*0.3333333333333333333f;
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
        d0 = qualfp32[_d[idx + k-1] - 33];
        d1 = _d[idx + k];
        tmm = 1.0f - qualfp32[in - 33] - qualfp32[d1 - 33];
        value = __shfl_up( Mprev[N-1], 1);
        y1 = __shfl_up( Iprev[N-1], 1);
        
        y2 = (_hap[0] == rs0) ?  (1.0f - qualfp32[q0 - 33]) : qualfp32[q0 - 33]*0.3333333333333333333f;
        y2 *= (value*tmm + 0.9f*(y1 + d0*10.0f*(y3 - value)));
    
        if(laneID == 0) y2 = 0.0f;
        
        M = y2;
        I = Mprev[0]*qualfp32[in - 33] + Iprev[0]*0.1f;

        value = M;
        
        for(int h = 1; h < N; h++){
            y2 = (_hap[h] == rs0) ? (1.0f - qualfp32[q0 - 33]) : qualfp32[q0 - 33]*0.3333333333333333333f;
            y2 *= (Mprev[h-1]*tmm + 0.9f*(Iprev[h-1] + d0*y3));

            y3 = y3*0.1f + Mprev[h-1];

            Mprev[h-1] = M;
            Iprev[h-1] = I;

            M = y2;
            I = Mprev[h]*qualfp32[in - 33] + Iprev[h]*0.1f;  
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
        finalsum[seqidx] = y1;
    }
}

template <unsigned int N, unsigned int P> 
__global__ void pairHMM_shared_fp32(unsigned long long* rslen, unsigned long long* haplen, unsigned char* hap, unsigned char* rs, unsigned char* q, unsigned char* i, unsigned char* d, float* finalsum)
{
    __shared__ float _aux[P+1];
    __shared__ float _M[P+1];
    __shared__ float _I[P+1];

    _M[0] = _I[0] = _aux[0] = 0.0f;
    
    const unsigned int laneID = threadIdx.x%64, tid = threadIdx.x, seqidx = blockIdx.x;
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
        Mprev[h] = (_hap[h] == rs0) ?  (1.0f - qualfp32[q0 - 33]) : qualfp32[q0 - 33]*0.3333333333333333333f;
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
        d0 = qualfp32[d1 - 33];
        d1 = d[ridx + k];
        tmm = 1.0f - qualfp32[in - 33] - qualfp32[d1 - 33];
        value = __shfl_up( Mprev[N-1], 1);
        y1 = __shfl_up( Iprev[N-1], 1);
        
        y2 = (_hap[0] == rs0) ?  (1.0f - qualfp32[q0 - 33]) : qualfp32[q0 - 33]*0.3333333333333333333f;
        if(laneID == 0) y2 *= (_M[tid/64]*tmm + 0.9f*(_I[tid/64] + d0*10.0f*(y3 - _M[tid/64])));
        else y2 *= (value*tmm + 0.9f*(y1 + d0*10.0f*(y3 - value)));

        M = y2;
        I = Mprev[0]*qualfp32[in - 33] + Iprev[0]*0.1f;

        value = M;
        
        for(int h = 1; h < N; h++){
            y2 = (_hap[h] == rs0) ?  (1.0f - qualfp32[q0 - 33]) : qualfp32[q0 - 33]*0.3333333333333333333f;
            y2 *= (Mprev[h-1]*tmm + 0.9f*(Iprev[h-1] + d0*y3));

            y3 = y3*0.1f + Mprev[h-1];

            Mprev[h-1] = M;
            Iprev[h-1] = I;

            M = y2;
            I = Mprev[h]*qualfp32[in - 33] + Iprev[h]*0.1f;  
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

float pairhmm_benchmark_fp32(unsigned long long* rslen, unsigned long long* haplen, unsigned char* hap, unsigned char* rs, unsigned char* q, unsigned char* in, unsigned char* d, float* sumGPU, float* sumCPU, unsigned long long gcups, int maxsize, unsigned int seq_size){
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

    if(maxsize == 32){
        hipEventRecord(start);
        pairHMM_register_fp32_subwarp<16,32><<<seq_size/32,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU);
        hipEventRecord(stop);
    }
    else if(maxsize == 64){
        hipEventRecord(start);
        pairHMM_register_fp32_subwarp<16,16><<<seq_size/16,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU);
        hipEventRecord(stop);
    }
    else if(maxsize == 128){
        hipEventRecord(start);
        pairHMM_register_fp32_subwarp<16,8><<<seq_size/8,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU);
        hipEventRecord(stop);
    }
    else if(maxsize == 256){
        hipEventRecord(start);
        pairHMM_register_fp32_subwarp<16,4><<<seq_size/4,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU);
        hipEventRecord(stop);
    }
    else if(maxsize == 512){
        hipEventRecord(start);
        pairHMM_register_fp32_subwarp<16,2><<<seq_size/2,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU);
        hipEventRecord(stop);
    }
    else if(maxsize == 1024){
        hipEventRecord(start);
        pairHMM_register_fp32_subwarp<16,1><<<seq_size,64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU);
        hipEventRecord(stop);
    }
    else if(maxsize == 2048){
        hipEventRecord(start);
        pairHMM_shared_fp32<16,2><<<seq_size,128>>>(rslen, haplen, hap, rs, q, in, d, sumGPU);
        hipEventRecord(stop);
    }
    else if(maxsize == 4096){
        hipEventRecord(start);
        pairHMM_shared_fp32<16,4><<<seq_size,256>>>(rslen, haplen, hap, rs, q, in, d, sumGPU);
        hipEventRecord(stop);
    }
    else if(maxsize == 8192){
        hipEventRecord(start);
        pairHMM_shared_fp32<16,8><<<seq_size,512>>>(rslen, haplen, hap, rs, q, in, d, sumGPU);
        hipEventRecord(stop);
    }
    else if(maxsize == 16384){
        hipEventRecord(start);
        pairHMM_shared_fp32<16,16><<<seq_size,1024>>>(rslen, haplen, hap, rs, q, in, d, sumGPU);
        hipEventRecord(stop);
    }
    else if(maxsize == 32768){
        hipEventRecord(start);
        pairHMM_shared_fp32<32,16><<<seq_size,1024>>>(rslen, haplen, hap, rs, q, in, d, sumGPU);
        hipEventRecord(stop);
    }
    else if(maxsize == 65536){
        hipEventRecord(start);
        pairHMM_shared_fp32<64,16><<<seq_size,1024>>>(rslen, haplen, hap, rs, q, in, d, sumGPU);
        hipEventRecord(stop);
    }
    else if(maxsize == 131072){
        hipEventRecord(start);
        pairHMM_shared_fp32<128,16><<<seq_size,1024>>>(rslen, haplen, hap, rs, q, in, d, sumGPU);
        hipEventRecord(stop);
    }
    hipMemcpy(sumCPU, sumGPU, seq_size*sizeof(float),hipMemcpyDeviceToHost);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&total_time, start, stop);

    return total_time;
}
