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
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

using namespace std;

//FP64 FUNCTIONS
__constant__ double qualfp64[128];
__constant__ double powDfp64[1025];

template <unsigned int N, unsigned int P> 
__global__ void pairHMM_register_fp64_subwarp(unsigned long long* rslen, unsigned long long* haplen, unsigned char* hap, unsigned char* rs, unsigned char* q, unsigned char* i, unsigned char* d, double* finalsum)
{
    alignas(N) __shared__ char _rs[32*N];
    alignas(N) __shared__ char _q[32*N];
    alignas(N) __shared__ char _i[32*N];
    alignas(N) __shared__ char _d[32*N];  
    
    const unsigned int subwarp = 32/P;
    const unsigned int laneID = threadIdx.x%subwarp, warpID = ((threadIdx.x/subwarp)%P);
    const unsigned int seqidx = P*blockIdx.x + warpID;
    unsigned int idx = N*subwarp*warpID;

    double Mprev[N] = {0};
    double Iprev[N] = {0};
    char _hap[N] = {0};

    double M = 0; 
    double I = 0; 
    double y1 = 0; 
    double y2 = 0; 
    double y3 = 0; 
    double y4 = 0; 
    double value = 0;
    double d0 = 0;
    double tmm = 0;
    double one = 0;
    double third = 0;
    char rs0;
    char q0;
    char in;
    char d1;

    const unsigned long long rsize = -rslen[seqidx] + rslen[seqidx + 1];
    const unsigned long long hsize = -haplen[seqidx] + haplen[seqidx + 1];
    const unsigned long long ridx = rslen[seqidx];
    const unsigned long long hidx = haplen[seqidx];

    rs0 = rs[ridx];
    q0 = q[ridx];
    one = 1 - qualfp64[q0]; 
    third = qualfp64[q0]*0.3333333333333333333;
    
    for(unsigned int h = 0; h < N; h++){
        _hap[h] = hap[hidx + N*laneID + h];
        _rs[idx + N*laneID + h] = rs[ridx + N*laneID + h];
        _q[idx + N*laneID + h]  = q[ridx + N*laneID + h];
        _i[idx + N*laneID + h]  = i[ridx + N*laneID + h];
        _d[idx + N*laneID + h]  = d[ridx + N*laneID + h];
        Mprev[h] = (_hap[h] == rs0) ?  one : third;
        Mprev[h] *= 0.9*ldexp(1.0, 1020.0)/(double)(hsize); 
        value = value*0.1 + Mprev[h];  
    }

    y3 = __shfl_up_sync(0xffffffff, value, 1);
    
    if(laneID == 0) y3 = 0;

    y1 = __shfl_up_sync(0xffffffff, y3, 1);
    if (laneID >= 1) y3 = fma(y1,powDfp64[N],y3);
    y1 = __shfl_up_sync(0xffffffff, y3, 2);
    if (laneID >= 2) y3 = fma(y1,powDfp64[2*N],y3);
    y1 = __shfl_up_sync(0xffffffff, y3, 4);
    if (laneID >= 4) y3 = fma(y1,powDfp64[4*N],y3);
    y1 = __shfl_up_sync(0xffffffff, y3, 8);
    if (laneID >= 8) y3 = fma(y1,powDfp64[8*N],y3);

    y4 = __shfl_up_sync(0xffffffff, Mprev[N-1], 1);
    y1 = __shfl_up_sync(0xffffffff, Iprev[N-1], 1);

    for(int k = 1; k < rsize; k++){
        rs0 = _rs[idx + k];
        q0 = _q[idx + k];
        in = _i[idx + k];
        d0 = qualfp64[_d[idx + k-1]];
        d1 = _d[idx + k];
        tmm = 1 - qualfp64[in] - qualfp64[d1];
        one = 1 - qualfp64[q0]; 
        third = qualfp64[q0]*0.3333333333333333333;
        
        y2 = (_hap[0] == rs0) ?  one : third;
        y2 *= (y4*tmm + 0.9*(y1 + d0*10*(y3 - y4)));
    
        if(laneID == 0) y2 = 0;
        
        M = y2;
        I = Mprev[0]*qualfp64[in] + Iprev[0]*0.1;

        value = M;
        
        for(int h = 1; h < N; h++){
            y2 = (_hap[h] == rs0) ? one : third;
            y2 *= (Mprev[h-1]*tmm + 0.9*(Iprev[h-1] + d0*y3));

            y3 = fma(y3,0.1,Mprev[h-1]);

            Mprev[h-1] = M;
            Iprev[h-1] = I;

            M = y2;
            I = Mprev[h]*qualfp64[in] + Iprev[h]*0.1;  
            value = fma(value,0.1,M);        
        }       

        Mprev[N-1] = M;
        Iprev[N-1] = I; 

        y3 = __shfl_up_sync(0xffffffff, value, 1);
        
        if(laneID == 0) y3 = 0;

        y1 = __shfl_up_sync(0xffffffff, y3, 1);
        if (laneID >= 1) y3 = fma(y1,powDfp64[N],y3);
        y1 = __shfl_up_sync(0xffffffff, y3, 2);
        if (laneID >= 2) y3 = fma(y1,powDfp64[2*N],y3);
        y1 = __shfl_up_sync(0xffffffff, y3, 4);
        if (laneID >= 4) y3 = fma(y1,powDfp64[4*N],y3);
        y1 = __shfl_up_sync(0xffffffff, y3, 8);
        if (laneID >= 8) y3 = fma(y1,powDfp64[8*N],y3);

        y4 = __shfl_up_sync(0xffffffff, Mprev[N-1], 1);
        y1 = __shfl_up_sync(0xffffffff, Iprev[N-1], 1);
    }

    y1 = 0;
    for(int h = 0; h < N; h++)
        if(N*laneID + h < hsize) y1 += Mprev[h] + Iprev[h]; 

    y2 = __shfl_up_sync(0xffffffff, y1, 1);
    if (laneID >= 1) y1 += y2;
    y2 = __shfl_up_sync(0xffffffff, y1, 2);
    if (laneID >= 2) y1 += y2;
    y2 = __shfl_up_sync(0xffffffff, y1, 4);
    if (laneID >= 4) y1 += y2;
    y2 = __shfl_up_sync(0xffffffff, y1, 8);
    if (laneID >= 8) y1 += y2;
    y2 = __shfl_up_sync(0xffffffff, y1, 16);
    if (laneID >= 16) y1 += y2;

    if(laneID == subwarp - 1) finalsum[seqidx] = y1;
}

template <unsigned int N, unsigned int P> 
__global__ void pairHMM_shared_fp64(unsigned long long* rslen, unsigned long long* haplen, unsigned char* hap, unsigned char* rs, unsigned char* q, unsigned char* i, unsigned char* d, double* finalsum)
{
    __shared__ double _aux[P+1];
    __shared__ double _M[P+1];
    __shared__ double _I[P+1];

    _M[0] = _I[0] = _aux[0] = 0;
    
    const unsigned int laneID = threadIdx.x%32, tid = threadIdx.x, seqidx = blockIdx.x;
    double Mprev[N] = {0};
    double Iprev[N] = {0};
    char _hap[N] = {0};

    double M = 0; 
    double I = 0; 
    double y1 = 0; 
    double y2 = 0; 
    double y3 = 0; 
    double y4 = 0; 
    double value = 0;
    double d0 = 0;
    double tmm = 0;
    double one = 0;
    double third = 0;
    char rs0;
    char q0;
    char in;
    char d1;

    const unsigned long long rsize = -rslen[seqidx] + rslen[seqidx + 1];
    const unsigned long long hsize = -haplen[seqidx] + haplen[seqidx + 1];
    const unsigned long long ridx = rslen[seqidx];
    const unsigned long long hidx = haplen[seqidx];

    rs0 = rs[ridx];
    q0 = q[ridx];
    d1 = d[ridx];
    one = 1 - qualfp64[q0]; 
    third = qualfp64[q0]*0.3333333333333333333;
    
    for(unsigned int h = 0; h < N; h++){
        _hap[h] = hap[hidx + N*tid + h];
        Mprev[h] = (_hap[h] == rs0) ? one : third;
        Mprev[h] *= 0.9*ldexp(1.0, 1020.0)/(double)(hsize); 
        value = value*0.1 + Mprev[h];  
    }

    y3 = __shfl_up_sync(0xffffffff, value, 1);
    y4 = __shfl_up_sync(0xffffffff, Mprev[N-1], 1);
    y1 = __shfl_up_sync(0xffffffff, Iprev[N-1], 1);

    if(laneID == 0) y3 = 0;
    else if(laneID == 31){
        _M[1 + (tid/32)] = Mprev[N-1];
        _I[1 + (tid/32)] = Iprev[N-1];
    }

    y1 = __shfl_up_sync(0xffffffff, y3, 1);
    if (laneID >= 1) y3 = fma(y1,powDfp64[N],y3);
    y1 = __shfl_up_sync(0xffffffff, y3, 2);
    if (laneID >= 2) y3 = fma(y1,powDfp64[2*N],y3);
    y1 = __shfl_up_sync(0xffffffff, y3, 4);
    if (laneID >= 4) y3 = fma(y1,powDfp64[4*N],y3);
    y1 = __shfl_up_sync(0xffffffff, y3, 8);
    if (laneID >= 8) y3 = fma(y1,powDfp64[8*N],y3);

    if(laneID == 31) _aux[1 + (tid/32)] = y3*powDfp64[32] + value;
    __syncthreads();

    y3 += _aux[tid/32]*powDfp64[32*laneID];

    for(int k = 1; k < rsize; k++){
        rs0 = rs[ridx + k];
        q0 = q[ridx + k];
        in = i[ridx + k];
        d0 = qualfp64[d1];
        d1 = d[ridx + k];
        tmm = 1 - qualfp64[in] - qualfp64[d1];
        one = 1 - qualfp64[q0]; 
        third = qualfp64[q0]*0.3333333333333333333;
        
        y2 = (_hap[0] == rs0) ?  one : third;
        if(laneID == 0) y1 = (_M[tid/32]*tmm + 0.9*(_I[tid/32] + d0*10*(y3 - _M[tid/32])));
        else y1 = (y4*tmm + 0.9*(y1 + d0*10*(y3 - y4)));
        
        M = y2 * y1;
        I = Iprev[0]*0.1;

        value = M;
        
        for(int h = 1; h < N; h++){
            y2 = (_hap[h] == rs0) ? one : third;
            y1 = (Mprev[h-1]*tmm + 0.9*(Iprev[h-1] + d0*y3));

            y3 = y3*0.1 + Mprev[h-1];

            Iprev[h-1] = I + Mprev[h-1]*qualfp64[in];
            Mprev[h-1] = M;

            M = y2 * y1;
            I = Iprev[h]*0.1;  
            value = value*0.1 + M;       
        }       

        Iprev[N-1] = I + Mprev[N-1]*qualfp64[in];
        Mprev[N-1] = M;

        y3 = __shfl_up_sync(0xffffffff, value, 1);

        __syncthreads();
        if(laneID == 0) y3 = 0;
        else if(laneID == 31){
            _M[1 + (tid/32)] = Mprev[N-1];
            _I[1 + (tid/32)] = Iprev[N-1];
        }        

        y1 = __shfl_up_sync(0xffffffff, y3, 1);
        if (laneID >= 1) y3 = fma(y1,powDfp64[N],y3);
        y1 = __shfl_up_sync(0xffffffff, y3, 2);
        if (laneID >= 2) y3 = fma(y1,powDfp64[2*N],y3);
        y1 = __shfl_up_sync(0xffffffff, y3, 4);
        if (laneID >= 4) y3 = fma(y1,powDfp64[4*N],y3);
        y1 = __shfl_up_sync(0xffffffff, y3, 8);
        if (laneID >= 8) y3 = fma(y1,powDfp64[8*N],y3);

        y4 = __shfl_up_sync(0xffffffff, Mprev[N-1], 1);
        y1 = __shfl_up_sync(0xffffffff, Iprev[N-1], 1);

        if(laneID == 31) _aux[1 + (tid/32)] = y3*powDfp64[32] + value;
        __syncthreads();

        y3 += _aux[tid/32]*powDfp64[32*laneID];
    }
    

    y1 = 0;
    for(int h = 0; h < N; h++)
        if(N*tid + h < hsize) y1 += Mprev[h] + Iprev[h]; 
    
    y2 = __shfl_up_sync(0xffffffff, y1, 1);
    if (laneID >= 1) y1 += y2;
    y2 = __shfl_up_sync(0xffffffff, y1, 2);
    if (laneID >= 2) y1 += y2;
    y2 = __shfl_up_sync(0xffffffff, y1, 4);
    if (laneID >= 4) y1 += y2;
    y2 = __shfl_up_sync(0xffffffff, y1, 8);
    if (laneID >= 8) y1 += y2;
    y2 = __shfl_up_sync(0xffffffff, y1, 16);
    if (laneID >= 16) y1 += y2;

    if(laneID == 31) _aux[1 + (tid/32)] = y1;
    __syncthreads();

    if(tid < 32){
        if(tid < P) y1 = _aux[1 + tid];
        else y1 = 0;

        y2 = __shfl_up_sync(0xffffffff, y1, 1);
        if (laneID >= 1) y1 += y2;
        y2 = __shfl_up_sync(0xffffffff, y1, 2);
        if (laneID >= 2) y1 += y2;
        y2 = __shfl_up_sync(0xffffffff, y1, 4);
        if (laneID >= 4) y1 += y2;
        y2 = __shfl_up_sync(0xffffffff, y1, 8);
        if (laneID >= 8) y1 += y2;
        y2 = __shfl_up_sync(0xffffffff, y1, 16);
        if (laneID >= 16) y1 += y2;

        if(laneID == 31) finalsum[seqidx] = y1;
    }    
}


template <unsigned int N, unsigned int P> 
__global__ void pairHMM_tbcluster_fp64(unsigned long long* rslen, unsigned long long* haplen, unsigned char* hap, unsigned char* rs, unsigned char* q, unsigned char* i, unsigned char* d, double* finalsum)
{
    cg::cluster_group cluster = cg::this_cluster();

    __shared__ double _aux[P+1];
    __shared__ double _M[P+1];
    __shared__ double _I[P+1];

    extern __shared__ double smem[];

    _M[0] = _I[0] = _aux[0] = 0;

    const unsigned int B = cluster.num_blocks();

    for(int i = threadIdx.x; i < 3 + B; i += blockDim.x) smem[i] = 0;

    cluster.sync();

    const unsigned int laneID = threadIdx.x%32, tid = threadIdx.x, tidB = cluster.thread_rank(), seqidx = (blockIdx.x)/B;
    unsigned int laneID2 = 0, block = cluster.block_rank();

    if(tidB%blockDim.x < 32) laneID2 = tid;
    else laneID2 = 32;

    double Mprev[N] = {0};
    double Iprev[N] = {0};
    char _hap[N] = {0};

    double M = 0; 
    double I = 0; 
    double y1 = 0; 
    double y2 = 0; 
    double y3 = 0; 
    double value = 0;
    double d0 = 0;
    double tmm = 0;
    char rs0;
    char q0;
    char in;
    char d1;

    double *M_smem = cluster.map_shared_rank(smem, ((block + 1)%B));
    double *I_smem = cluster.map_shared_rank(smem + 1, ((block + 1)%B));
    double *aux_smem = cluster.map_shared_rank(smem + 2, ((block + 1)%B));
    double *res_smem = cluster.map_shared_rank(smem + 3, 0);
    
    const unsigned long long rsize = -rslen[seqidx] + rslen[seqidx + 1];
    const unsigned long long hsize = -haplen[seqidx] + haplen[seqidx + 1];
    const unsigned long long ridx = rslen[seqidx];
    const unsigned long long hidx = haplen[seqidx];

    rs0 = rs[ridx];
    q0 = q[ridx];
    d1 = d[ridx];
    
    for(unsigned int h = 0; h < N; h++){
        _hap[h] = hap[hidx + N*tidB + h];
        Mprev[h] = (_hap[h] == rs0) ?  (1 - qualfp64[q0]) : qualfp64[q0]*0.3333333333333333333;
        Mprev[h] *= 0.9*ldexp(1.0, 1020.0)/(double)(hsize); 
        value = value*0.1 + Mprev[h];  
    }

    y3 = __shfl_up_sync(0xffffffff, value, 1);
    if(laneID == 0) y3 = 0;
    
    if(laneID == 31){
        _M[1 + (tid/32)] = Mprev[N-1];
        _I[1 + (tid/32)] = Iprev[N-1];
    }

    if(tidB%blockDim.x == blockDim.x - 1 && ((block + 1)%B) != 0){
        M_smem[0] = Mprev[N-1];
        I_smem[0] = Iprev[N-1];
    }
   
    y1 = __shfl_up_sync(0xffffffff, y3, 1);
    if (laneID >= 1) y3 = fma(y1,powDfp64[N],y3);
    y1 = __shfl_up_sync(0xffffffff, y3, 2);
    if (laneID >= 2) y3 = fma(y1,powDfp64[2*N],y3);
    y1 = __shfl_up_sync(0xffffffff, y3, 4);
    if (laneID >= 4) y3 = fma(y1,powDfp64[4*N],y3);
    y1 = __shfl_up_sync(0xffffffff, y3, 8);
    if (laneID >= 8) y3 = fma(y1,powDfp64[8*N],y3);

    if(laneID == 31) _aux[1 + (tid/32)] = y3*powDfp64[32] + value;
    __syncthreads();

    if(tidB%blockDim.x == blockDim.x - 1 && (block + 1)%B != 0) aux_smem[0] = y3*powDfp64[32] + value;
    cluster.sync();

    y3 += _aux[tid/32]*powDfp64[32*laneID] + smem[2]*powDfp64[32*laneID2]; 
    
    for(int k = 1; k < rsize; k++){
        rs0 = rs[ridx+ k];
        q0 = q[ridx + k];
        in = i[ridx + k];
        d0 = qualfp64[d1];
        d1 = d[ridx + k];
        tmm = 1 - qualfp64[in] - qualfp64[d1];
        value = __shfl_up_sync(0xffffffff, Mprev[N-1], 1);
        y1 = __shfl_up_sync(0xffffffff, Iprev[N-1], 1);
        
        y2 = (_hap[0] == rs0) ?  (1 - qualfp64[q0]) : qualfp64[q0]*0.3333333333333333333;
        if(laneID == 0){
            if(tidB%blockDim.x != 0) y1 = (_M[tid/32]*tmm + 0.9*(_I[tid/32] + d0*10*(y3 - _M[tid/32])));
            else y1 = (smem[0]*tmm + 0.9*(smem[1] + d0*10*(y3 - smem[0])));
        }
        else y1 = (value*tmm + 0.9*(y1 + d0*10*(y3 - value)));
        
        M = y2 * y1;
        I = Iprev[0]*0.1;

        value = M;
        
        for(int h = 1; h < N; h++){
            y2 = (_hap[h] == rs0) ? (1 - qualfp64[q0]) : qualfp64[q0]*0.3333333333333333333;
            y1 = (Mprev[h-1]*tmm + 0.9*(Iprev[h-1] + d0*y3));

            y3 = y3*0.1 + Mprev[h-1];

            Iprev[h-1] = I + Mprev[h-1]*qualfp64[in];
            Mprev[h-1] = M;

            M = y2 * y1;
            I = Iprev[h]*0.1;  
            value = value*0.1 + M;       
        }       

        Iprev[N-1] = I + Mprev[N-1]*qualfp64[in];
        Mprev[N-1] = M;

        y3 = __shfl_up_sync(0xffffffff, value, 1);

        __syncthreads();
        if(laneID == 0) y3 = 0;
        if(laneID == 31){
            _M[1 + (tid/32)] = Mprev[N-1];
            _I[1 + (tid/32)] = Iprev[N-1];
        }
        if(tidB%blockDim.x == blockDim.x - 1 && ((block + 1)%B) != 0){
            M_smem[0] = Mprev[N-1];
            I_smem[0] = Iprev[N-1];
        }

        y1 = __shfl_up_sync(0xffffffff, y3, 1);
        if (laneID >= 1) y3 = fma(y1,powDfp64[N],y3);
        y1 = __shfl_up_sync(0xffffffff, y3, 2);
        if (laneID >= 2) y3 = fma(y1,powDfp64[2*N],y3);
        y1 = __shfl_up_sync(0xffffffff, y3, 4);
        if (laneID >= 4) y3 = fma(y1,powDfp64[4*N],y3);
        y1 = __shfl_up_sync(0xffffffff, y3, 8);
        if (laneID >= 8) y3 = fma(y1,powDfp64[8*N],y3);

        if(laneID == 31) _aux[1 + (tid/32)] = y3*powDfp64[32] + value;
        __syncthreads();

        if(tidB%blockDim.x == blockDim.x - 1 && (block + 1)%B != 0) aux_smem[0] = y3*powDfp64[32] + value;
        cluster.sync();

        y3 += _aux[tid/32]*powDfp64[32*laneID] + smem[2]*powDfp64[32*laneID2];
    }
    
    y1 = 0;
    for(int h = 0; h < N; h++)
        if(N*tidB + h < hsize) y1 += Mprev[h] + Iprev[h]; 
    
    y2 = __shfl_up_sync(0xffffffff, y1, 1);
    if (laneID >= 1) y1 += y2;
    y2 = __shfl_up_sync(0xffffffff, y1, 2);
    if (laneID >= 2) y1 += y2;
    y2 = __shfl_up_sync(0xffffffff, y1, 4);
    if (laneID >= 4) y1 += y2;
    y2 = __shfl_up_sync(0xffffffff, y1, 8);
    if (laneID >= 8) y1 += y2;
    y2 = __shfl_up_sync(0xffffffff, y1, 16);
    if (laneID >= 16) y1 += y2;

    if(laneID == 31) _aux[1 + (tid/32)] = y1;
    __syncthreads();

    if(tid < 32){
        if(tid < P) y1 = _aux[1 + tid];
        else y1 = 0;

        y2 = __shfl_up_sync(0xffffffff, y1, 1);
        if (laneID >= 1) y1 += y2;
        y2 = __shfl_up_sync(0xffffffff, y1, 2);
        if (laneID >= 2) y1 += y2;
        y2 = __shfl_up_sync(0xffffffff, y1, 4);
        if (laneID >= 4) y1 += y2;
        y2 = __shfl_up_sync(0xffffffff, y1, 8);
        if (laneID >= 8) y1 += y2;
        y2 = __shfl_up_sync(0xffffffff, y1, 16);
        if (laneID >= 16) y1 += y2;

        if(laneID == 31){
            if((block%B) == 0) smem[3 + (block%B)] = y1;
            else res_smem[(block%B)] = y1;
        } 
    }    

    cluster.sync();

    if((block%B) == 0 && tid < 32){
        if(tid < B){
            y1 = smem[3 + tid];
        }
        else y1 = 0;

        y2 = __shfl_up_sync(0xffffffff, y1, 1);
        if (laneID >= 1) y1 += y2;
        y2 = __shfl_up_sync(0xffffffff, y1, 2);
        if (laneID >= 2) y1 += y2;
        y2 = __shfl_up_sync(0xffffffff, y1, 4);
        if (laneID >= 4) y1 += y2;
        y2 = __shfl_up_sync(0xffffffff, y1, 8);
        if (laneID >= 8) y1 += y2;
        y2 = __shfl_up_sync(0xffffffff, y1, 16);
        if (laneID >= 16) y1 += y2;

        if(laneID == 31) finalsum[seqidx] = y1;
    }
}

float pairhmm_benchmark_fp64(unsigned long long* rlenGPU, unsigned long long* hlenGPU, unsigned char* hapGPU, unsigned char* readGPU, unsigned char* qGPU, unsigned char* iGPU, unsigned char* dGPU, double* sumGPU, double* sumCPU, int maxsize, unsigned int seq_size){
    float total_time = 0;
    double g_ctxd = log10(ldexp(1.0, 1020.0));
    double powvalues[128] = {0};
    
    for(int i = 0; i < 128; i++){
        powvalues[i] = pow(10.0, -(double)i/10.0);
    }

    cudaMemcpyToSymbol(qualfp64, powvalues, sizeof(powvalues));

    double powd[1025] = {0};
    for(int i = 0; i < 1025; i++){
        powd[i] = pow(10.0, -(double)(i));
    }

    cudaMemcpyToSymbol(powDfp64, powd, sizeof(powd));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if(maxsize == 32){
        cudaEventRecord(start);
        pairHMM_register_fp64_subwarp<32,32><<<seq_size/32,32>>>(rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, sumGPU);
        cudaEventRecord(stop);
    }
    else if(maxsize == 64){
        cudaEventRecord(start);
        pairHMM_register_fp64_subwarp<32,16><<<seq_size/16,32>>>(rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, sumGPU);
        cudaEventRecord(stop);
    }
    else if(maxsize == 128){
        cudaEventRecord(start);
        pairHMM_register_fp64_subwarp<32,8><<<seq_size/8,32>>>(rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, sumGPU);
        cudaEventRecord(stop);
    }
    else if(maxsize == 256){
        cudaEventRecord(start);
        pairHMM_register_fp64_subwarp<32,4><<<seq_size/4,32>>>(rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, sumGPU);
        cudaEventRecord(stop);
    }
    else if(maxsize == 512){
        cudaEventRecord(start);
        pairHMM_register_fp64_subwarp<32,2><<<seq_size/2,32>>>(rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, sumGPU);
        cudaEventRecord(stop);
    }
    else if(maxsize == 1024){
        cudaEventRecord(start);
        pairHMM_register_fp64_subwarp<32,1><<<seq_size,32>>>(rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, sumGPU);
        cudaEventRecord(stop);
    }
    else if(maxsize == 2048){
        cudaEventRecord(start);
        pairHMM_shared_fp64<32,2><<<seq_size,64>>>(rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, sumGPU);
        cudaEventRecord(stop);
    }
    else if(maxsize == 4096){
        cudaEventRecord(start);
        pairHMM_shared_fp64<32,4><<<seq_size,128>>>(rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, sumGPU);
        cudaEventRecord(stop);
    }
    else if(maxsize == 8192){
        cudaEventRecord(start);
        pairHMM_shared_fp64<32,8><<<seq_size,256>>>(rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, sumGPU);
        cudaEventRecord(stop);
    }
    else if(maxsize == 16384){
        cudaEventRecord(start);
        pairHMM_shared_fp64<64,8><<<seq_size,256>>>(rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, sumGPU);
        cudaEventRecord(stop);
    }
    else if(maxsize == 32768){
        cudaEventRecord(start);
        pairHMM_shared_fp64<128,8><<<seq_size,256>>>(rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, sumGPU);
        cudaEventRecord(stop);
    }
    else if(maxsize == 65536){
        cudaEventRecord(start);
        pairHMM_shared_fp64<256,8><<<seq_size,256>>>(rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, sumGPU);
        cudaEventRecord(stop);
    }
    else if(maxsize == 131072){
        cudaEventRecord(start);
        pairHMM_shared_fp64<512,8><<<seq_size,256>>>(rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, sumGPU);
        cudaEventRecord(stop);
    }
    else if(maxsize == 16384){
        cudaLaunchConfig_t config = {0};
        config.gridDim = 2*seq_size;
        config.blockDim = 32*8;

        int cluster_size = 2;

        config.dynamicSmemBytes = (3 + cluster_size) * sizeof(double);

        cudaFuncSetAttribute((void *)pairHMM_tbcluster_fp64<32, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes);

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 2;
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;

        config.numAttrs = 1;
        config.attrs = attribute;

        cudaEventRecord(start);
        cudaLaunchKernelEx(&config, pairHMM_tbcluster_fp64<32, 8>, rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, sumGPU);
        cudaEventRecord(stop);
    }
    else if(maxsize == 32768){
        cudaLaunchConfig_t config = {0};
        config.gridDim = 4*seq_size;
        config.blockDim = 32*8;

        int cluster_size = 4;

        config.dynamicSmemBytes = (3 + cluster_size) * sizeof(double);

        cudaFuncSetAttribute((void *)pairHMM_tbcluster_fp64<32, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes);

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 4;
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;

        config.numAttrs = 1;
        config.attrs = attribute;
        
        cudaEventRecord(start);
        cudaLaunchKernelEx(&config, pairHMM_tbcluster_fp64<32, 8>, rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, sumGPU);
        cudaEventRecord(stop);
    }
    else if(maxsize == 65536){
        cudaLaunchConfig_t config = {0};
        config.gridDim = 8*seq_size;
        config.blockDim = 32*8;

        int cluster_size = 8;

        config.dynamicSmemBytes = (3 + cluster_size) * sizeof(double);

        cudaFuncSetAttribute((void *)pairHMM_tbcluster_fp64<32, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes);

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 8;
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;

        config.numAttrs = 1;
        config.attrs = attribute;
        
        cudaEventRecord(start);
        cudaLaunchKernelEx(&config, pairHMM_tbcluster_fp64<32, 8>, rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, sumGPU);
        cudaEventRecord(stop);
    }
    else if(maxsize == 131072){
        cudaLaunchConfig_t config = {0};
        config.gridDim = 8*seq_size;
        config.blockDim = 32*8;

        int cluster_size = 8;

        config.dynamicSmemBytes = (3 + cluster_size) * sizeof(double);

        cudaFuncSetAttribute((void *)pairHMM_tbcluster_fp64<64, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes);

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 8;
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;

        config.numAttrs = 1;
        config.attrs = attribute;
        
        cudaEventRecord(start);
        cudaLaunchKernelEx(&config, pairHMM_tbcluster_fp64<64, 8>, rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, sumGPU);
        cudaEventRecord(stop);
    }

    cudaMemcpy(sumCPU, sumGPU, seq_size*sizeof(double), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&total_time, start, stop);

    return total_time;
}


//FP32 FUNCTIONS
__constant__ float qualfp32[128];
__constant__ float powDfp32[2048];

template <unsigned int N, unsigned int P> 
__global__ void pairHMM_register_fp32_subwarp(unsigned long long* rslen, unsigned long long* haplen, unsigned char* hap, unsigned char* rs, unsigned char* q, unsigned char* i, unsigned char* d, float* finalsum)
{
    alignas(N) __shared__ unsigned char _rs[32*N];
    alignas(N) __shared__ unsigned char _q[32*N];
    alignas(N) __shared__ unsigned char _i[32*N];
    alignas(N) __shared__ unsigned char _d[32*N];   
    
    const unsigned int subwarp = 32/P;
    const unsigned int laneID = threadIdx.x%subwarp, warpID = ((threadIdx.x/subwarp)%P);
    const unsigned int seqidx = P*blockIdx.x + warpID;
    unsigned int idx = N*subwarp*warpID;

    float Mprev[N] = {0};
    float Iprev[N] = {0};
    unsigned char _hap[N] = {0};

    float M = 0.0f;  
    float y1 = 0.0f; 
    float y2 = 0.0f; 
    float y3 = 0.0f;
    float y4 = 0.0f;
    float value = 0.0f;
    float d0 = 0.0f;
    float tmm = 0.0f;
    float third = 0.0f;
    float one = 0.0f;
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
    third = qualfp32[q0]*0.3333333333333333333f;
    one = 1.0f - qualfp32[q0];
    
    for(unsigned int h = 0; h < N; h++){
        _hap[h] = hap[hidx + N*laneID + h];
        
        _rs[idx + N*laneID + h] = rs[ridx + N*laneID + h];
        _q[idx + N*laneID + h]  = q[ridx + N*laneID + h];
        _i[idx + N*laneID + h]  = i[ridx + N*laneID + h];
        _d[idx + N*laneID + h]  = d[ridx + N*laneID + h];
        
        Mprev[h] = (_hap[h] == rs0) ?  one : third; 
        Mprev[h] *= 0.9f*ldexpf(1.f, 120.f)/(float)(hsize);
        value = value*0.1f + Mprev[h];  
    }

    y3 = __shfl_up_sync(0xffffffff, value, 1);

    y1 = __shfl_up_sync(0xffffffff, y3, 1);
    if (laneID >= 1) y3 = fmaf(y1,powDfp32[N],y3);

    y4 = __shfl_up_sync(0xffffffff, Mprev[N-1], 1);
    y1 = __shfl_up_sync(0xffffffff, Iprev[N-1], 1);
    idx++;

    for(int k = 1; k < rsize; k++){
        rs0 = _rs[idx];
        q0 = _q[idx];
        in = _i[idx];
        d0 = 0.9f*qualfp32[_d[idx - 1]];
        d1 = _d[idx];
        tmm = 1.0f - qualfp32[in] - qualfp32[d1];
        third = qualfp32[q0]*0.3333333333333333333f;
        one = 1.0f - qualfp32[q0];
        
        y2 = (_hap[0] == rs0) ?  one : third;
        y2 *= (y4*tmm + 0.9f*y1 + d0*10.0f*(y3 - y4));
    
        if(laneID == 0) y2 = y3 = 0.0f;
        
        M = value = y2;
        idx++;
        
        for(int h = 1; h < N; h++){
            y2 = (_hap[h] == rs0) ? one : third;
            y2 *= (Mprev[h-1]*tmm + 0.9f*Iprev[h-1] + d0*y3);

            y3 = fmaf(y3,0.1f, Mprev[h-1]);

            Iprev[h-1] = Mprev[h-1]*qualfp32[in] + Iprev[h-1]*0.1f;
            Mprev[h-1] = M;

            M = y2;
            value = fmaf(value,0.1f,y2);        
        }    

        Iprev[N-1] = Mprev[N-1]*qualfp32[in] + Iprev[N-1]*0.1f;
        Mprev[N-1] = M;

        y3 = __shfl_up_sync(0xffffffff, value, 1); 
        y4 = __shfl_up_sync(0xffffffff, Mprev[N-1], 1);
        y1 = __shfl_up_sync(0xffffffff, Iprev[N-1], 1);
        
        y2 = __shfl_up_sync(0xffffffff, y3, 1);
        if (laneID >= 1) y3 = fmaf(y2,powDfp32[N],y3);
    }

    y1 = 0.0f;
    for(int h = 0; h < N; h++) 
        if(N*laneID + h < hsize) y1 += Mprev[h] + Iprev[h]; 

    y2 = __shfl_up_sync(0xffffffff, y1, 1);
    if (laneID >= 1) y1 += y2;
    y2 = __shfl_up_sync(0xffffffff, y1, 2);
    if (laneID >= 2) y1 += y2;
    y2 = __shfl_up_sync(0xffffffff, y1, 4);
    if (laneID >= 4) y1 += y2;
    y2 = __shfl_up_sync(0xffffffff, y1, 8);
    if (laneID >= 8) y1 += y2;
    y2 = __shfl_up_sync(0xffffffff, y1, 16);
    if (laneID >= 16) y1 += y2;

    if(laneID == subwarp - 1) finalsum[seqidx] = y1;
}

template <unsigned int N, unsigned int P> 
__global__ void pairHMM_shared_fp32(unsigned long long* rslen, unsigned long long* haplen, unsigned char* hap, unsigned char* rs, unsigned char* q, unsigned char* i, unsigned char* d, float* finalsum)
{
    alignas(4) __shared__ float _aux[P+1];
    alignas(4) __shared__ float _M[P+1];
    alignas(4) __shared__ float _I[P+1];

    _M[0] = _I[0] = _aux[0] = 0.0f;
    
    const unsigned int laneID = threadIdx.x%32, tid = threadIdx.x, seqidx = blockIdx.x;
    float Mprev[N] = {0};
    float Iprev[N] = {0};
    char _hap[N] = {0};

    float M = 0.0f; 
    float I = 0.0f; 
    float y1 = 0.0f; 
    float y2 = 0.0f; 
    float y3 = 0.0f; 
    float y4 = 0.0f; 
    float value = 0.0f;
    float d0 = 0.0f;
    float tmm = 0.0f;
    float third = 0.0f;
    float one = 0.0f;
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
    third = qualfp32[q0]*0.3333333333333333333f;
    one = 1.0f - qualfp32[q0];
    
    for(unsigned int h = 0; h < N; h++){
        _hap[h] = hap[hidx + N*tid + h];
        Mprev[h] = (_hap[h] == rs0) ?  one : third;
        Mprev[h] *= 0.9f*ldexpf(1.f, 120.f)/(float)(hsize); 
        value = value*0.1f + Mprev[h];  
    }

    y3 = __shfl_up_sync(0xffffffff, value, 1);
    y4 = __shfl_up_sync(0xffffffff, Mprev[N-1], 1);
    y1 = __shfl_up_sync(0xffffffff, Iprev[N-1], 1);

    if(laneID == 0) y3 = 0.0f;
    __syncthreads();
    if(laneID == 31){
        _M[1 + (tid/32)] = M;
        _I[1 + (tid/32)] = I;
        //_aux[1 + (tid/32)] = y3*powDfp32[64] + value;
        _aux[1 + (tid/32)] = value;
    }        
    __syncthreads();

    y3 += _aux[tid/32]*powDfp32[64*laneID];

    for(int k = 1; k < rsize; k++){
        d0 = qualfp32[d1];
        rs0 = rs[ridx + k];
        q0 = q[ridx + k];
        in = i[ridx + k];
        d1 = d[ridx + k];
        tmm = 1.0f - qualfp32[in] - qualfp32[d1];
        third = qualfp32[q0]*0.3333333333333333333f;
        one = 1.0f - qualfp32[q0];
        
        y2 = (_hap[0] == rs0) ?  one : third;
        if(laneID == 0) y2 *= (_M[tid/32]*tmm + 0.9f*(_I[tid/32] + d0*10.0f*(y3 - _M[tid/32])));
        else y2 *= (y4*tmm + 0.9f*(y1 + d0*10.0f*(y3 - y4)));

        M = y2;
        I = Mprev[0]*qualfp32[in] + Iprev[0]*0.1f;

        value = M;
        
        for(int h = 1; h < N; h++){
            y2 = (_hap[h] == rs0) ? one : third;
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

        y3 = __shfl_up_sync(0xffffffff, value, 1);
        y4 = __shfl_up_sync(0xffffffff, Mprev[N-1], 1);
        y1 = __shfl_up_sync(0xffffffff, Iprev[N-1], 1);

        if(laneID == 0) y3 = 0.0f;

        __syncthreads();
        if(laneID == 31){
            _M[1 + (tid/32)] = M;
            _I[1 + (tid/32)] = I;
            //_aux[1 + (tid/32)] = y3*powDfp32[64] + value;
            _aux[1 + (tid/32)] = value;
        }        
        __syncthreads();

        y3 += _aux[tid/32]*powDfp32[64*laneID];
    }
    

    y1 = 0.0f;
    for(int h = 0; h < N; h++)
        if(N*tid + h < hsize) y1 += Mprev[h] + Iprev[h]; 
    
    y2 = __shfl_up_sync(0xffffffff, y1, 1);
    if (laneID >= 1) y1 += y2;
    y2 = __shfl_up_sync(0xffffffff, y1, 2);
    if (laneID >= 2) y1 += y2;
    y2 = __shfl_up_sync(0xffffffff, y1, 4);
    if (laneID >= 4) y1 += y2;
    y2 = __shfl_up_sync(0xffffffff, y1, 8);
    if (laneID >= 8) y1 += y2;
    y2 = __shfl_up_sync(0xffffffff, y1, 16);
    if (laneID >= 16) y1 += y2;

    if(laneID == 31) _aux[1 + (tid/32)] = y1;
    __syncthreads();

    if(tid < 32){
        if(tid < P) y1 = _aux[1 + tid];
        else y1 = 0.0f;

        y2 = __shfl_up_sync(0xffffffff, y1, 1);
        if (laneID >= 1) y1 += y2;
        y2 = __shfl_up_sync(0xffffffff, y1, 2);
        if (laneID >= 2) y1 += y2;
        y2 = __shfl_up_sync(0xffffffff, y1, 4);
        if (laneID >= 4) y1 += y2;
        y2 = __shfl_up_sync(0xffffffff, y1, 8);
        if (laneID >= 8) y1 += y2;
        y2 = __shfl_up_sync(0xffffffff, y1, 16);
        if (laneID >= 16) y1 += y2;

        if(laneID == 31) finalsum[seqidx] = y1;
    }    
}


template <unsigned int N, unsigned int P> 
__global__ void pairHMM_tbcluster_fp32(unsigned long long* rslen, unsigned long long* haplen, unsigned char* hap, unsigned char* rs, unsigned char* q, unsigned char* i, unsigned char* d, float* finalsum)
{
    cg::cluster_group cluster = cg::this_cluster();

    alignas(4) __shared__ float _aux[P+1];
    alignas(4) __shared__ float _M[P+1];
    alignas(4) __shared__ float _I[P+1];

    extern __shared__ float smem2[];

    _M[0] = _I[0] = _aux[0] = 0;

    const unsigned int B = cluster.num_blocks();

    for(int i = threadIdx.x; i < 3 + B; i += blockDim.x) smem2[i] = 0;

    cluster.sync();
    
    const unsigned int laneID = threadIdx.x%32, tid = threadIdx.x, tidB = cluster.thread_rank(), seqidx = (blockIdx.x)/B;
    unsigned int laneID2 = 0, block = cluster.block_rank();

    if(tidB%blockDim.x < 32) laneID2 = tid;
    else laneID2 = 32;

    float Mprev[N] = {0};
    float Iprev[N] = {0};
    char _hap[N] = {0};

    float M = 0.0f; 
    float I = 0.0f; 
    float y1 = 0.0f; 
    float y2 = 0.0f; 
    float y3 = 0.0f; 
    float value = 0.0f;
    float d0 = 0.0f;
    float one = 0.0f;
    float third = 0.0f;
    float tmm = 0.0f;
    char rs0;
    char q0;
    char in;
    char d1;

    float *M_smem = cluster.map_shared_rank(smem2, ((block + 1)%B));
    float *I_smem = cluster.map_shared_rank(smem2 + 1, ((block + 1)%B));
    float *aux_smem = cluster.map_shared_rank(smem2 + 2, ((block + 1)%B));
    float *res_smem = cluster.map_shared_rank(smem2 + 3, 0);
    
    const unsigned long long rsize = -rslen[seqidx] + rslen[seqidx + 1];
    const unsigned long long hsize = -haplen[seqidx] + haplen[seqidx + 1];
    const unsigned long long ridx = rslen[seqidx];
    const unsigned long long hidx = haplen[seqidx];

    rs0 = rs[ridx];
    q0 = q[ridx];
    d1 = d[ridx];
    third = qualfp32[q0]*0.3333333333333333333f;
    one = 1.0f - qualfp32[q0];
    
    for(unsigned int h = 0; h < N; h++){
        _hap[h] = hap[hidx + N*tidB + h];
        Mprev[h] = (_hap[h] == rs0) ?  one : third;
        Mprev[h] *= 0.9f*ldexpf(1.f, 120.f)/(float)(hsize); 
        value = value*0.1f + Mprev[h];  
    }

    y3 = __shfl_up_sync(0xffffffff, value, 1);
    if(laneID == 0) y3 = 0.0f;
    
    if(laneID == 31){
        _M[1 + (tid/32)] = Mprev[N-1];
        _I[1 + (tid/32)] = Iprev[N-1];
        _aux[1 + (tid/32)] = value;
    }
    __syncthreads();
    
    if(tidB%blockDim.x == blockDim.x - 1 && ((block + 1)%B) != 0){
        M_smem[0] = Mprev[N-1];
        I_smem[0] = Iprev[N-1];
        aux_smem[0] = value;
    }

    cluster.sync();

    //y3 += _aux[tid/32]*powDfp32[32*laneID] + smem2[2]*powDfp32[32*laneID2]; 
    
    for(int k = 1; k < rsize; k++){
        rs0 = rs[ridx + k];
        q0 = q[ridx + k];
        in = i[ridx + k];
        d0 = qualfp32[d1];
        d1 = d[ridx + k];
        tmm = 1.0f - qualfp32[in] - qualfp32[d1];
        value = __shfl_up_sync(0xffffffff, Mprev[N-1], 1);
        y1 = __shfl_up_sync(0xffffffff, Iprev[N-1], 1);
        third = qualfp32[q0]*0.3333333333333333333f;
        one = 1.0f - qualfp32[q0];
        
        y2 = (_hap[0] == rs0) ?  one : third;
        if(laneID == 0){
            if(tidB%blockDim.x != 0) y1 = (_M[tid/32]*tmm + 0.9f*(_I[tid/32] + d0*10.0f*(y3 - _M[tid/32])));
            else y1 = (smem2[0]*tmm + 0.9f*(smem2[1] + d0*10.0f*(y3 - smem2[0])));
        }
        else y1 = (value*tmm + 0.9f*(y1 + d0*10.0f*(y3 - value)));
        
        M = y2 * y1;
        I = Iprev[0]*0.1f;

        value = M;
        
        for(int h = 1; h < N; h++){
            y2 = (_hap[h] == rs0) ? one : third;
            y1 = (Mprev[h-1]*tmm + 0.9f*(Iprev[h-1] + d0*y3));

            y3 = y3*0.1f + Mprev[h-1];

            Iprev[h-1] = I + Mprev[h-1]*qualfp32[in];
            Mprev[h-1] = M;

            M = y2 * y1;
            I = Iprev[h]*0.1f;  
            value = value*0.1f + M;       
        }       

        Iprev[N-1] = I + Mprev[N-1]*qualfp32[in];
        Mprev[N-1] = M;

        y3 = __shfl_up_sync(0xffffffff, value, 1);

        if(laneID == 0) y3 = 0.0f;

        __syncthreads();
        if(laneID == 31){
            _M[1 + (tid/32)] = Mprev[N-1];
            _I[1 + (tid/32)] = Iprev[N-1];
            _aux[1 + (tid/32)] = value;
        }
        __syncthreads();

        if(tidB%blockDim.x == blockDim.x - 1 && ((block + 1)%B) != 0){
            M_smem[0] = Mprev[N-1];
            I_smem[0] = Iprev[N-1];
            aux_smem[0] = value;
        }
        cluster.sync();

        //y3 += _aux[tid/32]*powDfp32[32*laneID] + smem2[2]*powDfp32[32*laneID2];
    }
    
    y1 = 0.0f;
    for(int h = 0; h < N; h++)
        if(N*tidB + h < hsize) y1 += Mprev[h] + Iprev[h]; 
    
    y2 = __shfl_up_sync(0xffffffff, y1, 1);
    if (laneID >= 1) y1 += y2;
    y2 = __shfl_up_sync(0xffffffff, y1, 2);
    if (laneID >= 2) y1 += y2;
    y2 = __shfl_up_sync(0xffffffff, y1, 4);
    if (laneID >= 4) y1 += y2;
    y2 = __shfl_up_sync(0xffffffff, y1, 8);
    if (laneID >= 8) y1 += y2;
    y2 = __shfl_up_sync(0xffffffff, y1, 16);
    if (laneID >= 16) y1 += y2;

    if(laneID == 31) _aux[1 + (tid/32)] = y1;
    __syncthreads();

    if(tid < 32){
        if(tid < P) y1 = _aux[1 + tid];
        else y1 = 0.0f;

        y2 = __shfl_up_sync(0xffffffff, y1, 1);
        if (laneID >= 1) y1 += y2;
        y2 = __shfl_up_sync(0xffffffff, y1, 2);
        if (laneID >= 2) y1 += y2;
        y2 = __shfl_up_sync(0xffffffff, y1, 4);
        if (laneID >= 4) y1 += y2;
        y2 = __shfl_up_sync(0xffffffff, y1, 8);
        if (laneID >= 8) y1 += y2;
        y2 = __shfl_up_sync(0xffffffff, y1, 16);
        if (laneID >= 16) y1 += y2;

        if(laneID == 31){
            if((block%B) == 0) smem2[3 + (block%B)] = y1;
            else res_smem[(block%B)] = y1;
        } 
    }    

    cluster.sync();

    if((block%B) == 0 && tid < 32){
        if(tid < B) y1 = smem2[3 + tid];
        else y1 = 0.0f;

        y2 = __shfl_up_sync(0xffffffff, y1, 1);
        if (laneID >= 1) y1 += y2;
        y2 = __shfl_up_sync(0xffffffff, y1, 2);
        if (laneID >= 2) y1 += y2;
        y2 = __shfl_up_sync(0xffffffff, y1, 4);
        if (laneID >= 4) y1 += y2;
        y2 = __shfl_up_sync(0xffffffff, y1, 8);
        if (laneID >= 8) y1 += y2;
        y2 = __shfl_up_sync(0xffffffff, y1, 16);
        if (laneID >= 16) y1 += y2;

        if(laneID == 31) finalsum[seqidx] = y1;
    }
}

float pairhmm_benchmark_fp32(unsigned long long* rlenGPU, unsigned long long* hlenGPU, unsigned char* hapGPU, unsigned char* readGPU, unsigned char* qGPU, unsigned char* iGPU, unsigned char* dGPU, float* sumGPU, float* sumCPU, int maxsize, int seq_size){
    float total_time = 0;
    float g_ctxf = log10f(ldexpf(1.f, 120.f));
    float powvalues[128] = {0};

    for(int i = 0; i < 128; i++){
        powvalues[i] = powf(10.0, -(float)i/10.0);
    }

    cudaMemcpyToSymbol(qualfp32, powvalues, sizeof(powvalues));

    float powd[2048] = {0};

    for(int i = 0; i < 2048; i++){
        powd[i] = powf(10.0, -(float)(i));
    }

    cudaMemcpyToSymbol(powDfp32, powd, sizeof(powd));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if(maxsize == 32){
        cudaEventRecord(start);
        pairHMM_register_fp32_subwarp<32,32><<<seq_size/32,32>>>(rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, sumGPU);
        cudaEventRecord(stop);
    }
    else if(maxsize == 64){
        cudaEventRecord(start);
        pairHMM_register_fp32_subwarp<32,16><<<seq_size/16,32>>>(rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, sumGPU);
        cudaEventRecord(stop);
    }
    else if(maxsize == 128){
        cudaEventRecord(start);
        pairHMM_register_fp32_subwarp<32,8><<<seq_size/8,32>>>(rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, sumGPU);
        cudaEventRecord(stop);
    }
    else if(maxsize == 256){
        cudaEventRecord(start);
        pairHMM_register_fp32_subwarp<32,4><<<seq_size/4,32>>>(rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, sumGPU);
        cudaEventRecord(stop);
    }
    else if(maxsize == 512){
        cudaEventRecord(start);
        pairHMM_register_fp32_subwarp<32,2><<<seq_size/2,32>>>(rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, sumGPU);
        cudaEventRecord(stop);
    }
    else if(maxsize == 1024){
        cudaEventRecord(start);
        pairHMM_register_fp32_subwarp<32,1><<<seq_size,32>>>(rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, sumGPU);
        cudaEventRecord(stop);
    }
    else if(maxsize == 2048){
        cudaEventRecord(start);
        pairHMM_register_fp32_subwarp<64,1><<<seq_size,32>>>(rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, sumGPU);
        cudaEventRecord(stop);
    }
    else if(maxsize == 4096){
        cudaEventRecord(start);
        pairHMM_shared_fp32<64,2><<<seq_size,64>>>(rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, sumGPU);
        cudaEventRecord(stop);
    }
    else if(maxsize == 8192){
        cudaEventRecord(start);
        pairHMM_shared_fp32<64,4><<<seq_size,128>>>(rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, sumGPU);
        cudaEventRecord(stop);
    }
    else if(maxsize == 16384){
        cudaEventRecord(start);
        pairHMM_shared_fp32<64,8><<<seq_size,256>>>(rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, sumGPU);
        cudaEventRecord(stop);
    }    
    else if(maxsize == 32768){
        cudaLaunchConfig_t config = {0};
        config.gridDim = 2*seq_size;
        config.blockDim = 32*8;

        int cluster_size = 2;

        config.dynamicSmemBytes = (3 + cluster_size) * sizeof(float);

        cudaFuncSetAttribute((void *)pairHMM_tbcluster_fp32<64, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes);

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 2;
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;

        config.numAttrs = 1;
        config.attrs = attribute;
        
        cudaEventRecord(start);
        cudaLaunchKernelEx(&config, pairHMM_tbcluster_fp32<64, 8>, rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, sumGPU);
        cudaEventRecord(stop);
    }
    else if(maxsize == 65536){
        cudaLaunchConfig_t config = {0};
        config.gridDim = 4*seq_size;
        config.blockDim = 32*8;

        int cluster_size = 4;

        config.dynamicSmemBytes = (3 + cluster_size) * sizeof(float);

        cudaFuncSetAttribute((void *)pairHMM_tbcluster_fp32<64, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes);

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 4;
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;

        config.numAttrs = 1;
        config.attrs = attribute;
        
        cudaEventRecord(start);
        cudaLaunchKernelEx(&config, pairHMM_tbcluster_fp32<64, 8>, rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, sumGPU);
        cudaEventRecord(stop);
    }
    else if(maxsize == 131072){
        cudaLaunchConfig_t config = {0};
        config.gridDim = 8*seq_size;
        config.blockDim = 32*8;

        int cluster_size = 8;

        config.dynamicSmemBytes = (3 + cluster_size) * sizeof(float);

        cudaFuncSetAttribute((void *)pairHMM_tbcluster_fp32<64, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes);

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 8;
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;

        config.numAttrs = 1;
        config.attrs = attribute;
        
        cudaEventRecord(start);
        cudaLaunchKernelEx(&config, pairHMM_tbcluster_fp32<64, 8>, rlenGPU, hlenGPU, hapGPU, readGPU, qGPU, iGPU, dGPU, sumGPU);
        cudaEventRecord(stop);
    }
    
    cudaMemcpy(sumCPU, sumGPU, seq_size*sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&total_time, start, stop);

    return total_time;
}
