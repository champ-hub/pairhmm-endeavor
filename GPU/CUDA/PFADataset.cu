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
template <unsigned int N, unsigned int P> 
__global__ void pairHMM_register_fp64_subwarp_dataset(unsigned long long* rslen, unsigned long long* haplen, unsigned char* hap, unsigned char* rs, unsigned char* q, unsigned char* i, unsigned char* d, double* finalsum, unsigned int* blockDist, unsigned int seq, unsigned int blockCounts)
{
    alignas(N) __shared__ char _rs[32*N];
    alignas(N) __shared__ char _q[32*N];
    alignas(N) __shared__ char _i[32*N];
    alignas(N) __shared__ char _d[32*N];  
    
    const unsigned int subwarp = 32/P;
    const unsigned int laneID = threadIdx.x%subwarp, warpID = ((threadIdx.x/subwarp)%P);
    const unsigned int bidx = (P*blockIdx.x + warpID)%blockCounts;
    const unsigned int seqidx = blockDist[seq + bidx], idx = N*subwarp*warpID;

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
    
    const unsigned int laneID = threadIdx.x%32, tid = threadIdx.x, seqidx = blockDist[seq + blockIdx.x];
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
__global__ void pairHMM_tbcluster_fp64_dataset(unsigned long long* rslen, unsigned long long* haplen, unsigned char* hap, unsigned char* rs, unsigned char* q, unsigned char* i, unsigned char* d, double* finalsum, unsigned int* blockDist, unsigned seq)
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

    const unsigned int laneID = threadIdx.x%32, tid = threadIdx.x, tidB = cluster.thread_rank(), seqidx = blockDist[seq + (blockIdx.x)/B];
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
        Mprev[h] = (_hap[h] == rs0) ?  (1 - qualfp64[q0 - 33]) : qualfp64[q0 - 33]*0.3333333333333333333;
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
        d0 = qualfp64[d1 - 33];
        d1 = d[ridx + k];
        tmm = 1 - qualfp64[in - 33] - qualfp64[d1 - 33];
        value = __shfl_up_sync(0xffffffff, Mprev[N-1], 1);
        y1 = __shfl_up_sync(0xffffffff, Iprev[N-1], 1);
        
        y2 = (_hap[0] == rs0) ?  (1 - qualfp64[q0 - 33]) : qualfp64[q0 - 33]*0.3333333333333333333;
        if(laneID == 0){
            if(tidB%blockDim.x != 0) y1 = (_M[tid/32]*tmm + 0.9*(_I[tid/32] + d0*10*(y3 - _M[tid/32])));
            else y1 = (smem[0]*tmm + 0.9*(smem[1] + d0*10*(y3 - smem[0])));
        }
        else y1 = (value*tmm + 0.9*(y1 + d0*10*(y3 - value)));
        
        M = y2 * y1;
        I = Iprev[0]*0.1;

        value = M;
        
        for(int h = 1; h < N; h++){
            y2 = (_hap[h] == rs0) ? (1 - qualfp64[q0 - 33]) : qualfp64[q0 - 33]*0.3333333333333333333;
            y1 = (Mprev[h-1]*tmm + 0.9*(Iprev[h-1] + d0*y3));

            y3 = y3*0.1 + Mprev[h-1];

            Iprev[h-1] = I + Mprev[h-1]*qualfp64[in - 33];
            Mprev[h-1] = M;

            M = y2 * y1;
            I = Iprev[h]*0.1;  
            value = value*0.1 + M;       
        }       

        Iprev[N-1] = I + Mprev[N-1]*qualfp64[in - 33];
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

float runKernelsFP64(unsigned long long* rslen, unsigned long long* haplen, unsigned char* hap, unsigned char* rs, unsigned char* q, unsigned char* in, unsigned char* d, double* sumGPU, double* sumCPU, unsigned int* blockDist, unsigned int* blockCounts, FILE* fpout, int maxsize, unsigned int seq_size){
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

    cudaEventRecord(start);
    if(blockCounts[0] != 0) pairHMM_register_fp64_subwarp_dataset<32,32><<<1 + blockCounts[0]/32,32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 0, blockCounts[0]); //32
    if(blockCounts[1] != 0) pairHMM_register_fp64_subwarp_dataset<32,16><<<1 + blockCounts[1]/16,32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, seq_size, blockCounts[1]); //64
    if(blockCounts[2] != 0) pairHMM_register_fp64_subwarp_dataset<32,10><<<1 + blockCounts[2]/10,32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 2*seq_size, blockCounts[2]); //96
    if(blockCounts[3] != 0) pairHMM_register_fp64_subwarp_dataset<32,8><<<1 + blockCounts[3]/8,32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 3*seq_size, blockCounts[3]); //128
    if(blockCounts[4] != 0) pairHMM_register_fp64_subwarp_dataset<32,6><<<1 + blockCounts[4]/6,32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 4*seq_size, blockCounts[4]); //160
    if(blockCounts[5] != 0) pairHMM_register_fp64_subwarp_dataset<32,5><<<1 + blockCounts[5]/5,32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 5*seq_size, blockCounts[5]); //192
    if(blockCounts[6] != 0) pairHMM_register_fp64_subwarp_dataset<32,4><<<1 + blockCounts[6]/4,32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 6*seq_size, blockCounts[6]); //224
    if(blockCounts[7] != 0) pairHMM_register_fp64_subwarp_dataset<32,4><<<1 + blockCounts[7]/4,32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 7*seq_size, blockCounts[7]); //256
    if(blockCounts[8] != 0) pairHMM_register_fp64_subwarp_dataset<32,3><<<1 + blockCounts[8]/3,32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 8*seq_size, blockCounts[8]); //288
    if(blockCounts[9] != 0) pairHMM_register_fp64_subwarp_dataset<32,3><<<1 + blockCounts[9]/3,32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 9*seq_size, blockCounts[9]); //320
    for(int i = 10; i < 16; i++){
        if(blockCounts[i] != 0) pairHMM_register_fp64_subwarp_dataset<32,2><<<1 + blockCounts[i]/2,32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, i*seq_size, blockCounts[i]);
    }
    for(int i = 16; i < 32; i++){
        if(blockCounts[i] != 0) pairHMM_register_fp64_subwarp_dataset<32,1><<<blockCounts[i],32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, i*seq_size, blockCounts[i]);
    }
    if(blockCounts[32] != 0) pairHMM_shared_fp64_dataset<32,2><<<blockCounts[32],64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 32*seq_size);
    if(blockCounts[33] != 0) pairHMM_shared_fp64_dataset<32,4><<<blockCounts[33],128>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 33*seq_size);
    if(blockCounts[34] != 0) pairHMM_shared_fp64_dataset<32,8><<<blockCounts[34],256>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 34*seq_size);
    if(blockCounts[35] != 0){
        cudaLaunchConfig_t config = {0};
        config.gridDim = 2*blockCounts[35];
        config.blockDim = 32*8;

        int cluster_size = 2;

        config.dynamicSmemBytes = (3 + cluster_size) * sizeof(double);

        cudaFuncSetAttribute((void *)pairHMM_tbcluster_fp64_dataset<32, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes);

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 2;
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;

        config.numAttrs = 1;
        config.attrs = attribute;

        cudaLaunchKernelEx(&config, pairHMM_tbcluster_fp64_dataset<32, 8>, rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 35*seq_size);
    }
    if(blockCounts[36] != 0){
        cudaLaunchConfig_t config = {0};
        config.gridDim = 4*blockCounts[36];
        config.blockDim = 32*8;

        int cluster_size = 4;

        config.dynamicSmemBytes = (3 + cluster_size) * sizeof(double);

        cudaFuncSetAttribute((void *)pairHMM_tbcluster_fp64_dataset<32, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes);

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 4;
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;

        config.numAttrs = 1;
        config.attrs = attribute;

        cudaLaunchKernelEx(&config, pairHMM_tbcluster_fp64_dataset<32, 8>, rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 36*seq_size);
    }
    if(blockCounts[37] != 0){
        cudaLaunchConfig_t config = {0};
        config.gridDim = 8*blockCounts[37];
        config.blockDim = 32*8;

        int cluster_size = 8;

        config.dynamicSmemBytes = (3 + cluster_size) * sizeof(double);

        cudaFuncSetAttribute((void *)pairHMM_tbcluster_fp64_dataset<32, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes);

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 8;
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;

        config.numAttrs = 1;
        config.attrs = attribute;

        cudaLaunchKernelEx(&config, pairHMM_tbcluster_fp64_dataset<32, 8>, rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 37*seq_size);
    }
    if(blockCounts[38] != 0){
        cudaLaunchConfig_t config = {0};
        config.gridDim = 8*blockCounts[38];
        config.blockDim = 32*8;

        int cluster_size = 8;

        config.dynamicSmemBytes = (3 + cluster_size) * sizeof(double);

        cudaFuncSetAttribute((void *)pairHMM_tbcluster_fp64_dataset<64, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes);

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 8;
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;

        config.numAttrs = 1;
        config.attrs = attribute;

        cudaEventRecord(start);
        cudaLaunchKernelEx(&config, pairHMM_tbcluster_fp64_dataset<64, 8>, rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 38*seq_size);
        cudaEventRecord(stop);
    }
    cudaEventRecord(stop);
    cudaMemcpy(sumCPU, sumGPU, seq_size*sizeof(double), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&total_time, start, stop);


    return total_time;
}



//FP32 FUNCTIONS
template <unsigned int N, unsigned int P> 
__global__ void pairHMM_register_fp32_subwarp_dataset(unsigned long long* rslen, unsigned long long* haplen, unsigned char* hap, unsigned char* rs, unsigned char* q, unsigned char* i, unsigned char* d, float* finalsum, unsigned int* blockDist, unsigned int seq, unsigned int blockCounts)
{
    alignas(32) __shared__ char _rs[32*N];
    alignas(32) __shared__ char _q[32*N];
    alignas(32) __shared__ char _i[32*N];
    alignas(32) __shared__ char _d[32*N];   
    
    const unsigned int subwarp = 32/P;
    const unsigned int laneID = threadIdx.x%subwarp, warpID = ((threadIdx.x/subwarp)%P);
    const unsigned int bidx = (P*blockIdx.x + warpID)%blockCounts;
    const unsigned int seqidx = blockDist[seq + bidx];
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
    char rs0;
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

    if(laneID == subwarp - 1){
        if(blockIdx.x != blockDim.x - 1) finalsum[seqidx] = y1;
        else if(blockIdx.x == blockDim.x - 1 && bidx > P) finalsum[seqidx] = y1;
    }
    
}


template <unsigned int N, unsigned int P> 
__global__ void pairHMM_shared_fp32_dataset(unsigned long long* rslen, unsigned long long* haplen, unsigned char* hap, unsigned char* rs, unsigned char* q, unsigned char* i, unsigned char* d, float* finalsum, unsigned int* blockDist, unsigned int seq)
{
    alignas(4) __shared__ float _aux[P+1];
    alignas(4) __shared__ float _M[P+1];
    alignas(4) __shared__ float _I[P+1];

    _M[0] = _I[0] = _aux[0] = 0.0f;
    
    const unsigned int laneID = threadIdx.x%32, tid = threadIdx.x, seqidx = blockDist[seq + blockIdx.x];
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
    float tmm = 0.0f;
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
    
    for(unsigned int h = 0; h < N; h++){
        _hap[h] = hap[hidx + N*tid + h];
        Mprev[h] = (_hap[h] == rs0) ?  (1.0f - qualfp32[q0 - 33]) : qualfp32[q0 - 33]*0.3333333333333333333f;
        Mprev[h] *= 0.9f*ldexpf(1.f, 120.f)/(float)(hsize); 
        value = value*0.1f + Mprev[h];  
    }

    y3 = __shfl_up_sync(0xffffffff, value, 1);
    if(laneID == 0) y3 = 0.0f;
    else if(laneID == 31){
        _M[1 + (tid/32)] = Mprev[N-1];
        _I[1 + (tid/32)] = Iprev[N-1];
    }

    if(laneID == 31) _aux[1 + (tid/32)] = y3*powDfp32[32] + value;
    __syncthreads();

    y3 += _aux[tid/32]*powDfp32[32*laneID];

    for(int k = 1; k < rsize; k++){
        rs0 = rs[ridx + k];
        q0 = q[ridx + k];
        in = i[ridx + k];
        d0 = qualfp32[d1 - 33];
        d1 = d[ridx + k];
        tmm = 1.0f - qualfp32[in - 33] - qualfp32[d1 - 33];
        value = __shfl_up_sync(0xffffffff, Mprev[N-1], 1);
        y1 = __shfl_up_sync(0xffffffff, Iprev[N-1], 1);
        
        y2 = (_hap[0] == rs0) ?  (1.0f - qualfp32[q0 - 33]) : qualfp32[q0 - 33]*0.3333333333333333333f;
        if(laneID == 0) y2 *= (_M[tid/32]*tmm + 0.9f*(_I[tid/32] + d0*10.0f*(y3 - _M[tid/32])));
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

        y3 = __shfl_up_sync(0xffffffff, value, 1);

        __syncthreads();
        if(laneID == 0) y3 = 0.0f;
        else if(laneID == 31){
            _M[1 + (tid/32)] = M;
            _I[1 + (tid/32)] = I;
        }        

        if(laneID == 31) _aux[1 + (tid/32)] = y3*powDfp32[32] + value;
        __syncthreads();

        y3 += _aux[tid/32]*powDfp32[32*laneID];
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
__global__ void pairHMM_tbcluster_fp32_dataset(unsigned long long* rslen, unsigned long long* haplen, unsigned char* hap, unsigned char* rs, unsigned char* q, unsigned char* i, unsigned char* d, float* finalsum, unsigned int* blockDist, unsigned seq)
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
    
    const unsigned int laneID = threadIdx.x%32, tid = threadIdx.x, tidB = cluster.thread_rank(), seqidx = blockDist[seq + (blockIdx.x)/B];
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
    
    for(unsigned int h = 0; h < N; h++){
        _hap[h] = hap[hidx + N*tidB + h];
        Mprev[h] = (_hap[h] == rs0) ?  (1.0f - qualfp32[q0 - 33]) : qualfp32[q0 - 33]*0.3333333333333333333f;
        Mprev[h] *= 0.9f*ldexpf(1.f, 120.f)/(float)(hsize); 
        value = value*0.1f + Mprev[h];  
    }

    y3 = __shfl_up_sync(0xffffffff, value, 1);
    if(laneID == 0) y3 = 0.0f;
    
    if(laneID == 31){
        _M[1 + (tid/32)] = Mprev[N-1];
        _I[1 + (tid/32)] = Iprev[N-1];
    }
    
    if(tidB%blockDim.x == blockDim.x - 1 && ((block + 1)%B) != 0){
        M_smem[0] = Mprev[N-1];
        I_smem[0] = Iprev[N-1];
    }

    if(laneID == 31) _aux[1 + (tid/32)] = y3*powDfp32[32] + value;
    __syncthreads();

    if(tidB%blockDim.x == blockDim.x - 1 && (block + 1)%B != 0) aux_smem[0] = y3*powDfp32[32] + value;
    cluster.sync();

    y3 += _aux[tid/32]*powDfp32[32*laneID] + smem2[2]*powDfp32[32*laneID2]; 
    
    for(int k = 1; k < rsize; k++){
        rs0 = rs[ridx + k];
        q0 = q[ridx + k];
        in = i[ridx + k];
        d0 = qualfp32[d1 - 33];
        d1 = d[ridx + k];
        tmm = 1.0f - qualfp32[in - 33] - qualfp32[d1 - 33];
        value = __shfl_up_sync(0xffffffff, Mprev[N-1], 1);
        y1 = __shfl_up_sync(0xffffffff, Iprev[N-1], 1);
        
        y2 = (_hap[0] == rs0) ?  (1.0f - qualfp32[q0 - 33]) : qualfp32[q0 - 33]*0.3333333333333333333f;
        if(laneID == 0){
            if(tidB%blockDim.x != 0) y1 = (_M[tid/32]*tmm + 0.9f*(_I[tid/32] + d0*10.0f*(y3 - _M[tid/32])));
            else y1 = (smem2[0]*tmm + 0.9f*(smem2[1] + d0*10.0f*(y3 - smem2[0])));
        }
        else y1 = (value*tmm + 0.9f*(y1 + d0*10.0f*(y3 - value)));
        
        M = y2 * y1;
        I = Iprev[0]*0.1f;

        value = M;
        
        for(int h = 1; h < N; h++){
            y2 = (_hap[h] == rs0) ? (1.0f - qualfp32[q0 - 33]) : qualfp32[q0 - 33]*0.3333333333333333333f;
            y1 = (Mprev[h-1]*tmm + 0.9f*(Iprev[h-1] + d0*y3));

            y3 = y3*0.1f + Mprev[h-1];

            Iprev[h-1] = I + Mprev[h-1]*qualfp32[in - 33];
            Mprev[h-1] = M;

            M = y2 * y1;
            I = Iprev[h]*0.1f;  
            value = value*0.1f + M;       
        }       

        Iprev[N-1] = I + Mprev[N-1]*qualfp32[in - 33];
        Mprev[N-1] = M;

        y3 = __shfl_up_sync(0xffffffff, value, 1);

        __syncthreads();
        if(laneID == 0) y3 = 0.0f;
        if(laneID == 31){
            _M[1 + (tid/32)] = Mprev[N-1];
            _I[1 + (tid/32)] = Iprev[N-1];
        }
        if(tidB%blockDim.x == blockDim.x - 1 && ((block + 1)%B) != 0){
            M_smem[0] = Mprev[N-1];
            I_smem[0] = Iprev[N-1];
        }

        if(laneID == 31) _aux[1 + (tid/32)] = y3*powDfp32[32] + value;
        __syncthreads();

        if(tidB%blockDim.x == blockDim.x - 1 && (block + 1)%B != 0) aux_smem[0] = y3*powDfp32[32] + value;
        cluster.sync();

        y3 += _aux[tid/32]*powDfp32[32*laneID] + smem2[2]*powDfp32[32*laneID2];
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


float runKernelsFP32(unsigned long long* rslen, unsigned long long* haplen, unsigned char* hap, unsigned char* rs, unsigned char* q, unsigned char* in, unsigned char* d, float* sumGPU, float* sumCPU, unsigned int* blockDist, unsigned int* blockCounts, FILE* fpout, int maxsize, unsigned int seq_size){
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

    cudaEventRecord(start);
    
    if(blockCounts[0] != 0) pairHMM_register_fp32_subwarp_dataset<32,32><<<1 + blockCounts[0]/32,32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 0, blockCounts[0]); //32
    if(blockCounts[1] != 0) pairHMM_register_fp32_subwarp_dataset<32,16><<<1 + blockCounts[1]/16,32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, seq_size, blockCounts[1]); //64
    if(blockCounts[2] != 0) pairHMM_register_fp32_subwarp_dataset<24,8><<<1 + blockCounts[2]/8,32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 2*seq_size, blockCounts[2]); //96
    if(blockCounts[3] != 0) pairHMM_register_fp32_subwarp_dataset<32,8><<<1 + blockCounts[3]/8,32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 3*seq_size, blockCounts[3]); //128
    if(blockCounts[4] != 0) pairHMM_register_fp32_subwarp_dataset<20,4><<<1 + blockCounts[4]/4,32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 4*seq_size, blockCounts[4]); //160
    if(blockCounts[5] != 0) pairHMM_register_fp32_subwarp_dataset<24,4><<<1 + blockCounts[5]/4,32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 5*seq_size, blockCounts[5]); //192
    if(blockCounts[6] != 0) pairHMM_register_fp32_subwarp_dataset<28,4><<<1 + blockCounts[6]/4,32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 6*seq_size, blockCounts[6]); //224
    if(blockCounts[7] != 0) pairHMM_register_fp32_subwarp_dataset<32,4><<<1 + blockCounts[7]/4,32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 7*seq_size, blockCounts[7]); //256
    if(blockCounts[8] != 0) pairHMM_register_fp32_subwarp_dataset<18,2><<<1 + blockCounts[8]/2,32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 8*seq_size, blockCounts[8]); //288
    if(blockCounts[9] != 0) pairHMM_register_fp32_subwarp_dataset<20,2><<<1 + blockCounts[9]/2,32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 9*seq_size, blockCounts[9]); //320
    if(blockCounts[10] != 0) pairHMM_register_fp32_subwarp_dataset<22,2><<<1 + blockCounts[10]/2,32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 10*seq_size, blockCounts[10]); //352
    if(blockCounts[11] != 0) pairHMM_register_fp32_subwarp_dataset<24,2><<<1 + blockCounts[11]/2,32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 11*seq_size, blockCounts[11]); //384
    if(blockCounts[12] != 0) pairHMM_register_fp32_subwarp_dataset<26,2><<<1 + blockCounts[12]/2,32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 12*seq_size, blockCounts[12]); //416
    if(blockCounts[13] != 0) pairHMM_register_fp32_subwarp_dataset<28,2><<<1 + blockCounts[13]/2,32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 13*seq_size, blockCounts[13]); //448
    if(blockCounts[14] != 0) pairHMM_register_fp32_subwarp_dataset<30,2><<<1 + blockCounts[14]/2,32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 14*seq_size, blockCounts[14]); //480
    if(blockCounts[15] != 0) pairHMM_register_fp32_subwarp_dataset<32,2><<<1 + blockCounts[15]/2,32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 15*seq_size, blockCounts[15]); //512

    if(blockCounts[16] != 0) pairHMM_register_fp32_subwarp_dataset<17,1><<<blockCounts[16],32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 16*seq_size, blockCounts[16]);
    if(blockCounts[17] != 0) pairHMM_register_fp32_subwarp_dataset<18,1><<<blockCounts[17],32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 17*seq_size, blockCounts[17]);
    if(blockCounts[18] != 0) pairHMM_register_fp32_subwarp_dataset<19,1><<<blockCounts[18],32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 18*seq_size, blockCounts[18]);
    if(blockCounts[19] != 0) pairHMM_register_fp32_subwarp_dataset<20,1><<<blockCounts[19],32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 19*seq_size, blockCounts[19]);
    if(blockCounts[20] != 0) pairHMM_register_fp32_subwarp_dataset<21,1><<<blockCounts[20],32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 20*seq_size, blockCounts[20]);
    if(blockCounts[21] != 0) pairHMM_register_fp32_subwarp_dataset<22,1><<<blockCounts[21],32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 21*seq_size, blockCounts[21]);
    if(blockCounts[22] != 0) pairHMM_register_fp32_subwarp_dataset<23,1><<<blockCounts[22],32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 22*seq_size, blockCounts[22]);
    if(blockCounts[23] != 0) pairHMM_register_fp32_subwarp_dataset<24,1><<<blockCounts[23],32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 23*seq_size, blockCounts[23]);
    if(blockCounts[24] != 0) pairHMM_register_fp32_subwarp_dataset<25,1><<<blockCounts[24],32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 24*seq_size, blockCounts[24]);
    if(blockCounts[25] != 0) pairHMM_register_fp32_subwarp_dataset<26,1><<<blockCounts[25],32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 25*seq_size, blockCounts[25]);
    if(blockCounts[26] != 0) pairHMM_register_fp32_subwarp_dataset<27,1><<<blockCounts[26],32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 26*seq_size, blockCounts[26]);
    if(blockCounts[27] != 0) pairHMM_register_fp32_subwarp_dataset<28,1><<<blockCounts[27],32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 27*seq_size, blockCounts[27]);
    if(blockCounts[28] != 0) pairHMM_register_fp32_subwarp_dataset<29,1><<<blockCounts[28],32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 28*seq_size, blockCounts[28]);
    if(blockCounts[29] != 0) pairHMM_register_fp32_subwarp_dataset<30,1><<<blockCounts[29],32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 29*seq_size, blockCounts[29]);
    if(blockCounts[30] != 0) pairHMM_register_fp32_subwarp_dataset<31,1><<<blockCounts[30],32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 30*seq_size, blockCounts[30]);
    if(blockCounts[31] != 0) pairHMM_register_fp32_subwarp_dataset<32,1><<<blockCounts[31],32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 31*seq_size, blockCounts[31]);

    if(blockCounts[32] != 0) pairHMM_register_fp32_subwarp_dataset<64,1><<<blockCounts[32],32>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 32*seq_size, blockCounts[32]);
    if(blockCounts[33] != 0) pairHMM_shared_fp32_dataset<64,2><<<blockCounts[33],64>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 33*seq_size);
    if(blockCounts[34] != 0) pairHMM_shared_fp32_dataset<64,4><<<blockCounts[34],128>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 34*seq_size);
    if(blockCounts[35] != 0) pairHMM_shared_fp32_dataset<64,8><<<blockCounts[35],256>>>(rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 35*seq_size);
    if(blockCounts[36] != 0){
        cudaLaunchConfig_t config = {0};
        config.gridDim = 4*blockCounts[36];
        config.blockDim = 32*8;

        int cluster_size = 4;

        config.dynamicSmemBytes = (3 + cluster_size) * sizeof(double);

        cudaFuncSetAttribute((void *)pairHMM_tbcluster_fp32_dataset<32, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes);

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 4;
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;

        config.numAttrs = 1;
        config.attrs = attribute;

        cudaLaunchKernelEx(&config, pairHMM_tbcluster_fp32_dataset<32, 8>, rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 36*seq_size);
    }
    if(blockCounts[37] != 0){
        cudaLaunchConfig_t config = {0};
        config.gridDim = 8*blockCounts[37];
        config.blockDim = 32*8;

        int cluster_size = 8;

        config.dynamicSmemBytes = (3 + cluster_size) * sizeof(double);

        cudaFuncSetAttribute((void *)pairHMM_tbcluster_fp32_dataset<32, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes);

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 8;
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;

        config.numAttrs = 1;
        config.attrs = attribute;

        cudaLaunchKernelEx(&config, pairHMM_tbcluster_fp32_dataset<32, 8>, rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 37*seq_size);
    }
    if(blockCounts[38] != 0){
        cudaLaunchConfig_t config = {0};
        config.gridDim = 8*blockCounts[38];
        config.blockDim = 32*8;

        int cluster_size = 8;

        config.dynamicSmemBytes = (3 + cluster_size) * sizeof(double);

        cudaFuncSetAttribute((void *)pairHMM_tbcluster_fp32_dataset<64, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes);

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 8;
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;

        config.numAttrs = 1;
        config.attrs = attribute;

        cudaLaunchKernelEx(&config, pairHMM_tbcluster_fp32_dataset<64, 8>, rslen, haplen, hap, rs, q, in, d, sumGPU, blockDist, 38*seq_size);
    }
    cudaEventRecord(stop);
    cudaMemcpy(sumCPU, sumGPU, seq_size*sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&total_time, start, stop);

    return total_time;
}
