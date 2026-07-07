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


#define NSEC_IN_SEC 1000000000.0;

using namespace std;

//AVX
template <int N> 
double pairhmm_fp64(unsigned long long* rslen, unsigned long long* haplen, unsigned long long* hap, unsigned long long* read, unsigned long long* q, unsigned long long* i, unsigned long long* d, double* qualfp64, double* finalsum, int seqsize){
    double exec_time = 0;

    omp_set_num_threads(min(seqsize/4, 192));
    
    exec_time = -omp_get_wtime();
    #pragma omp parallel
    { 
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        for(int seqidx = 4*tid; seqidx < seqsize; seqidx += 4*nthreads){
            
            __m256d Mprev[N];
            __m256d Iprev[N];
            __m256i _hap[N];

            __m256d M = _mm256_set1_pd(0);  
            __m256d y1 = _mm256_set1_pd(0); 
            __m256d y2 = _mm256_set1_pd(0); 
            __m256d y3 = _mm256_set1_pd(0);
            __m256d one;
            __m256d one0 = _mm256_set1_pd(0.1);
            __m256d nine = _mm256_set1_pd(0.9);
            __m256d third;
            __m256d d0 = _mm256_set1_pd(0);
            __m256d tmm = _mm256_set1_pd(0);
            __m256d sigma_aux = _mm256_set1_pd(0);
            __m256d qual = _mm256_set1_pd(0);
            __m256d qualin = _mm256_set1_pd(0);
            __m256d mask = _mm256_set1_pd(0);
            __m256i rs0;
            __m256i q0;
            __m256i in;
            __m256i d1;
            
            const unsigned long long rsize = -rslen[seqidx] + rslen[seqidx + 1];
            const unsigned long long hsize = -haplen[seqidx] + haplen[seqidx + 1];
            const unsigned long long ridx = rslen[seqidx];
            const unsigned long long hidx = haplen[seqidx];
            
            for(int h = 0; h < N; h++) _hap[h] = _mm256_load_si256((__m256i const*)(hap + hidx + h*4));
            
            rs0 = _mm256_load_si256((__m256i const*)(read + ridx));
            q0 = _mm256_load_si256((__m256i const*)(q + ridx));
            d1 = _mm256_load_si256((__m256i const*)(d + ridx));
            qual = _mm256_i64gather_pd(qualfp64, q0, 8);
            third = _mm256_mul_pd(_mm256_set1_pd(0.3333333333333333333333333333333333),qual);
            one = _mm256_sub_pd(_mm256_set1_pd(1),qual);
            
            for(int h = 0; h < N; h++){
                mask = _mm256_castsi256_pd(_mm256_cmpeq_epi64(rs0, _hap[h]));
                sigma_aux = _mm256_blendv_pd(third, one, mask);
                Mprev[h] = _mm256_mul_pd(sigma_aux, _mm256_mul_pd(nine, _mm256_div_pd(_mm256_set1_pd(ldexp(1.0, 1020.0)),_mm256_set1_pd(hsize))));
                Iprev[h] = _mm256_set1_pd(0);
            }  

            for(int k = 1; k < rsize; k++){
                rs0 = _mm256_load_si256((__m256i const*)(read + ridx + k*4));
                q0 = _mm256_load_si256((__m256i const*)(q + ridx + k*4));
                in = _mm256_load_si256((__m256i const*)(i + ridx + k*4));
                d0 = _mm256_mul_pd(nine,_mm256_i64gather_pd(qualfp64, d1, 8));
                d1 = _mm256_load_si256((__m256i const*)(d + ridx + k*4));

                mask = _mm256_castsi256_pd(_mm256_cmpeq_epi64(rs0, _hap[0]));
                qual = _mm256_i64gather_pd(qualfp64, q0, 8);
                qualin = _mm256_i64gather_pd(qualfp64, in, 8);
                third = _mm256_mul_pd(_mm256_set1_pd(0.3333333333333333333333333333333333),qual);
                one = _mm256_sub_pd(_mm256_set1_pd(1),qual);
                sigma_aux = _mm256_blendv_pd(third, one, mask);
                tmm = _mm256_sub_pd(_mm256_sub_pd(_mm256_set1_pd(1), _mm256_i64gather_pd(qualfp64, d1, 8)), qualin);
                y2 = _mm256_set1_pd(0);
                y3 = _mm256_set1_pd(0);
                M = y2;

                for(int h = 1; h < N; h++){
                    mask = _mm256_castsi256_pd(_mm256_cmpeq_epi64(rs0, _hap[h]));
                    sigma_aux = _mm256_blendv_pd(third, one, mask);

                    y2 = _mm256_mul_pd(sigma_aux, _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(Mprev[h-1],tmm), _mm256_mul_pd(nine,Iprev[h-1])), _mm256_mul_pd(d0,y3)));
                    y3 = _mm256_fmadd_pd(y3,one0,Mprev[h-1]);    

                    Iprev[h-1] = _mm256_fmadd_pd(Mprev[h-1],qualin, _mm256_mul_pd(Iprev[h-1],one0));
                    Mprev[h-1] = M;

                    M = y2;      
                }    

                Iprev[N-1] = _mm256_fmadd_pd(Mprev[N-1],qualin, _mm256_mul_pd(Iprev[N-1],one0));
                Mprev[N-1] = M;
            }

            y1 = _mm256_set1_pd(0);

            for(int h = 0; h < N; h++) y1 = _mm256_add_pd(y1, _mm256_add_pd(Mprev[h], Iprev[h]));

            _mm256_storeu_pd((finalsum + seqidx), y1);
        }
    }
    
    exec_time += omp_get_wtime();

    return exec_time;
}

template <int N, int P> 
double pairhmm_fp64_v2(unsigned long long* rslen, unsigned long long* haplen, unsigned long long* hap, unsigned long long* read, unsigned long long* q, unsigned long long* i, unsigned long long* d, double* qualfp64, double* finalsum, int seqsize){
    double exec_time = 0;

    omp_set_num_threads(min(seqsize/4, 192));
    
    exec_time = -omp_get_wtime();
    #pragma omp parallel
    { 
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        for(int seqidx = 4*tid; seqidx < seqsize; seqidx += 4*nthreads){
            
            __m256d Mprev[N];
            __m256d Iprev[N];
            __m256i _hap[N];
            __m256d auxM[N*P] = {_mm256_set1_pd(0)};
            __m256d auxI[N*P] = {_mm256_set1_pd(0)};
            __m256d auxD[N*P] = {_mm256_set1_pd(0)};

            __m256d M = _mm256_set1_pd(0);  
            __m256d y1 = _mm256_set1_pd(0); 
            __m256d y2 = _mm256_set1_pd(0); 
            __m256d y3 = _mm256_set1_pd(0);
            __m256d one;
            __m256d one0 = _mm256_set1_pd(0.1);
            __m256d nine = _mm256_set1_pd(0.9);
            __m256d third;
            __m256d d0 = _mm256_set1_pd(0);
            __m256d tmm = _mm256_set1_pd(0);
            __m256d sigma_aux = _mm256_set1_pd(0);
            __m256d qual = _mm256_set1_pd(0);
            __m256d qualin = _mm256_set1_pd(0);
            __m256d mask = _mm256_set1_pd(0);
            __m256d finalf = _mm256_set1_pd(0);
            __m256i rs0;
            __m256i q0;
            __m256i in;
            __m256i d1;
            
            const unsigned long long rsize = -rslen[seqidx] + rslen[seqidx + 1];
            const unsigned long long hsize = -haplen[seqidx] + haplen[seqidx + 1];
            const unsigned long long ridx = rslen[seqidx];
            const unsigned long long hidx = haplen[seqidx];

            for(int p = 0; p < P; p++){
            
                for(int h = 0; h < N; h++) _hap[h] = _mm256_load_si256((__m256i const*)(hap + hidx + p*N*4 + h*4));
                
                rs0 = _mm256_load_si256((__m256i const*)(read + ridx));
                q0 = _mm256_load_si256((__m256i const*)(q + ridx));
                d1 = _mm256_load_si256((__m256i const*)(d + ridx));
                qual = _mm256_i64gather_pd(qualfp64, q0, 8);
                third = _mm256_mul_pd(_mm256_set1_pd(0.3333333333333333333333333333333333),qual);
                one = _mm256_sub_pd(_mm256_set1_pd(1),qual);
                
                for(int h = 0; h < N; h++){
                    mask = _mm256_castsi256_pd(_mm256_cmpeq_epi64(rs0, _hap[h]));
                    sigma_aux = _mm256_blendv_pd(third, one, mask);
                    Mprev[h] = _mm256_mul_pd(sigma_aux, _mm256_mul_pd(nine, _mm256_div_pd(_mm256_set1_pd(ldexp(1.0, 1020.0)),_mm256_set1_pd(hsize))));
                    Iprev[h] = _mm256_set1_pd(0);
                }  

                for(int k = 1; k < rsize; k++){
                    rs0 = _mm256_load_si256((__m256i const*)(read + ridx + k*4));
                    q0 = _mm256_load_si256((__m256i const*)(q + ridx + k*4));
                    in = _mm256_load_si256((__m256i const*)(i + ridx + k*4));
                    d0 = _mm256_mul_pd(nine,_mm256_i64gather_pd(qualfp64, d1, 8));
                    d1 = _mm256_load_si256((__m256i const*)(d + ridx + k*4));

                    mask = _mm256_castsi256_pd(_mm256_cmpeq_epi64(rs0, _hap[0]));
                    qual = _mm256_i64gather_pd(qualfp64, q0, 8);
                    qualin = _mm256_i64gather_pd(qualfp64, in, 8);
                    third = _mm256_mul_pd(_mm256_set1_pd(0.3333333333333333333333333333333333),qual);
                    one = _mm256_sub_pd(_mm256_set1_pd(1),qual);
                    sigma_aux = _mm256_blendv_pd(third, one, mask);
                    tmm = _mm256_sub_pd(_mm256_sub_pd(_mm256_set1_pd(1), _mm256_i64gather_pd(qualfp64, d1, 8)), qualin);

                    M = _mm256_mul_pd(sigma_aux, _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(auxM[k-1],tmm), _mm256_mul_pd(nine,auxI[k-1])), _mm256_mul_pd(d0,auxD[k-1])));
                    y3 = _mm256_fmadd_pd(auxD[k-1],one0,auxM[k-1]);

                    for(int h = 1; h < N; h++){
                        mask = _mm256_castsi256_pd(_mm256_cmpeq_epi64(rs0, _hap[h]));
                        sigma_aux = _mm256_blendv_pd(third, one, mask);

                        y2 = _mm256_mul_pd(sigma_aux, _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(Mprev[h-1],tmm), _mm256_mul_pd(nine,Iprev[h-1])), _mm256_mul_pd(d0,y3)));
                        y3 = _mm256_fmadd_pd(y3,one0,Mprev[h-1]);    

                        Iprev[h-1] = _mm256_fmadd_pd(Mprev[h-1],qualin, _mm256_mul_pd(Iprev[h-1],one0));
                        Mprev[h-1] = M;

                        M = y2;      
                    }    

                    auxM[k-1] = Mprev[N-1];
                    auxI[k-1] = Iprev[N-1];
                    auxD[k-1] = y3;

                    Iprev[N-1] = _mm256_fmadd_pd(Mprev[N-1],qualin, _mm256_mul_pd(Iprev[N-1],one0));
                    Mprev[N-1] = M;
                }

                for(int h = 0; h < N; h++) finalf = _mm256_add_pd(finalf, _mm256_add_pd(Mprev[h], Iprev[h]));
            }

            _mm256_storeu_pd((finalsum + seqidx), finalf);
        }
    }
    
    exec_time += omp_get_wtime();

    return exec_time;
}

template <int N, int P> 
double pairhmm_fp64_v3(unsigned long long* rslen, unsigned long long* haplen, unsigned long long* hap, unsigned long long* read, unsigned long long* q, unsigned long long* i, unsigned long long* d, double* qualfp64, double* finalsum, int seqsize){
    double exec_time = 0;

    omp_set_num_threads(min(seqsize/4, 192));
    
    exec_time = -omp_get_wtime();
    #pragma omp parallel
    { 
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        __m256d* auxM = (__m256d*)aligned_alloc(32, N*P*sizeof(__m256d));
        __m256d* auxI = (__m256d*)aligned_alloc(32, N*P*sizeof(__m256d));
        __m256d* auxD = (__m256d*)aligned_alloc(32, N*P*sizeof(__m256d));

        for(int seqidx = 4*tid; seqidx < seqsize; seqidx += 4*nthreads){
            
            __m256d Mprev[N];
            __m256d Iprev[N];
            __m256i _hap[N];

            __m256d M = _mm256_set1_pd(0);  
            __m256d y1 = _mm256_set1_pd(0); 
            __m256d y2 = _mm256_set1_pd(0); 
            __m256d y3 = _mm256_set1_pd(0);
            __m256d one;
            __m256d one0 = _mm256_set1_pd(0.1);
            __m256d nine = _mm256_set1_pd(0.9);
            __m256d third;
            __m256d d0 = _mm256_set1_pd(0);
            __m256d tmm = _mm256_set1_pd(0);
            __m256d sigma_aux = _mm256_set1_pd(0);
            __m256d qual = _mm256_set1_pd(0);
            __m256d qualin = _mm256_set1_pd(0);
            __m256d mask = _mm256_set1_pd(0);
            __m256d finalf = _mm256_set1_pd(0);
            __m256i rs0;
            __m256i q0;
            __m256i in;
            __m256i d1;
            
            const unsigned long long rsize = -rslen[seqidx] + rslen[seqidx + 1];
            const unsigned long long hsize = -haplen[seqidx] + haplen[seqidx + 1];
            const unsigned long long ridx = rslen[seqidx];
            const unsigned long long hidx = haplen[seqidx];

            for(int p = 0; p < P; p++){
            
                for(int h = 0; h < N; h++) _hap[h] = _mm256_load_si256((__m256i const*)(hap + hidx + p*N*4 + h*4));
                
                rs0 = _mm256_load_si256((__m256i const*)(read + ridx));
                q0 = _mm256_load_si256((__m256i const*)(q + ridx));
                d1 = _mm256_load_si256((__m256i const*)(d + ridx));
                qual = _mm256_i64gather_pd(qualfp64, q0, 8);
                third = _mm256_mul_pd(_mm256_set1_pd(0.3333333333333333333333333333333333),qual);
                one = _mm256_sub_pd(_mm256_set1_pd(1),qual);
                
                for(int h = 0; h < N; h++){
                    mask = _mm256_castsi256_pd(_mm256_cmpeq_epi64(rs0, _hap[h]));
                    sigma_aux = _mm256_blendv_pd(third, one, mask);
                    Mprev[h] = _mm256_mul_pd(sigma_aux, _mm256_mul_pd(nine, _mm256_div_pd(_mm256_set1_pd(ldexp(1.0, 1020.0)),_mm256_set1_pd(hsize))));
                    Iprev[h] = _mm256_set1_pd(0);
                }  

                for(int k = 1; k < rsize; k++){
                    rs0 = _mm256_load_si256((__m256i const*)(read + ridx + k*4));
                    q0 = _mm256_load_si256((__m256i const*)(q + ridx + k*4));
                    in = _mm256_load_si256((__m256i const*)(i + ridx + k*4));
                    d0 = _mm256_mul_pd(nine,_mm256_i64gather_pd(qualfp64, d1, 8));
                    d1 = _mm256_load_si256((__m256i const*)(d + ridx + k*4));

                    mask = _mm256_castsi256_pd(_mm256_cmpeq_epi64(rs0, _hap[0]));
                    qual = _mm256_i64gather_pd(qualfp64, q0, 8);
                    qualin = _mm256_i64gather_pd(qualfp64, in, 8);
                    third = _mm256_mul_pd(_mm256_set1_pd(0.3333333333333333333333333333333333),qual);
                    one = _mm256_sub_pd(_mm256_set1_pd(1),qual);
                    sigma_aux = _mm256_blendv_pd(third, one, mask);
                    tmm = _mm256_sub_pd(_mm256_sub_pd(_mm256_set1_pd(1), _mm256_i64gather_pd(qualfp64, d1, 8)), qualin);

                    M = _mm256_mul_pd(sigma_aux, _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(auxM[k-1],tmm), _mm256_mul_pd(nine,auxI[k-1])), _mm256_mul_pd(d0,auxD[k-1])));
                    y3 = _mm256_fmadd_pd(auxD[k-1],one0,auxM[k-1]);

                    for(int h = 1; h < N; h++){
                        mask = _mm256_castsi256_pd(_mm256_cmpeq_epi64(rs0, _hap[h]));
                        sigma_aux = _mm256_blendv_pd(third, one, mask);

                        y2 = _mm256_mul_pd(sigma_aux, _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(Mprev[h-1],tmm), _mm256_mul_pd(nine,Iprev[h-1])), _mm256_mul_pd(d0,y3)));
                        y3 = _mm256_fmadd_pd(y3,one0,Mprev[h-1]);    

                        Iprev[h-1] = _mm256_fmadd_pd(Mprev[h-1],qualin, _mm256_mul_pd(Iprev[h-1],one0));
                        Mprev[h-1] = M;

                        M = y2;      
                    }    

                    auxM[k-1] = Mprev[N-1];
                    auxI[k-1] = Iprev[N-1];
                    auxD[k-1] = y3;

                    Iprev[N-1] = _mm256_fmadd_pd(Mprev[N-1],qualin, _mm256_mul_pd(Iprev[N-1],one0));
                    Mprev[N-1] = M;
                }

                for(int h = 0; h < N; h++) finalf = _mm256_add_pd(finalf, _mm256_add_pd(Mprev[h], Iprev[h]));
            }

            _mm256_storeu_pd((finalsum + seqidx), finalf);
        }

        free(auxM);
        free(auxI);
        free(auxD);
    }
    
    exec_time += omp_get_wtime();

    return exec_time;
}

double pairhmm_benchmark_fp64(unsigned long long* rslen, unsigned long long* haplen, unsigned long long* hap, unsigned long long* rs, unsigned long long* q, unsigned long long* i, unsigned long long* d, double* qual, double* finalsum, int maxsize, int seq_size){

    double total_time = 0;

    if(maxsize == 32) total_time = pairhmm_fp64<32>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 64) total_time = pairhmm_fp64<64>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 128) total_time = pairhmm_fp64_v2<64,2>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 256) total_time = pairhmm_fp64_v2<64,4>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 512) total_time = pairhmm_fp64_v2<64,8>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 1024) total_time = pairhmm_fp64_v2<64,16>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 2048) total_time = pairhmm_fp64_v2<64,32>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 4096) total_time = pairhmm_fp64_v2<64,64>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 8192) total_time = pairhmm_fp64_v2<64,128>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 16384) total_time = pairhmm_fp64_v2<64,256>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 32768) total_time = pairhmm_fp64_v2<64,512>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 65536) total_time = pairhmm_fp64_v3<64,1024>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 131072) total_time = pairhmm_fp64_v3<64,2048>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);

    return total_time;
}

template <int N> 
double pairhmm_fp32(unsigned long long* rslen, unsigned long long* haplen, unsigned int* hap, unsigned int* read, unsigned int* q, unsigned int* i, unsigned int* d, float* qualfp32, float* finalsum, int seqsize){
    double exec_time = 0;

    omp_set_num_threads(min(seqsize/8, 192));
    
    exec_time = -omp_get_wtime();
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        for(int seqidx = 8*tid; seqidx < seqsize; seqidx += 8*nthreads){
            
            __m256 Mprev[N];
            __m256 Iprev[N];
            __m256i _hap[N];

            __m256 M = _mm256_set1_ps(0.0f);  
            __m256 y1 = _mm256_set1_ps(0.0f); 
            __m256 y2 = _mm256_set1_ps(0.0f); 
            __m256 y3 = _mm256_set1_ps(0.0f);
            __m256 one = _mm256_set1_ps(1.0f);
            __m256 one0 = _mm256_set1_ps(0.1f);
            __m256 nine = _mm256_set1_ps(0.9f);
            __m256 third = _mm256_set1_ps(0.333333333333333333333333333333f);
            __m256 d0 = _mm256_set1_ps(0.0f);
            __m256 tmm = _mm256_set1_ps(0.0f);
            __m256 sigma_aux = _mm256_set1_ps(0.0f);
            __m256 qual = _mm256_set1_ps(0.0f);
            __m256 qualin = _mm256_set1_ps(0.0f);
            __m256 mask = _mm256_set1_ps(0.0f);
            __m256i rs0;
            __m256i q0;
            __m256i in;
            __m256i d1;
            
            const unsigned long long rsize = -rslen[seqidx] + rslen[seqidx + 1];
            const unsigned long long hsize = -haplen[seqidx] + haplen[seqidx + 1];
            const unsigned long long ridx = rslen[seqidx];
            const unsigned long long hidx = haplen[seqidx];
            
            for(int h = 0; h < N; h++) _hap[h] = _mm256_load_si256((__m256i const*)(hap + hidx + h*8));
            
            rs0 = _mm256_load_si256((__m256i const*)(read + ridx));
            q0 = _mm256_load_si256((__m256i const*)(q + ridx));
            d1 = _mm256_load_si256((__m256i const*)(d + ridx));
            qual = _mm256_i32gather_ps(qualfp32, q0, 4);
            third = _mm256_mul_ps(_mm256_set1_ps(0.3333333333333333333333333333333333f),qual);
            one = _mm256_sub_ps(_mm256_set1_ps(1.0f),qual);
            
            for(int h = 0; h < N; h++){
                mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(rs0, _hap[h]));
                sigma_aux = _mm256_blendv_ps(third, one, mask);
                Mprev[h] = _mm256_mul_ps(sigma_aux, _mm256_mul_ps(nine, _mm256_div_ps(_mm256_set1_ps(ldexpf(1.f, 120.f)),_mm256_set1_ps(hsize))));
                Iprev[h] = _mm256_set1_ps(0.0f);
            }  

            for(int k = 1; k < rsize; k++){
                rs0 = _mm256_load_si256((__m256i const*)(read + ridx + k*8));
                q0 = _mm256_load_si256((__m256i const*)(q + ridx + k*8));
                in = _mm256_load_si256((__m256i const*)(i + ridx + k*8));
                d0 = _mm256_mul_ps(nine,_mm256_i32gather_ps(qualfp32, d1, 4));
                d1 = _mm256_load_si256((__m256i const*)(d + ridx + k*8));

                mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(rs0, _hap[0]));
                qual = _mm256_i32gather_ps(qualfp32, q0, 4);
                qualin = _mm256_i32gather_ps(qualfp32, in, 4);
                third = _mm256_mul_ps(_mm256_set1_ps(0.3333333333333333333333333333333333f),qual);
                one = _mm256_sub_ps(_mm256_set1_ps(1.0f),qual);
                sigma_aux = _mm256_blendv_ps(third, one, mask);
                tmm = _mm256_sub_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_i32gather_ps(qualfp32, d1, 4)), qualin);
                y2 = _mm256_set1_ps(0.0f);
                y3 = _mm256_set1_ps(0.0f);
                M = y2;

                for(int h = 1; h < N; h++){
                    mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(rs0, _hap[h]));
                    sigma_aux = _mm256_blendv_ps(third, one, mask);

                    y2 = _mm256_mul_ps(sigma_aux, _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(Mprev[h-1],tmm), _mm256_mul_ps(nine,Iprev[h-1])), _mm256_mul_ps(d0,y3)));
                    y3 = _mm256_fmadd_ps(y3,one0,Mprev[h-1]);    

                    Iprev[h-1] = _mm256_fmadd_ps(Mprev[h-1],qualin, _mm256_mul_ps(Iprev[h-1],one0));
                    Mprev[h-1] = M;

                    M = y2;      
                }    

                Iprev[N-1] = _mm256_fmadd_ps(Mprev[N-1],qualin, _mm256_mul_ps(Iprev[N-1],one0));
                Mprev[N-1] = M; 
            }

            y1 = _mm256_set1_ps(0.0f);

            for(int h = 0; h < N; h++) y1 = _mm256_add_ps(y1, _mm256_add_ps(Mprev[h], Iprev[h]));

            _mm256_storeu_ps((finalsum + seqidx), y1);
        }
    }

    exec_time += omp_get_wtime();

    return exec_time;
}

template <int N, int P> 
double pairhmm_fp32_v2(unsigned long long* rslen, unsigned long long* haplen, unsigned int* hap, unsigned int* read, unsigned int* q, unsigned int* i, unsigned int* d, float* qualfp32, float* finalsum, int seqsize){
    double exec_time = 0;

    omp_set_num_threads(min(seqsize/8, 192));
    
    exec_time = -omp_get_wtime();
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        for(int seqidx = 8*tid; seqidx < seqsize; seqidx += 8*nthreads){
            
            __m256 Mprev[N];
            __m256 Iprev[N];
            __m256i _hap[N];
            __m256 auxM[N*P] = {_mm256_set1_ps(0)};
            __m256 auxI[N*P] = {_mm256_set1_ps(0)};
            __m256 auxD[N*P] = {_mm256_set1_ps(0)};

            __m256 M = _mm256_set1_ps(0.0f);  
            __m256 y1 = _mm256_set1_ps(0.0f); 
            __m256 y2 = _mm256_set1_ps(0.0f); 
            __m256 y3 = _mm256_set1_ps(0.0f);
            __m256 one = _mm256_set1_ps(1.0f);
            __m256 one0 = _mm256_set1_ps(0.1f);
            __m256 nine = _mm256_set1_ps(0.9f);
            __m256 third = _mm256_set1_ps(0.333333333333333333333333333333f);
            __m256 d0 = _mm256_set1_ps(0.0f);
            __m256 tmm = _mm256_set1_ps(0.0f);
            __m256 sigma_aux = _mm256_set1_ps(0.0f);
            __m256 finalf = _mm256_set1_ps(0.0f);
            __m256 qual = _mm256_set1_ps(0.0f);
            __m256 qualin = _mm256_set1_ps(0.0f);
            __m256 mask = _mm256_set1_ps(0.0f);
            __m256i rs0;
            __m256i q0;
            __m256i in;
            __m256i d1;
            
            const unsigned long long rsize = -rslen[seqidx] + rslen[seqidx + 1];
            const unsigned long long hsize = -haplen[seqidx] + haplen[seqidx + 1];
            const unsigned long long ridx = rslen[seqidx];
            const unsigned long long hidx = haplen[seqidx];

            for(int p = 0; p < P; p++){
            
                for(int h = 0; h < N; h++) _hap[h] = _mm256_load_si256((__m256i const*)(hap + hidx + p*N*8 + h*8));
                
                rs0 = _mm256_load_si256((__m256i const*)(read + ridx));
                q0 = _mm256_load_si256((__m256i const*)(q + ridx));
                d1 = _mm256_load_si256((__m256i const*)(d + ridx));
                qual = _mm256_i32gather_ps(qualfp32, q0, 4);
                third = _mm256_mul_ps(_mm256_set1_ps(0.3333333333333333333333333333333333f),qual);
                one = _mm256_sub_ps(_mm256_set1_ps(1.0f),qual);
                
                for(int h = 0; h < N; h++){
                    mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(rs0, _hap[h]));
                    sigma_aux = _mm256_blendv_ps(third, one, mask);
                    Mprev[h] = _mm256_mul_ps(sigma_aux, _mm256_mul_ps(nine, _mm256_div_ps(_mm256_set1_ps(ldexpf(1.f, 120.f)),_mm256_set1_ps(hsize))));
                    Iprev[h] = _mm256_set1_ps(0.0f);
                }  

                for(int k = 1; k < rsize; k++){
                    rs0 = _mm256_load_si256((__m256i const*)(read + ridx + k*8));
                    q0 = _mm256_load_si256((__m256i const*)(q + ridx + k*8));
                    in = _mm256_load_si256((__m256i const*)(i + ridx + k*8));
                    d0 = _mm256_mul_ps(nine,_mm256_i32gather_ps(qualfp32, d1, 4));
                    d1 = _mm256_load_si256((__m256i const*)(d + ridx + k*8));

                    mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(rs0, _hap[0]));
                    qual = _mm256_i32gather_ps(qualfp32, q0, 4);
                    qualin = _mm256_i32gather_ps(qualfp32, in, 4);
                    third = _mm256_mul_ps(_mm256_set1_ps(0.3333333333333333333333333333333333f),qual);
                    one = _mm256_sub_ps(_mm256_set1_ps(1.0f),qual);
                    sigma_aux = _mm256_blendv_ps(third, one, mask);
                    tmm = _mm256_sub_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_i32gather_ps(qualfp32, d1, 4)), qualin);

                    M = _mm256_mul_ps(sigma_aux, _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(auxM[k-1],tmm), _mm256_mul_ps(nine,auxI[k-1])), _mm256_mul_ps(d0,auxD[k-1])));
                    y3 = _mm256_fmadd_ps(auxD[k-1],one0,auxM[k-1]);

                    for(int h = 1; h < N; h++){
                        mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(rs0, _hap[h]));
                        sigma_aux = _mm256_blendv_ps(third, one, mask);

                        y2 = _mm256_mul_ps(sigma_aux, _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(Mprev[h-1],tmm), _mm256_mul_ps(nine,Iprev[h-1])), _mm256_mul_ps(d0,y3)));
                        y3 = _mm256_fmadd_ps(y3,one0,Mprev[h-1]);    

                        Iprev[h-1] = _mm256_fmadd_ps(Mprev[h-1],qualin, _mm256_mul_ps(Iprev[h-1],one0));
                        Mprev[h-1] = M;

                        M = y2;      
                    }    

                    auxM[k-1] = Mprev[N-1];
                    auxI[k-1] = Iprev[N-1];
                    auxD[k-1] = y3;

                    Iprev[N-1] = _mm256_fmadd_ps(Mprev[N-1],qualin, _mm256_mul_ps(Iprev[N-1],one0));
                    Mprev[N-1] = M; 
                }

                for(int h = 0; h < N; h++) finalf = _mm256_add_ps(finalf, _mm256_add_ps(Mprev[h], Iprev[h]));
            }

            _mm256_storeu_ps((finalsum + seqidx), finalf);
        }
    }

    exec_time += omp_get_wtime();

    return exec_time;
}

template <int N, int P> 
double pairhmm_fp32_v3(unsigned long long* rslen, unsigned long long* haplen, unsigned int* hap, unsigned int* read, unsigned int* q, unsigned int* i, unsigned int* d, float* qualfp32, float* finalsum, int seqsize){
    double exec_time = 0;

    omp_set_num_threads(min(seqsize/8, 192));
    
    exec_time = -omp_get_wtime();
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        __m256* auxM = (__m256*)aligned_alloc(32, N*P*sizeof(__m256));
        __m256* auxI = (__m256*)aligned_alloc(32, N*P*sizeof(__m256));
        __m256* auxD = (__m256*)aligned_alloc(32, N*P*sizeof(__m256));

        for(int seqidx = 8*tid; seqidx < seqsize; seqidx += 8*nthreads){
            
            __m256 Mprev[N];
            __m256 Iprev[N];
            __m256i _hap[N];

            __m256 M = _mm256_set1_ps(0.0f);  
            __m256 y1 = _mm256_set1_ps(0.0f); 
            __m256 y2 = _mm256_set1_ps(0.0f); 
            __m256 y3 = _mm256_set1_ps(0.0f);
            __m256 one = _mm256_set1_ps(1.0f);
            __m256 one0 = _mm256_set1_ps(0.1f);
            __m256 nine = _mm256_set1_ps(0.9f);
            __m256 third = _mm256_set1_ps(0.333333333333333333333333333333f);
            __m256 d0 = _mm256_set1_ps(0.0f);
            __m256 tmm = _mm256_set1_ps(0.0f);
            __m256 sigma_aux = _mm256_set1_ps(0.0f);
            __m256 finalf = _mm256_set1_ps(0.0f);
            __m256 qual = _mm256_set1_ps(0.0f);
            __m256 qualin = _mm256_set1_ps(0.0f);
            __m256 mask = _mm256_set1_ps(0.0f);
            __m256i rs0;
            __m256i q0;
            __m256i in;
            __m256i d1;
            
            const unsigned long long rsize = -rslen[seqidx] + rslen[seqidx + 1];
            const unsigned long long hsize = -haplen[seqidx] + haplen[seqidx + 1];
            const unsigned long long ridx = rslen[seqidx];
            const unsigned long long hidx = haplen[seqidx];

            for(int p = 0; p < P; p++){
            
                for(int h = 0; h < N; h++) _hap[h] = _mm256_load_si256((__m256i const*)(hap + hidx + p*N*8 + h*8));
                
                rs0 = _mm256_load_si256((__m256i const*)(read + ridx));
                q0 = _mm256_load_si256((__m256i const*)(q + ridx));
                d1 = _mm256_load_si256((__m256i const*)(d + ridx));
                qual = _mm256_i32gather_ps(qualfp32, q0, 4);
                third = _mm256_mul_ps(_mm256_set1_ps(0.3333333333333333333333333333333333f),qual);
                one = _mm256_sub_ps(_mm256_set1_ps(1.0f),qual);
                
                for(int h = 0; h < N; h++){
                    mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(rs0, _hap[h]));
                    sigma_aux = _mm256_blendv_ps(third, one, mask);
                    Mprev[h] = _mm256_mul_ps(sigma_aux, _mm256_mul_ps(nine, _mm256_div_ps(_mm256_set1_ps(ldexpf(1.f, 120.f)),_mm256_set1_ps(hsize))));
                    Iprev[h] = _mm256_set1_ps(0.0f);
                }  

                for(int k = 1; k < rsize; k++){
                    rs0 = _mm256_load_si256((__m256i const*)(read + ridx + k*8));
                    q0 = _mm256_load_si256((__m256i const*)(q + ridx + k*8));
                    in = _mm256_load_si256((__m256i const*)(i + ridx + k*8));
                    d0 = _mm256_mul_ps(nine,_mm256_i32gather_ps(qualfp32, d1, 4));
                    d1 = _mm256_load_si256((__m256i const*)(d + ridx + k*8));

                    mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(rs0, _hap[0]));
                    qual = _mm256_i32gather_ps(qualfp32, q0, 4);
                    qualin = _mm256_i32gather_ps(qualfp32, in, 4);
                    third = _mm256_mul_ps(_mm256_set1_ps(0.3333333333333333333333333333333333f),qual);
                    one = _mm256_sub_ps(_mm256_set1_ps(1.0f),qual);
                    sigma_aux = _mm256_blendv_ps(third, one, mask);
                    tmm = _mm256_sub_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_i32gather_ps(qualfp32, d1, 4)), qualin);

                    M = _mm256_mul_ps(sigma_aux, _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(auxM[k-1],tmm), _mm256_mul_ps(nine,auxI[k-1])), _mm256_mul_ps(d0,auxD[k-1])));
                    y3 = _mm256_fmadd_ps(auxD[k-1],one0,auxM[k-1]);

                    for(int h = 1; h < N; h++){
                        mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(rs0, _hap[h]));
                        sigma_aux = _mm256_blendv_ps(third, one, mask);

                        y2 = _mm256_mul_ps(sigma_aux, _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(Mprev[h-1],tmm), _mm256_mul_ps(nine,Iprev[h-1])), _mm256_mul_ps(d0,y3)));
                        y3 = _mm256_fmadd_ps(y3,one0,Mprev[h-1]);    

                        Iprev[h-1] = _mm256_fmadd_ps(Mprev[h-1],qualin, _mm256_mul_ps(Iprev[h-1],one0));
                        Mprev[h-1] = M;

                        M = y2;      
                    }    

                    auxM[k-1] = Mprev[N-1];
                    auxI[k-1] = Iprev[N-1];
                    auxD[k-1] = y3;

                    Iprev[N-1] = _mm256_fmadd_ps(Mprev[N-1],qualin, _mm256_mul_ps(Iprev[N-1],one0));
                    Mprev[N-1] = M; 
                }

                for(int h = 0; h < N; h++) finalf = _mm256_add_ps(finalf, _mm256_add_ps(Mprev[h], Iprev[h]));
            }

            _mm256_storeu_ps((finalsum + seqidx), finalf);
        }

        free(auxM);
        free(auxI);
        free(auxD);
    }

    exec_time += omp_get_wtime();

    return exec_time;
}



double pairhmm_benchmark_fp32(unsigned long long* rslen, unsigned long long* haplen, unsigned int* hap, unsigned int* rs, unsigned int* q, unsigned int* i, unsigned int* d, float* qual, float* finalsum, int maxsize, int seq_size){

    double total_time = 0;

    if(maxsize == 32) total_time = pairhmm_fp32<32>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 64) total_time = pairhmm_fp32<64>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 128) total_time = pairhmm_fp32_v2<64,2>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 256) total_time = pairhmm_fp32_v2<64,4>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 512) total_time = pairhmm_fp32_v2<64,8>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 1024) total_time = pairhmm_fp32_v2<64,16>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 2048) total_time = pairhmm_fp32_v2<64,32>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 4096) total_time = pairhmm_fp32_v2<64,64>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 8192) total_time = pairhmm_fp32_v2<64,128>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 16384) total_time = pairhmm_fp32_v2<64,256>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 32768) total_time = pairhmm_fp32_v2<64,512>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 65536) total_time = pairhmm_fp32_v3<64,1024>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 131072) total_time = pairhmm_fp32_v3<64,2048>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);

    return total_time;
}

//AVX-512

template <int N> 
double pairhmm_fp64_avx512(unsigned long long* rslen, unsigned long long* haplen, unsigned long long* hap, unsigned long long* read, unsigned long long* q, unsigned long long* i, unsigned long long* d, double* qualfp64, double* finalsum, int seqsize){
    double exec_time = 0;

    omp_set_num_threads(min(seqsize/8, 192));
    
    exec_time = -omp_get_wtime();
    #pragma omp parallel
    {

        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        for(int seqidx = 8*tid; seqidx < seqsize; seqidx += 8*nthreads){
            
            __m512d Mprev[N];
            __m512d Iprev[N];
            __m512i _hap[N];

            __m512d M = _mm512_set1_pd(0);  
            __m512d y1 = _mm512_set1_pd(0); 
            __m512d y2 = _mm512_set1_pd(0); 
            __m512d y3 = _mm512_set1_pd(0);
            __m512d one;
            __m512d one0 = _mm512_set1_pd(0.1);
            __m512d nine = _mm512_set1_pd(0.9);
            __m512d third;
            __m512d d0 = _mm512_set1_pd(0);
            __m512d tmm = _mm512_set1_pd(0);
            __m512d sigma_aux = _mm512_set1_pd(0);
            __m512d qual = _mm512_set1_pd(0);
            __m512d qualin = _mm512_set1_pd(0);
            __mmask8 mask;
            __m512i rs0;
            __m512i q0;
            __m512i in;
            __m512i d1;
            
            const unsigned long long rsize = -rslen[seqidx] + rslen[seqidx + 1];
            const unsigned long long hsize = -haplen[seqidx] + haplen[seqidx + 1];
            const unsigned long long ridx = rslen[seqidx];
            const unsigned long long hidx = haplen[seqidx];
            
            for(int h = 0; h < N; h++) _hap[h] = _mm512_load_si512((__m512i const*)(hap + hidx + h*8));
            
            rs0 = _mm512_load_si512((__m512i const*)(read + ridx));
            q0 = _mm512_load_si512((__m512i const*)(q + ridx));
            d1 = _mm512_load_si512((__m512i const*)(d + ridx));
            qual = _mm512_i64gather_pd(q0, qualfp64, 8);
            third = _mm512_mul_pd(_mm512_set1_pd(0.3333333333333333333333333333333333),qual);
            one = _mm512_sub_pd(_mm512_set1_pd(1),qual);
            
            for(int h = 0; h < N; h++){
                mask = _mm512_cmpeq_epi64_mask(rs0, _hap[h]);
                sigma_aux = _mm512_mask_blend_pd(mask, third, one);
                Mprev[h] = _mm512_mul_pd(sigma_aux, _mm512_mul_pd(nine, _mm512_div_pd(_mm512_set1_pd(ldexp(1.0, 1020.0)),_mm512_set1_pd(hsize))));
                Iprev[h] = _mm512_set1_pd(0);
            }  

            for(int k = 1; k < rsize; k++){
                rs0 = _mm512_load_si512((__m512i const*)(read + ridx + k*8));
                q0 = _mm512_load_si512((__m512i const*)(q + ridx + k*8));
                in = _mm512_load_si512((__m512i const*)(i + ridx + k*8));
                d0 = _mm512_mul_pd(nine,_mm512_i64gather_pd(d1, qualfp64, 8));
                d1 = _mm512_load_si512((__m512i const*)(d + ridx + k*8));

                mask = _mm512_cmpeq_epi64_mask(rs0, _hap[0]);
                qual = _mm512_i64gather_pd(q0, qualfp64, 8);
                qualin = _mm512_i64gather_pd(in, qualfp64, 8);
                third = _mm512_mul_pd(_mm512_set1_pd(0.3333333333333333333333333333333333),qual);
                one = _mm512_sub_pd(_mm512_set1_pd(1),qual);
                sigma_aux = _mm512_mask_blend_pd(mask, third, one);
                tmm = _mm512_sub_pd(_mm512_sub_pd(_mm512_set1_pd(1), _mm512_i64gather_pd(d1, qualfp64, 8)), qualin);
                y2 = _mm512_set1_pd(0);
                y3 = _mm512_set1_pd(0);
                M = y2;

                for(int h = 1; h < N; h++){
                    mask = _mm512_cmpeq_epi64_mask(rs0, _hap[h]);
                    sigma_aux = _mm512_mask_blend_pd(mask, third, one);

                    y2 = _mm512_mul_pd(sigma_aux, _mm512_add_pd(_mm512_add_pd(_mm512_mul_pd(Mprev[h-1],tmm), _mm512_mul_pd(nine,Iprev[h-1])), _mm512_mul_pd(d0,y3)));
                    y3 = _mm512_fmadd_pd(y3,one0,Mprev[h-1]);    

                    Iprev[h-1] = _mm512_fmadd_pd(Mprev[h-1],qualin, _mm512_mul_pd(Iprev[h-1],one0));
                    Mprev[h-1] = M;

                    M = y2;      
                }    

                Iprev[N-1] = _mm512_fmadd_pd(Mprev[N-1],qualin, _mm512_mul_pd(Iprev[N-1],one0));
                Mprev[N-1] = M;
            }

            y1 = _mm512_set1_pd(0);

            for(int h = 0; h < N; h++) y1 = _mm512_add_pd(y1, _mm512_add_pd(Mprev[h], Iprev[h]));

            _mm512_storeu_pd((finalsum + seqidx), y1);
        }
        
        
    }
    
    exec_time += omp_get_wtime();

    return exec_time;
}

template <int N, int P> 
double pairhmm_fp64_avx512_v2(unsigned long long* rslen, unsigned long long* haplen, unsigned long long* hap, unsigned long long* read, unsigned long long* q, unsigned long long* i, unsigned long long* d, double* qualfp64, double* finalsum, int seqsize){
    double exec_time = 0;

    omp_set_num_threads(min(seqsize/8, 192));
    
    exec_time = -omp_get_wtime();
    #pragma omp parallel
    {

        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        for(int seqidx = 8*tid; seqidx < seqsize; seqidx += 8*nthreads){
            
            __m512d Mprev[N];
            __m512d Iprev[N];
            __m512d auxM[N*P] = {_mm512_set1_pd(0)};
            __m512d auxI[N*P] = {_mm512_set1_pd(0)};
            __m512d auxD[N*P] = {_mm512_set1_pd(0)};
            __m512i _hap[N];

            __m512d M = _mm512_set1_pd(0);  
            __m512d y1 = _mm512_set1_pd(0); 
            __m512d y2 = _mm512_set1_pd(0); 
            __m512d y3 = _mm512_set1_pd(0);
            __m512d one;
            __m512d one0 = _mm512_set1_pd(0.1);
            __m512d nine = _mm512_set1_pd(0.9);
            __m512d third;
            __m512d finalf = _mm512_set1_pd(0);
            __m512d d0 = _mm512_set1_pd(0);
            __m512d tmm = _mm512_set1_pd(0);
            __m512d sigma_aux = _mm512_set1_pd(0);
            __m512d qual = _mm512_set1_pd(0);
            __m512d qualin = _mm512_set1_pd(0);
            __mmask8 mask;
            __m512i rs0;
            __m512i q0;
            __m512i in;
            __m512i d1;
            
            const unsigned long long rsize = -rslen[seqidx] + rslen[seqidx + 1];
            const unsigned long long hsize = -haplen[seqidx] + haplen[seqidx + 1];
            const unsigned long long ridx = rslen[seqidx];
            const unsigned long long hidx = haplen[seqidx];

            for(int p = 0; p < P; p++){
            
                for(int h = 0; h < N; h++) _hap[h] = _mm512_load_si512((__m512i const*)(hap + hidx + p*N*8 + h*8));
                
                rs0 = _mm512_load_si512((__m512i const*)(read + ridx));
                q0 = _mm512_load_si512((__m512i const*)(q + ridx));
                d1 = _mm512_load_si512((__m512i const*)(d + ridx));
                qual = _mm512_i64gather_pd(q0, qualfp64, 8);
                third = _mm512_mul_pd(_mm512_set1_pd(0.3333333333333333333333333333333333),qual);
                one = _mm512_sub_pd(_mm512_set1_pd(1),qual);
                
                for(int h = 0; h < N; h++){
                    mask = _mm512_cmpeq_epi64_mask(rs0, _hap[h]);
                    sigma_aux = _mm512_mask_blend_pd(mask, third, one);
                    Mprev[h] = _mm512_mul_pd(sigma_aux, _mm512_mul_pd(nine, _mm512_div_pd(_mm512_set1_pd(ldexp(1.0, 1020.0)), _mm512_set1_pd(hsize))));
                    Iprev[h] = _mm512_set1_pd(0);
                }  

                for(int k = 1; k < rsize; k++){
                    rs0 = _mm512_load_si512((__m512i const*)(read + ridx + k*8));
                    q0 = _mm512_load_si512((__m512i const*)(q + ridx + k*8));
                    in = _mm512_load_si512((__m512i const*)(i + ridx + k*8));
                    d0 = _mm512_mul_pd(nine,_mm512_i64gather_pd(d1, qualfp64, 8));
                    d1 = _mm512_load_si512((__m512i const*)(d + ridx + k*8));

                    mask = _mm512_cmpeq_epi64_mask(rs0, _hap[0]);
                    qual = _mm512_i64gather_pd(q0, qualfp64, 8);
                    qualin = _mm512_i64gather_pd(in, qualfp64, 8);
                    third = _mm512_mul_pd(_mm512_set1_pd(0.3333333333333333333333333333333333),qual);
                    one = _mm512_sub_pd(_mm512_set1_pd(1),qual);
                    sigma_aux = _mm512_mask_blend_pd(mask, third, one);
                    tmm = _mm512_sub_pd(_mm512_sub_pd(_mm512_set1_pd(1), _mm512_i64gather_pd(d1, qualfp64, 8)), qualin);

                    M = _mm512_mul_pd(sigma_aux, _mm512_add_pd(_mm512_add_pd(_mm512_mul_pd(auxM[k-1],tmm), _mm512_mul_pd(nine,auxI[k-1])), _mm512_mul_pd(d0,auxD[k-1])));

                    y3 = _mm512_fmadd_pd(auxD[k-1],one0,auxM[k-1]);

                    for(int h = 1; h < N; h++){
                        mask = _mm512_cmpeq_epi64_mask(rs0, _hap[h]);
                        sigma_aux = _mm512_mask_blend_pd(mask, third, one);

                        y2 = _mm512_mul_pd(sigma_aux, _mm512_add_pd(_mm512_add_pd(_mm512_mul_pd(Mprev[h-1],tmm), _mm512_mul_pd(nine,Iprev[h-1])), _mm512_mul_pd(d0,y3)));
                        y3 = _mm512_fmadd_pd(y3,one0,Mprev[h-1]);    

                        Iprev[h-1] = _mm512_fmadd_pd(Mprev[h-1],qualin, _mm512_mul_pd(Iprev[h-1],one0));
                        Mprev[h-1] = M;

                        M = y2;      
                    }

                    auxM[k-1] = Mprev[N-1];
                    auxI[k-1] = Iprev[N-1];
                    auxD[k-1] = y3;

                    Iprev[N-1] = _mm512_fmadd_pd(Mprev[N-1],qualin, _mm512_mul_pd(Iprev[N-1],one0));
                    Mprev[N-1] = M;                
                }

                for(int h = 0; h < N; h++) finalf = _mm512_add_pd(finalf, _mm512_add_pd(Mprev[h], Iprev[h]));
            }

            _mm512_storeu_pd((finalsum + seqidx), finalf);
        }
        
        
    }
    
    exec_time += omp_get_wtime();

    return exec_time;
}

template <int N, int P> 
double pairhmm_fp64_avx512_v3(unsigned long long* rslen, unsigned long long* haplen, unsigned long long* hap, unsigned long long* read, unsigned long long* q, unsigned long long* i, unsigned long long* d, double* qualfp64, double* finalsum, int seqsize){
    double exec_time = 0;

    omp_set_num_threads(min(seqsize/8, 192));
    
    exec_time = -omp_get_wtime();
    #pragma omp parallel
    {

        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        __m512d* auxM = (__m512d*)aligned_alloc(64, N*P*sizeof(__m512d));
        __m512d* auxI = (__m512d*)aligned_alloc(64, N*P*sizeof(__m512d));
        __m512d* auxD = (__m512d*)aligned_alloc(64, N*P*sizeof(__m512d));

        for(int seqidx = 8*tid; seqidx < seqsize; seqidx += 8*nthreads){
            
            __m512d Mprev[N];
            __m512d Iprev[N];
            __m512i _hap[N];

            __m512d M = _mm512_set1_pd(0);  
            __m512d y1 = _mm512_set1_pd(0); 
            __m512d y2 = _mm512_set1_pd(0); 
            __m512d y3 = _mm512_set1_pd(0);
            __m512d one;
            __m512d one0 = _mm512_set1_pd(0.1);
            __m512d nine = _mm512_set1_pd(0.9);
            __m512d third;
            __m512d finalf = _mm512_set1_pd(0);
            __m512d d0 = _mm512_set1_pd(0);
            __m512d tmm = _mm512_set1_pd(0);
            __m512d sigma_aux = _mm512_set1_pd(0);
            __m512d qual = _mm512_set1_pd(0);
            __m512d qualin = _mm512_set1_pd(0);
            __mmask8 mask;
            __m512i rs0;
            __m512i q0;
            __m512i in;
            __m512i d1;
            
            const unsigned long long rsize = -rslen[seqidx] + rslen[seqidx + 1];
            const unsigned long long hsize = -haplen[seqidx] + haplen[seqidx + 1];
            const unsigned long long ridx = rslen[seqidx];
            const unsigned long long hidx = haplen[seqidx];

            for(int p = 0; p < P; p++){
            
                for(int h = 0; h < N; h++) _hap[h] = _mm512_load_si512((__m512i const*)(hap + hidx + p*N*8 + h*8));
                
                rs0 = _mm512_load_si512((__m512i const*)(read + ridx));
                q0 = _mm512_load_si512((__m512i const*)(q + ridx));
                d1 = _mm512_load_si512((__m512i const*)(d + ridx));
                qual = _mm512_i64gather_pd(q0, qualfp64, 8);
                third = _mm512_mul_pd(_mm512_set1_pd(0.3333333333333333333333333333333333),qual);
                one = _mm512_sub_pd(_mm512_set1_pd(1),qual);
                
                for(int h = 0; h < N; h++){
                    mask = _mm512_cmpeq_epi64_mask(rs0, _hap[h]);
                    sigma_aux = _mm512_mask_blend_pd(mask, third, one);
                    Mprev[h] = _mm512_mul_pd(sigma_aux, _mm512_mul_pd(nine, _mm512_div_pd(_mm512_set1_pd(ldexp(1.0, 1020.0)), _mm512_set1_pd(hsize))));
                    Iprev[h] = _mm512_set1_pd(0);
                }  

                for(int k = 1; k < rsize; k++){
                    rs0 = _mm512_load_si512((__m512i const*)(read + ridx + k*8));
                    q0 = _mm512_load_si512((__m512i const*)(q + ridx + k*8));
                    in = _mm512_load_si512((__m512i const*)(i + ridx + k*8));
                    d0 = _mm512_mul_pd(nine,_mm512_i64gather_pd(d1, qualfp64, 8));
                    d1 = _mm512_load_si512((__m512i const*)(d + ridx + k*8));

                    mask = _mm512_cmpeq_epi64_mask(rs0, _hap[0]);
                    qual = _mm512_i64gather_pd(q0, qualfp64, 8);
                    qualin = _mm512_i64gather_pd(in, qualfp64, 8);
                    third = _mm512_mul_pd(_mm512_set1_pd(0.3333333333333333333333333333333333),qual);
                    one = _mm512_sub_pd(_mm512_set1_pd(1),qual);
                    sigma_aux = _mm512_mask_blend_pd(mask, third, one);
                    tmm = _mm512_sub_pd(_mm512_sub_pd(_mm512_set1_pd(1), _mm512_i64gather_pd(d1, qualfp64, 8)), qualin);

                    M = _mm512_mul_pd(sigma_aux, _mm512_add_pd(_mm512_add_pd(_mm512_mul_pd(auxM[k-1],tmm), _mm512_mul_pd(nine,auxI[k-1])), _mm512_mul_pd(d0,auxD[k-1])));

                    y3 = _mm512_fmadd_pd(auxD[k-1],one0,auxM[k-1]);

                    for(int h = 1; h < N; h++){
                        mask = _mm512_cmpeq_epi64_mask(rs0, _hap[h]);
                        sigma_aux = _mm512_mask_blend_pd(mask, third, one);

                        y2 = _mm512_mul_pd(sigma_aux, _mm512_add_pd(_mm512_add_pd(_mm512_mul_pd(Mprev[h-1],tmm), _mm512_mul_pd(nine,Iprev[h-1])), _mm512_mul_pd(d0,y3)));
                        y3 = _mm512_fmadd_pd(y3,one0,Mprev[h-1]);    

                        Iprev[h-1] = _mm512_fmadd_pd(Mprev[h-1],qualin, _mm512_mul_pd(Iprev[h-1],one0));
                        Mprev[h-1] = M;

                        M = y2;      
                    }

                    auxM[k-1] = Mprev[N-1];
                    auxI[k-1] = Iprev[N-1];
                    auxD[k-1] = y3;

                    Iprev[N-1] = _mm512_fmadd_pd(Mprev[N-1],qualin, _mm512_mul_pd(Iprev[N-1],one0));
                    Mprev[N-1] = M;                
                }

                for(int h = 0; h < N; h++) finalf = _mm512_add_pd(finalf, _mm512_add_pd(Mprev[h], Iprev[h]));
            }

            _mm512_storeu_pd((finalsum + seqidx), finalf);
        }

        free(auxM);
        free(auxI);
        free(auxD);
    }
    
    exec_time += omp_get_wtime();

    return exec_time;
}


double pairhmm_benchmark_fp64_avx512(unsigned long long* rslen, unsigned long long* haplen, unsigned long long* hap, unsigned long long* rs, unsigned long long* q, unsigned long long* i, unsigned long long* d, double* qual, double* finalsum, int maxsize, int seq_size){

    double total_time = 0;

    if(maxsize == 32) total_time = pairhmm_fp64_avx512<32>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 64) total_time = pairhmm_fp64_avx512<64>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 128) total_time = pairhmm_fp64_avx512_v2<64,2>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 256) total_time = pairhmm_fp64_avx512_v2<64,4>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 512) total_time = pairhmm_fp64_avx512_v2<64,8>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 1024) total_time = pairhmm_fp64_avx512_v2<64,16>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 2048) total_time = pairhmm_fp64_avx512_v2<64,32>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 4096) total_time = pairhmm_fp64_avx512_v2<64,64>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 8192) total_time = pairhmm_fp64_avx512_v2<64,128>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 16384) total_time = pairhmm_fp64_avx512_v2<64,256>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 32768) total_time = pairhmm_fp64_avx512_v2<64,512>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 65536) total_time = pairhmm_fp64_avx512_v3<64,1024>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 131072) total_time = pairhmm_fp64_avx512_v3<64,2048>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);

    return total_time;
}

template <int N> 
double pairhmm_fp32_avx512(unsigned long long* rslen, unsigned long long* haplen, unsigned int* hap, unsigned int* read, unsigned int* q, unsigned int* i, unsigned int* d, float* qualfp32, float* finalsum, int seqsize){
    double exec_time = 0;

    omp_set_num_threads(min(seqsize/16, 192));
    
    exec_time = -omp_get_wtime();
    #pragma omp parallel
    {
        
        
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        for(int seqidx = 16*tid; seqidx < seqsize; seqidx += 16*nthreads){
            
            __m512 Mprev[N];
            __m512 Iprev[N];
            __m512i _hap[N];

            __m512 M = _mm512_set1_ps(0.0f);  
            __m512 y1 = _mm512_set1_ps(0.0f); 
            __m512 y2 = _mm512_set1_ps(0.0f); 
            __m512 y3 = _mm512_set1_ps(0.0f);
            __m512 one;
            __m512 one0 = _mm512_set1_ps(0.1f);
            __m512 nine = _mm512_set1_ps(0.9f);
            __m512 third;
            __m512 d0 = _mm512_set1_ps(0.0f);
            __m512 tmm = _mm512_set1_ps(0.0f);
            __m512 sigma_aux = _mm512_set1_ps(0.0f);
            __m512 qual = _mm512_set1_ps(0.0f);
            __m512 qualin = _mm512_set1_ps(0.0f);
            __mmask16 mask;
            __m512i rs0;
            __m512i q0;
            __m512i in;
            __m512i d1;
            
            const unsigned long long rsize = -rslen[seqidx] + rslen[seqidx + 1];
            const unsigned long long hsize = -haplen[seqidx] + haplen[seqidx + 1];
            const unsigned long long ridx = rslen[seqidx];
            const unsigned long long hidx = haplen[seqidx];
            
            for(int h = 0; h < N; h++) _hap[h] = _mm512_load_si512((__m512i const*)(hap + hidx + h*16));
            
            rs0 = _mm512_load_si512((__m512i const*)(read + ridx));
            q0 = _mm512_load_si512((__m512i const*)(q + ridx));
            d1 = _mm512_load_si512((__m512i const*)(d + ridx));
            qual = _mm512_i32gather_ps(q0, qualfp32, 4);
            third = _mm512_mul_ps(_mm512_set1_ps(0.3333333333333333333333333333333333f),qual);
            one = _mm512_sub_ps(_mm512_set1_ps(1.0f),qual);
            
            for(int h = 0; h < N; h++){
                mask = _mm512_cmpeq_epi32_mask(rs0, _hap[h]);
                sigma_aux = _mm512_mask_blend_ps(mask, third, one);
                Mprev[h] = _mm512_mul_ps(sigma_aux, _mm512_mul_ps(nine, _mm512_div_ps(_mm512_set1_ps(ldexpf(1.f, 120.f)),_mm512_set1_ps(hsize))));
                Iprev[h] = _mm512_set1_ps(0.0f);
            }  

            for(int k = 1; k < rsize; k++){
                rs0 = _mm512_load_si512((__m512i const*)(read + ridx + k*16));
                q0 = _mm512_load_si512((__m512i const*)(q + ridx + k*16));
                in = _mm512_load_si512((__m512i const*)(i + ridx + k*16));
                d0 = _mm512_mul_ps(nine,_mm512_i32gather_ps(d1, qualfp32, 4));
                d1 = _mm512_load_si512((__m512i const*)(d + ridx + k*16));

                mask = _mm512_cmpeq_epi32_mask(rs0, _hap[0]);
                qual = _mm512_i32gather_ps(q0, qualfp32, 4);
                qualin = _mm512_i32gather_ps(in, qualfp32, 4);
                third = _mm512_mul_ps(_mm512_set1_ps(0.3333333333333333333333333333333333f),qual);
                one = _mm512_sub_ps(_mm512_set1_ps(1.0f),qual);

                sigma_aux = _mm512_mask_blend_ps(mask, third, one);
                tmm = _mm512_sub_ps(_mm512_sub_ps(_mm512_set1_ps(1.0f), _mm512_i32gather_ps(d1, qualfp32, 4)), qualin);
                y2 = _mm512_set1_ps(0.0f);
                y3 = _mm512_set1_ps(0.0f);
                M = y2;

                for(int h = 1; h < N; h++){
                    mask = _mm512_cmpeq_epi32_mask(rs0, _hap[h]);
                    sigma_aux = _mm512_mask_blend_ps(mask, third, one);

                    y2 = _mm512_mul_ps(sigma_aux, _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(Mprev[h-1],tmm), _mm512_mul_ps(nine,Iprev[h-1])), _mm512_mul_ps(d0,y3)));
                    y3 = _mm512_fmadd_ps(y3,one0,Mprev[h-1]);    

                    Iprev[h-1] = _mm512_fmadd_ps(Mprev[h-1],qualin, _mm512_mul_ps(Iprev[h-1],one0));
                    Mprev[h-1] = M;

                    M = y2;      
                }    

                Iprev[N-1] = _mm512_fmadd_ps(Mprev[N-1],qualin, _mm512_mul_ps(Iprev[N-1],one0));
                Mprev[N-1] = M;
            }

            y1 = _mm512_set1_ps(0.0f);

            for(int h = 0; h < N; h++) y1 = _mm512_add_ps(y1, _mm512_add_ps(Mprev[h], Iprev[h]));

            _mm512_storeu_ps((finalsum + seqidx), y1);
        }
        
        
    }
    
    exec_time += omp_get_wtime();

    return exec_time;
}

template <int N, int P> 
double pairhmm_fp32_avx512_v2(unsigned long long* rslen, unsigned long long* haplen, unsigned int* hap, unsigned int* read, unsigned int* q, unsigned int* i, unsigned int* d, float* qualfp32, float* finalsum, int seqsize){
    double exec_time = 0;

    omp_set_num_threads(min(seqsize/16, 192));
    
    exec_time = -omp_get_wtime();
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        for(int seqidx = 16*tid; seqidx < seqsize; seqidx += 16*nthreads){
            
            __m512 Mprev[N];
            __m512 Iprev[N];
            __m512i _hap[N];
            __m512 auxM[N*P] = {_mm512_set1_ps(0.0f)};
            __m512 auxI[N*P] = {_mm512_set1_ps(0.0f)};
            __m512 auxD[N*P] = {_mm512_set1_ps(0.0f)};

            __m512 M = _mm512_set1_ps(0.0f);  
            __m512 y1 = _mm512_set1_ps(0.0f); 
            __m512 y2 = _mm512_set1_ps(0.0f); 
            __m512 y3 = _mm512_set1_ps(0.0f);
            __m512 one;
            __m512 one0 = _mm512_set1_ps(0.1f);
            __m512 nine = _mm512_set1_ps(0.9f);
            __m512 third;
            __m512 d0 = _mm512_set1_ps(0.0f);
            __m512 finalf = _mm512_set1_ps(0.0f);
            __m512 tmm = _mm512_set1_ps(0.0f);
            __m512 sigma_aux = _mm512_set1_ps(0.0f);
            __m512 qual = _mm512_set1_ps(0.0f);
            __m512 qualin = _mm512_set1_ps(0.0f);
            __mmask16 mask;
            __m512i rs0;
            __m512i q0;
            __m512i in;
            __m512i d1;
            
            const unsigned long long rsize = -rslen[seqidx] + rslen[seqidx + 1];
            const unsigned long long hsize = -haplen[seqidx] + haplen[seqidx + 1];
            const unsigned long long ridx = rslen[seqidx];
            const unsigned long long hidx = haplen[seqidx];

            for(int p = 0; p < P; p++){
            
                for(int h = 0; h < N; h++) _hap[h] = _mm512_load_si512((__m512i const*)(hap + hidx + p*N*16 + h*16));
                
                rs0 = _mm512_load_si512((__m512i const*)(read + ridx));
                q0 = _mm512_load_si512((__m512i const*)(q + ridx));
                d1 = _mm512_load_si512((__m512i const*)(d + ridx));
                qual = _mm512_i32gather_ps(q0, qualfp32, 4);
                third = _mm512_mul_ps(_mm512_set1_ps(0.3333333333333333333333333333333333f),qual);
                one = _mm512_sub_ps(_mm512_set1_ps(1.0f),qual);
                
                for(int h = 0; h < N; h++){
                    mask = _mm512_cmpeq_epi32_mask(rs0, _hap[h]);
                    sigma_aux = _mm512_mask_blend_ps(mask, third, one);
                    Mprev[h] = _mm512_mul_ps(sigma_aux, _mm512_mul_ps(nine, _mm512_div_ps(_mm512_set1_ps(ldexpf(1.f, 120.f)),_mm512_set1_ps(hsize))));
                    Iprev[h] = _mm512_set1_ps(0.0f);
                }  

                for(int k = 1; k < rsize; k++){
                    rs0 = _mm512_load_si512((__m512i const*)(read + ridx + k*16));
                    q0 = _mm512_load_si512((__m512i const*)(q + ridx + k*16));
                    in = _mm512_load_si512((__m512i const*)(i + ridx + k*16));
                    d0 = _mm512_mul_ps(nine,_mm512_i32gather_ps(d1, qualfp32, 4));
                    d1 = _mm512_load_si512((__m512i const*)(d + ridx + k*16));

                    mask = _mm512_cmpeq_epi32_mask(rs0, _hap[0]);
                    qual = _mm512_i32gather_ps(q0, qualfp32, 4);
                    qualin = _mm512_i32gather_ps(in, qualfp32, 4);
                    third = _mm512_mul_ps(_mm512_set1_ps(0.3333333333333333333333333333333333f),qual);
                    one = _mm512_sub_ps(_mm512_set1_ps(1.0f),qual);

                    sigma_aux = _mm512_mask_blend_ps(mask, third, one);
                    tmm = _mm512_sub_ps(_mm512_sub_ps(_mm512_set1_ps(1.0f), _mm512_i32gather_ps(d1, qualfp32, 4)), qualin);
                    M = _mm512_mul_ps(sigma_aux, _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(auxM[k-1],tmm), _mm512_mul_ps(nine,auxI[k-1])), _mm512_mul_ps(d0,auxD[k-1])));

                    y3 = _mm512_fmadd_ps(auxD[k-1],one0,auxM[k-1]);

                    for(int h = 1; h < N; h++){
                        mask = _mm512_cmpeq_epi32_mask(rs0, _hap[h]);
                        sigma_aux = _mm512_mask_blend_ps(mask, third, one);

                        y2 = _mm512_mul_ps(sigma_aux, _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(Mprev[h-1],tmm), _mm512_mul_ps(nine,Iprev[h-1])), _mm512_mul_ps(d0,y3)));
                        y3 = _mm512_fmadd_ps(y3,one0,Mprev[h-1]);    

                        Iprev[h-1] = _mm512_fmadd_ps(Mprev[h-1],qualin, _mm512_mul_ps(Iprev[h-1],one0));
                        Mprev[h-1] = M;

                        M = y2;      
                    }    

                    auxM[k-1] = Mprev[N-1];
                    auxI[k-1] = Iprev[N-1];
                    auxD[k-1] = y3;

                    Iprev[N-1] = _mm512_fmadd_ps(Mprev[N-1],qualin, _mm512_mul_ps(Iprev[N-1],one0));
                    Mprev[N-1] = M;                
                }

                for(int h = 0; h < N; h++) finalf = _mm512_add_ps(finalf, _mm512_add_ps(Mprev[h], Iprev[h]));
            }

            _mm512_storeu_ps((finalsum + seqidx), finalf);
        }
        
        
    }
    
    exec_time += omp_get_wtime();

    return exec_time;
}

template <int N, int P> 
double pairhmm_fp32_avx512_v3(unsigned long long* rslen, unsigned long long* haplen, unsigned int* hap, unsigned int* read, unsigned int* q, unsigned int* i, unsigned int* d, float* qualfp32, float* finalsum, int seqsize){
    double exec_time = 0;

    omp_set_num_threads(min(seqsize/16, 192));
    
    exec_time = -omp_get_wtime();
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        __m512* auxM = (__m512*)aligned_alloc(64, N*P*sizeof(__m512));
        __m512* auxI = (__m512*)aligned_alloc(64, N*P*sizeof(__m512));
        __m512* auxD = (__m512*)aligned_alloc(64, N*P*sizeof(__m512));

        for(int seqidx = 16*tid; seqidx < seqsize; seqidx += 16*nthreads){
            
            __m512 Mprev[N];
            __m512 Iprev[N];
            __m512i _hap[N];

            __m512 M = _mm512_set1_ps(0.0f);  
            __m512 y1 = _mm512_set1_ps(0.0f); 
            __m512 y2 = _mm512_set1_ps(0.0f); 
            __m512 y3 = _mm512_set1_ps(0.0f);
            __m512 one;
            __m512 one0 = _mm512_set1_ps(0.1f);
            __m512 nine = _mm512_set1_ps(0.9f);
            __m512 third;
            __m512 d0 = _mm512_set1_ps(0.0f);
            __m512 finalf = _mm512_set1_ps(0.0f);
            __m512 tmm = _mm512_set1_ps(0.0f);
            __m512 sigma_aux = _mm512_set1_ps(0.0f);
            __m512 qual = _mm512_set1_ps(0.0f);
            __m512 qualin = _mm512_set1_ps(0.0f);
            __mmask16 mask;
            __m512i rs0;
            __m512i q0;
            __m512i in;
            __m512i d1;
            
            const unsigned long long rsize = -rslen[seqidx] + rslen[seqidx + 1];
            const unsigned long long hsize = -haplen[seqidx] + haplen[seqidx + 1];
            const unsigned long long ridx = rslen[seqidx];
            const unsigned long long hidx = haplen[seqidx];

            for(int p = 0; p < P; p++){
            
                for(int h = 0; h < N; h++) _hap[h] = _mm512_load_si512((__m512i const*)(hap + hidx + p*N*16 + h*16));
                
                rs0 = _mm512_load_si512((__m512i const*)(read + ridx));
                q0 = _mm512_load_si512((__m512i const*)(q + ridx));
                d1 = _mm512_load_si512((__m512i const*)(d + ridx));
                qual = _mm512_i32gather_ps(q0, qualfp32, 4);
                third = _mm512_mul_ps(_mm512_set1_ps(0.3333333333333333333333333333333333f),qual);
                one = _mm512_sub_ps(_mm512_set1_ps(1.0f),qual);
                auxM[0] = auxI[0] = auxD[0] = _mm512_set1_ps(0.0f);
                
                for(int h = 0; h < N; h++){
                    mask = _mm512_cmpeq_epi32_mask(rs0, _hap[h]);
                    sigma_aux = _mm512_mask_blend_ps(mask, third, one);
                    Mprev[h] = _mm512_mul_ps(sigma_aux, _mm512_mul_ps(nine, _mm512_div_ps(_mm512_set1_ps(ldexpf(1.f, 120.f)),_mm512_set1_ps(hsize))));
                    Iprev[h] = _mm512_set1_ps(0.0f);
                }  

                for(int k = 1; k < rsize; k++){
                    rs0 = _mm512_load_si512((__m512i const*)(read + ridx + k*16));
                    q0 = _mm512_load_si512((__m512i const*)(q + ridx + k*16));
                    in = _mm512_load_si512((__m512i const*)(i + ridx + k*16));
                    d0 = _mm512_mul_ps(nine,_mm512_i32gather_ps(d1, qualfp32, 4));
                    d1 = _mm512_load_si512((__m512i const*)(d + ridx + k*16));

                    mask = _mm512_cmpeq_epi32_mask(rs0, _hap[0]);
                    qual = _mm512_i32gather_ps(q0, qualfp32, 4);
                    qualin = _mm512_i32gather_ps(in, qualfp32, 4);
                    third = _mm512_mul_ps(_mm512_set1_ps(0.3333333333333333333333333333333333f),qual);
                    one = _mm512_sub_ps(_mm512_set1_ps(1.0f),qual);

                    sigma_aux = _mm512_mask_blend_ps(mask, third, one);
                    tmm = _mm512_sub_ps(_mm512_sub_ps(_mm512_set1_ps(1.0f), _mm512_i32gather_ps(d1, qualfp32, 4)), qualin);
                    M = _mm512_mul_ps(sigma_aux, _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(auxM[k-1],tmm), _mm512_mul_ps(nine,auxI[k-1])), _mm512_mul_ps(d0,auxD[k-1])));

                    y3 = _mm512_fmadd_ps(auxD[k-1],one0,auxM[k-1]);

                    for(int h = 1; h < N; h++){
                        mask = _mm512_cmpeq_epi32_mask(rs0, _hap[h]);
                        sigma_aux = _mm512_mask_blend_ps(mask, third, one);

                        y2 = _mm512_mul_ps(sigma_aux, _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(Mprev[h-1],tmm), _mm512_mul_ps(nine,Iprev[h-1])), _mm512_mul_ps(d0,y3)));
                        y3 = _mm512_fmadd_ps(y3,one0,Mprev[h-1]);    

                        Iprev[h-1] = _mm512_fmadd_ps(Mprev[h-1],qualin, _mm512_mul_ps(Iprev[h-1],one0));
                        Mprev[h-1] = M;

                        M = y2;      
                    }    

                    auxM[k-1] = Mprev[N-1];
                    auxI[k-1] = Iprev[N-1];
                    auxD[k-1] = y3;

                    Iprev[N-1] = _mm512_fmadd_ps(Mprev[N-1],qualin, _mm512_mul_ps(Iprev[N-1],one0));
                    Mprev[N-1] = M;                
                }

                for(int h = 0; h < N; h++) finalf = _mm512_add_ps(finalf, _mm512_add_ps(Mprev[h], Iprev[h]));
            }

            _mm512_storeu_ps((finalsum + seqidx), finalf);
        }

        free(auxM);
        free(auxI);
        free(auxD);
    }
    
    exec_time += omp_get_wtime();

    return exec_time;
}

double pairhmm_benchmark_fp32_avx512(unsigned long long* rslen, unsigned long long* haplen, unsigned int* hap, unsigned int* rs, unsigned int* q, unsigned int* i, unsigned int* d, float* qual, float* finalsum, int maxsize, int seq_size){

    double total_time = 0;

    if(maxsize == 32) total_time = pairhmm_fp32_avx512<32>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 64) total_time = pairhmm_fp32_avx512<64>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 128) total_time = pairhmm_fp32_avx512_v2<64,2>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 256) total_time = pairhmm_fp32_avx512_v2<64,4>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 512) total_time = pairhmm_fp32_avx512_v2<64,8>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 1024) total_time = pairhmm_fp32_avx512_v2<64,16>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 2048) total_time = pairhmm_fp32_avx512_v2<64,32>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 4096) total_time = pairhmm_fp32_avx512_v2<64,64>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 8192) total_time = pairhmm_fp32_avx512_v2<64,128>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 16384) total_time = pairhmm_fp32_avx512_v2<64,256>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 32768) total_time = pairhmm_fp32_avx512_v2<64,512>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 65536) total_time = pairhmm_fp32_avx512_v3<64,1024>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);
    if(maxsize == 131072) total_time = pairhmm_fp32_avx512_v3<64,2048>( rslen, haplen, hap, rs, q, i, d, qual, finalsum, seq_size);

    return total_time;
}
