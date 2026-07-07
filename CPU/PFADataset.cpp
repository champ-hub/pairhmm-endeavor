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
double pairhmm_fp64_dataset(unsigned long long* rslen, unsigned long long* haplen, unsigned long long* hap, unsigned long long* read, __m256i* rslen4, __m256i* haplen4, unsigned long long* q, unsigned long long* i, unsigned long long* d, double* qualfp64, double* finalsum, unsigned int seqsize, int countIdx, unsigned int* seqCounts){
    double exec_time = 0;

    omp_set_num_threads(min((int)seqsize/4, 192));
    //omp_set_num_threads(min(1, 192));
    
    exec_time = -omp_get_wtime();
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        for(int s = tid; s < seqsize/4; s += nthreads){

            int seqidx = seqCounts[countIdx + s];
            
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
            __m256d hsize = _mm256_set1_pd(0);
            __m256i rs0;
            __m256i q0;
            __m256i in;
            __m256i d1;

            __m256d maskH;
            __m256d maskR;
            __m256i idxR;
            __m256i idxH;
            __m256d sol = _mm256_set1_pd(0);
            __m256d solfinal = _mm256_set1_pd(0);
            __m256i r4;
            __m256i h4;
            
            const unsigned long long rsize = -rslen[seqidx] + rslen[seqidx + 1];
            const unsigned long long ridx = rslen[seqidx];
            const unsigned long long hidx = haplen[seqidx];

            h4 = haplen4[seqidx];
            r4 = rslen4[seqidx];

            hsize = _mm256_set_pd(_mm256_extract_epi64(h4, 0), _mm256_extract_epi64(h4, 1), _mm256_extract_epi64(h4, 2), _mm256_extract_epi64(h4, 3));

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
                Mprev[h] = _mm256_mul_pd(sigma_aux, _mm256_mul_pd(nine, _mm256_div_pd(_mm256_set1_pd(ldexp(1.0, 1020.0)), hsize)));
                Iprev[h] = _mm256_set1_pd(0);
            }  
            
            for(int k = 1; k < N; k++){
                idxR = _mm256_set1_epi64x(k+1);
                maskR = _mm256_castsi256_pd(_mm256_cmpeq_epi64(idxR, r4));
                
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
                    idxH = _mm256_set1_epi64x(h-1);
                    maskH = _mm256_castsi256_pd(_mm256_cmpgt_epi64(h4, idxH));

                    mask = _mm256_castsi256_pd(_mm256_cmpeq_epi64(rs0, _hap[h]));
                    sigma_aux = _mm256_blendv_pd(third, one, mask);

                    y2 = _mm256_mul_pd(sigma_aux, _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(Mprev[h-1],tmm), _mm256_mul_pd(nine,Iprev[h-1])), _mm256_mul_pd(d0,y3)));
                    y3 = _mm256_fmadd_pd(y3,one0,Mprev[h-1]);    

                    Iprev[h-1] = _mm256_fmadd_pd(Mprev[h-1],qualin, _mm256_mul_pd(Iprev[h-1],one0));
                    Mprev[h-1] = M;

                    solfinal = _mm256_add_pd(solfinal, _mm256_and_pd(maskR, _mm256_and_pd(maskH, _mm256_add_pd(Mprev[h-1], Iprev[h-1]) ) ) );

                    M = y2;      
                }

                idxH = _mm256_set1_epi64x(N-1);
                maskH = _mm256_castsi256_pd(_mm256_cmpgt_epi64(h4, idxH));

                Iprev[N-1] = _mm256_fmadd_pd(Mprev[N-1],qualin, _mm256_mul_pd(Iprev[N-1],one0));
                Mprev[N-1] = M;

                solfinal = _mm256_add_pd(solfinal, _mm256_and_pd(maskR, _mm256_and_pd(maskH, _mm256_add_pd(Mprev[N-1], Iprev[N-1]))));             
            }

            _mm256_storeu_pd((finalsum + 4*seqidx), solfinal);
        }     
    }
    
    exec_time += omp_get_wtime();

    return exec_time;
}


double pairhmm_process_fp64(unsigned long long* rslen, unsigned long long* haplen, unsigned long long* hap, unsigned long long* rs, __m256i* rslen4, __m256i* haplen4, unsigned long long* q, unsigned long long* i, unsigned long long* d, double* qual, double* finalsum, unsigned int* seq_size, unsigned int* seqCounts, int batchsize){

    double total_time = 0;

    if(seq_size[0] != 0) total_time += pairhmm_fp64_dataset<32>( rslen, haplen, hap, rs, rslen4, haplen4, q, i, d, qual, finalsum, seq_size[0], 0, seqCounts);
    if(seq_size[1] != 0) total_time += pairhmm_fp64_dataset<64>( rslen, haplen, hap, rs, rslen4, haplen4, q, i, d, qual, finalsum, seq_size[1], (batchsize/4 + 1), seqCounts);
    if(seq_size[2] != 0) total_time += pairhmm_fp64_dataset<96>( rslen, haplen, hap, rs, rslen4, haplen4, q, i, d, qual, finalsum, seq_size[2], 2*(batchsize/4 + 1), seqCounts);
    if(seq_size[3] != 0) total_time += pairhmm_fp64_dataset<128>( rslen, haplen, hap, rs, rslen4, haplen4, q, i, d, qual, finalsum, seq_size[3], 3*(batchsize/4 + 1), seqCounts);
    if(seq_size[4] != 0) total_time += pairhmm_fp64_dataset<160>( rslen, haplen, hap, rs, rslen4, haplen4, q, i, d, qual, finalsum, seq_size[4], 4*(batchsize/4 + 1), seqCounts);
    if(seq_size[5] != 0) total_time += pairhmm_fp64_dataset<192>( rslen, haplen, hap, rs, rslen4, haplen4, q, i, d, qual, finalsum, seq_size[5], 5*(batchsize/4 + 1), seqCounts);
    if(seq_size[6] != 0) total_time += pairhmm_fp64_dataset<224>( rslen, haplen, hap, rs, rslen4, haplen4, q, i, d, qual, finalsum, seq_size[6], 6*(batchsize/4 + 1), seqCounts);
    if(seq_size[7] != 0) total_time += pairhmm_fp64_dataset<256>( rslen, haplen, hap, rs, rslen4, haplen4, q, i, d, qual, finalsum, seq_size[7], 7*(batchsize/4 + 1), seqCounts);
    if(seq_size[8] != 0) total_time += pairhmm_fp64_dataset<288>( rslen, haplen, hap, rs, rslen4, haplen4, q, i, d, qual, finalsum, seq_size[8], 8*(batchsize/4 + 1), seqCounts);
    if(seq_size[9] != 0) total_time += pairhmm_fp64_dataset<320>( rslen, haplen, hap, rs, rslen4, haplen4, q, i, d, qual, finalsum, seq_size[1], 9*(batchsize/4 + 1), seqCounts);
    if(seq_size[19] != 0) total_time += pairhmm_fp64_dataset<352>( rslen, haplen, hap, rs, rslen4, haplen4, q, i, d, qual, finalsum, seq_size[2], 10*(batchsize/4 + 1), seqCounts);
    if(seq_size[11] != 0) total_time += pairhmm_fp64_dataset<384>( rslen, haplen, hap, rs, rslen4, haplen4, q, i, d, qual, finalsum, seq_size[3], 11*(batchsize/4 + 1), seqCounts);
    if(seq_size[12] != 0) total_time += pairhmm_fp64_dataset<416>( rslen, haplen, hap, rs, rslen4, haplen4, q, i, d, qual, finalsum, seq_size[4], 12*(batchsize/4 + 1), seqCounts);
    if(seq_size[13] != 0) total_time += pairhmm_fp64_dataset<448>( rslen, haplen, hap, rs, rslen4, haplen4, q, i, d, qual, finalsum, seq_size[5], 13*(batchsize/4 + 1), seqCounts);
    if(seq_size[14] != 0) total_time += pairhmm_fp64_dataset<480>( rslen, haplen, hap, rs, rslen4, haplen4, q, i, d, qual, finalsum, seq_size[6], 14*(batchsize/4 + 1), seqCounts);
    if(seq_size[15] != 0) total_time += pairhmm_fp64_dataset<512>( rslen, haplen, hap, rs, rslen4, haplen4, q, i, d, qual, finalsum, seq_size[7], 15*(batchsize/4 + 1), seqCounts);



    return total_time;
}



template <int N> 
double pairhmm_fp32_dataset(unsigned long long* rslen, unsigned long long* haplen, unsigned int* hap, unsigned int* read, __m256i* rslen8, __m256i* haplen8, unsigned int* q, unsigned int* i, unsigned int* d, float* qualfp32, float* finalsum, unsigned int seqsize, int countIdx, unsigned int* seqCounts){
    double exec_time = 0;

    omp_set_num_threads(min((int)seqsize/8, 192));
    
    exec_time = -omp_get_wtime();
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        for(int s = tid; s < seqsize/8; s += nthreads){

            int seqidx = seqCounts[countIdx + s];
            
            __m256 Mprev[N];
            __m256 Iprev[N];
            __m256i _hap[N];

            __m256 M = _mm256_set1_ps(0.0f);  
            __m256 y1 = _mm256_set1_ps(0.0f); 
            __m256 y2 = _mm256_set1_ps(0.0f); 
            __m256 y3 = _mm256_set1_ps(0.0f);
            __m256 one;
            __m256 one0 = _mm256_set1_ps(0.1f);
            __m256 nine = _mm256_set1_ps(0.9f);
            __m256 third;
            __m256 d0 = _mm256_set1_ps(0.0f);
            __m256 tmm = _mm256_set1_ps(0.0f);
            __m256 sigma_aux = _mm256_set1_ps(0.0f);
            __m256 qual = _mm256_set1_ps(0.0f);
            __m256 qualin = _mm256_set1_ps(0.0f);
            __m256 mask = _mm256_set1_ps(0.0f);
            __m256 hsize = _mm256_set1_ps(0.0f);
            __m256i rs0;
            __m256i q0;
            __m256i in;
            __m256i d1;

            __m256 maskH;
            __m256 maskR;
            __m256i idxR;
            __m256i idxH;
            __m256 sol = _mm256_set1_ps(0.0f);
            __m256 solfinal = _mm256_set1_ps(0.0f);
            __m256i r4;
            __m256i h4;
            
            const unsigned long long rsize = -rslen[seqidx] + rslen[seqidx + 1];
            const unsigned long long ridx = rslen[seqidx];
            const unsigned long long hidx = haplen[seqidx];

            h4 = haplen8[seqidx];
            r4 = rslen8[seqidx];

            hsize = _mm256_set_ps(_mm256_extract_epi32(h4, 0), _mm256_extract_epi32(h4, 1), _mm256_extract_epi32(h4, 2), _mm256_extract_epi32(h4, 3), _mm256_extract_epi32(h4, 4), _mm256_extract_epi32(h4, 5), _mm256_extract_epi32(h4, 6), _mm256_extract_epi32(h4, 7));

            for(int h = 0; h < N; h++) _hap[h] = _mm256_load_si256((__m256i const*)(hap + hidx + h*8));
            
            rs0 = _mm256_load_si256((__m256i const*)(read + ridx));
            q0 = _mm256_load_si256((__m256i const*)(q + ridx));
            d1 = _mm256_load_si256((__m256i const*)(d + ridx));
            qual = _mm256_i32gather_ps(qualfp32, q0, 4);
            third = _mm256_mul_ps(_mm256_set1_ps(0.3333333333333333333333333333333333f), qual);
            one = _mm256_sub_ps(_mm256_set1_ps(1.0f), qual);

            for(int h = 0; h < N; h++){
                mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(rs0, _hap[h]));
                sigma_aux = _mm256_blendv_ps(third, one, mask);
                Mprev[h] = _mm256_mul_ps(sigma_aux, _mm256_mul_ps(nine, _mm256_div_ps(_mm256_set1_ps(ldexpf(1.f, 120.f)), hsize)));
                Iprev[h] = _mm256_set1_ps(0.0f);
            }  
            
            for(int k = 1; k < N; k++){
                idxR = _mm256_set1_epi32(k+1);
                maskR = _mm256_castsi256_ps(_mm256_cmpeq_epi32(idxR, r4));
                
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
                    idxH = _mm256_set1_epi32(h-1);
                    maskH = _mm256_castsi256_ps(_mm256_cmpgt_epi32(h4, idxH));

                    mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(rs0, _hap[h]));
                    sigma_aux = _mm256_blendv_ps(third, one, mask);

                    y2 = _mm256_mul_ps(sigma_aux, _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(Mprev[h-1],tmm), _mm256_mul_ps(nine,Iprev[h-1])), _mm256_mul_ps(d0,y3)));
                    y3 = _mm256_fmadd_ps(y3,one0,Mprev[h-1]);    

                    Iprev[h-1] = _mm256_fmadd_ps(Mprev[h-1],qualin, _mm256_mul_ps(Iprev[h-1],one0));
                    Mprev[h-1] = M;

                    solfinal = _mm256_add_ps(solfinal, _mm256_and_ps(maskR, _mm256_and_ps(maskH, _mm256_add_ps(Mprev[h-1], Iprev[h-1]) ) ) );

                    M = y2;      
                }

                idxH = _mm256_set1_epi32(N-1);
                maskH = _mm256_castsi256_ps(_mm256_cmpgt_epi32(h4, idxH));

                Iprev[N-1] = _mm256_fmadd_ps(Mprev[N-1],qualin, _mm256_mul_ps(Iprev[N-1],one0));
                Mprev[N-1] = M;

                solfinal = _mm256_add_ps(solfinal, _mm256_and_ps(maskR, _mm256_and_ps(maskH, _mm256_add_ps(Mprev[N-1], Iprev[N-1]))));             
            }

            _mm256_storeu_ps((finalsum + 8*seqidx), solfinal);
        }
    }
    
    exec_time += omp_get_wtime();

    return exec_time;
}



double pairhmm_process_fp32(unsigned long long* rslen, unsigned long long* haplen, unsigned int* hap, unsigned int* rs, __m256i* rslen8, __m256i* haplen8, unsigned int* q, unsigned int* i, unsigned int* d, float* qual, float* finalsum, unsigned int* seq_size, unsigned int* seqCounts, int batchsize){

    double total_time = 0;

    if(seq_size[0] != 0) total_time += pairhmm_fp32_dataset<32>( rslen, haplen, hap, rs, rslen8, haplen8, q, i, d, qual, finalsum, seq_size[0], 0, seqCounts);
    if(seq_size[1] != 0) total_time += pairhmm_fp32_dataset<64>( rslen, haplen, hap, rs, rslen8, haplen8, q, i, d, qual, finalsum, seq_size[1], (batchsize/8 + 1), seqCounts);
    if(seq_size[2] != 0) total_time += pairhmm_fp32_dataset<96>( rslen, haplen, hap, rs, rslen8, haplen8, q, i, d, qual, finalsum, seq_size[2], 2*(batchsize/8 + 1), seqCounts);
    if(seq_size[3] != 0) total_time += pairhmm_fp32_dataset<128>( rslen, haplen, hap, rs, rslen8, haplen8, q, i, d, qual, finalsum, seq_size[3], 3*(batchsize/8 + 1), seqCounts);
    if(seq_size[4] != 0) total_time += pairhmm_fp32_dataset<160>( rslen, haplen, hap, rs, rslen8, haplen8, q, i, d, qual, finalsum, seq_size[4], 4*(batchsize/8 + 1), seqCounts);
    if(seq_size[5] != 0) total_time += pairhmm_fp32_dataset<192>( rslen, haplen, hap, rs, rslen8, haplen8, q, i, d, qual, finalsum, seq_size[5], 5*(batchsize/8 + 1), seqCounts);
    if(seq_size[6] != 0) total_time += pairhmm_fp32_dataset<224>( rslen, haplen, hap, rs, rslen8, haplen8, q, i, d, qual, finalsum, seq_size[6], 6*(batchsize/8 + 1), seqCounts);
    if(seq_size[7] != 0) total_time += pairhmm_fp32_dataset<256>( rslen, haplen, hap, rs, rslen8, haplen8, q, i, d, qual, finalsum, seq_size[7], 7*(batchsize/8 + 1), seqCounts);
    if(seq_size[8] != 0) total_time += pairhmm_fp32_dataset<288>( rslen, haplen, hap, rs, rslen8, haplen8, q, i, d, qual, finalsum, seq_size[8], 8*(batchsize/8 + 1), seqCounts);
    if(seq_size[9] != 0) total_time += pairhmm_fp32_dataset<320>( rslen, haplen, hap, rs, rslen8, haplen8, q, i, d, qual, finalsum, seq_size[9], 9*(batchsize/8 + 1), seqCounts);
    if(seq_size[19] != 0) total_time += pairhmm_fp32_dataset<352>( rslen, haplen, hap, rs, rslen8, haplen8, q, i, d, qual, finalsum, seq_size[10], 10*(batchsize/8 + 1), seqCounts);
    if(seq_size[11] != 0) total_time += pairhmm_fp32_dataset<384>( rslen, haplen, hap, rs, rslen8, haplen8, q, i, d, qual, finalsum, seq_size[11], 11*(batchsize/8 + 1), seqCounts);
    if(seq_size[12] != 0) total_time += pairhmm_fp32_dataset<416>( rslen, haplen, hap, rs, rslen8, haplen8, q, i, d, qual, finalsum, seq_size[12], 12*(batchsize/8 + 1), seqCounts);
    if(seq_size[13] != 0) total_time += pairhmm_fp32_dataset<448>( rslen, haplen, hap, rs, rslen8, haplen8, q, i, d, qual, finalsum, seq_size[13], 13*(batchsize/8 + 1), seqCounts);
    if(seq_size[14] != 0) total_time += pairhmm_fp32_dataset<480>( rslen, haplen, hap, rs, rslen8, haplen8, q, i, d, qual, finalsum, seq_size[14], 14*(batchsize/8 + 1), seqCounts);
    if(seq_size[15] != 0) total_time += pairhmm_fp32_dataset<512>( rslen, haplen, hap, rs, rslen8, haplen8, q, i, d, qual, finalsum, seq_size[15], 15*(batchsize/8 + 1), seqCounts);

    return total_time;
}

//AVX-512

template <int N> 
double pairhmm_fp64_dataset_avx512(unsigned long long* rslen, unsigned long long* haplen, unsigned long long* hap, unsigned long long* read, unsigned long long* q, unsigned long long* i, unsigned long long* d, double* qualfp64, double* finalsum, int seqsize){
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

double pairhmm_process_fp64_avx512(unsigned long long* rslen, unsigned long long* haplen, unsigned long long* hap, unsigned long long* rs, unsigned long long* q, unsigned long long* i, unsigned long long* d, double* qual, double* finalsum, int maxsize, int seq_size){

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

    return total_time;
}

template <int N> 
double pairhmm_fp32_dataset_avx512(unsigned long long* rslen, unsigned long long* haplen, unsigned int* hap, unsigned int* read, __m512i* rslen16, __m512i* haplen16, unsigned int* q, unsigned int* i, unsigned int* d, float* qualfp32, float* finalsum, unsigned int seqsize, int countIdx, unsigned int* seqCounts){
    double exec_time = 0;

    omp_set_num_threads(min((int)seqsize/16, 192));
    
    exec_time = -omp_get_wtime();
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        for(int s = tid; s < seqsize/16; s += nthreads){

            int seqidx = seqCounts[countIdx + s];
            
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
            __m512 hsize;

            __mmask16 maskH;
            __mmask16 maskR;
            __m512i idxR;
            __m512i idxH;
            __m512 sol = _mm512_set1_ps(0.0f);
            __m512 solfinal = _mm512_set1_ps(0.0f);
            __m512i r4;
            __m512i h4;
            
            const unsigned long long rsize = -rslen[seqidx] + rslen[seqidx + 1];
            const unsigned long long ridx = rslen[seqidx];
            const unsigned long long hidx = haplen[seqidx];

            h4 = haplen16[seqidx];
            r4 = rslen16[seqidx];

            __m256i lower = _mm512_extracti32x8_epi32(h4, 0);
            __m256i upper = _mm512_extracti32x8_epi32(h4, 1);

            hsize = _mm512_set_ps(_mm256_extract_epi32(lower, 0), _mm256_extract_epi32(lower, 1), _mm256_extract_epi32(lower, 2), _mm256_extract_epi32(lower, 3), _mm256_extract_epi32(lower, 4), _mm256_extract_epi32(lower, 5), _mm256_extract_epi32(lower, 6), _mm256_extract_epi32(lower, 7),
                                  _mm256_extract_epi32(upper, 0), _mm256_extract_epi32(upper, 1), _mm256_extract_epi32(upper, 2), _mm256_extract_epi32(upper, 3), _mm256_extract_epi32(upper, 4), _mm256_extract_epi32(upper, 5), _mm256_extract_epi32(upper, 6), _mm256_extract_epi32(upper, 7));
            
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
                Mprev[h] = _mm512_mul_ps(sigma_aux, _mm512_mul_ps(nine, _mm512_div_ps(_mm512_set1_ps(ldexpf(1.f, 120.f)), hsize)));
                Iprev[h] = _mm512_set1_ps(0.0f);
            }  

            for(int k = 1; k < N; k++){
                idxR = _mm512_set1_epi32(k+1);
                maskR = _mm512_cmpeq_epi32_mask(idxR, r4);

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
                    idxH = _mm512_set1_epi32(h-1);
                    maskH = _mm512_cmpgt_epi32_mask(h4, idxH);

                    mask = _mm512_cmpeq_epi32_mask(rs0, _hap[h]);
                    sigma_aux = _mm512_mask_blend_ps(mask, third, one);

                    y2 = _mm512_mul_ps(sigma_aux, _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(Mprev[h-1],tmm), _mm512_mul_ps(nine,Iprev[h-1])), _mm512_mul_ps(d0,y3)));
                    y3 = _mm512_fmadd_ps(y3,one0,Mprev[h-1]);    

                    solfinal = _mm512_mask_add_ps(solfinal, maskR & maskH, solfinal, _mm512_add_ps(Mprev[h-1], Iprev[h-1]));

                    M = y2;      
                }

                idxH = _mm512_set1_epi32(N-1);
                maskH = _mm512_cmpgt_epi32_mask(h4, idxH);

                Iprev[N-1] = _mm512_fmadd_ps(Mprev[N-1],qualin, _mm512_mul_ps(Iprev[N-1],one0));
                Mprev[N-1] = M;

                solfinal = _mm512_mask_add_ps(solfinal, maskR & maskH, solfinal,_mm512_add_ps(Mprev[N-1], Iprev[N-1]));
            }

            _mm512_storeu_ps((finalsum + 16*seqidx), solfinal);
        }
    }
    
    exec_time += omp_get_wtime();

    return exec_time;
}

double pairhmm_process_fp32_avx512(unsigned long long* rslen, unsigned long long* haplen, unsigned int* hap, unsigned int* rs, __m512i* rslen16, __m512i* haplen16, unsigned int* q, unsigned int* i, unsigned int* d, float* qual, float* finalsum, unsigned int* seq_size, unsigned int* seqCounts, int batchsize){

    double total_time = 0;

    if(seq_size[0] != 0) total_time += pairhmm_fp32_dataset_avx512<32>( rslen, haplen, hap, rs, rslen16, haplen16, q, i, d, qual, finalsum, seq_size[0], 0, seqCounts);
    if(seq_size[1] != 0) total_time += pairhmm_fp32_dataset_avx512<64>( rslen, haplen, hap, rs, rslen16, haplen16, q, i, d, qual, finalsum, seq_size[1], (batchsize/16 + 1), seqCounts);
    if(seq_size[2] != 0) total_time += pairhmm_fp32_dataset_avx512<96>( rslen, haplen, hap, rs, rslen16, haplen16, q, i, d, qual, finalsum, seq_size[2], 2*(batchsize/16 + 1), seqCounts);
    if(seq_size[3] != 0) total_time += pairhmm_fp32_dataset_avx512<128>( rslen, haplen, hap, rs, rslen16, haplen16, q, i, d, qual, finalsum, seq_size[3], 3*(batchsize/16 + 1), seqCounts);
    if(seq_size[4] != 0) total_time += pairhmm_fp32_dataset_avx512<160>( rslen, haplen, hap, rs, rslen16, haplen16, q, i, d, qual, finalsum, seq_size[4], 4*(batchsize/16 + 1), seqCounts);
    if(seq_size[5] != 0) total_time += pairhmm_fp32_dataset_avx512<192>( rslen, haplen, hap, rs, rslen16, haplen16, q, i, d, qual, finalsum, seq_size[5], 5*(batchsize/16 + 1), seqCounts);
    if(seq_size[6] != 0) total_time += pairhmm_fp32_dataset_avx512<224>( rslen, haplen, hap, rs, rslen16, haplen16, q, i, d, qual, finalsum, seq_size[6], 6*(batchsize/16 + 1), seqCounts);
    if(seq_size[7] != 0) total_time += pairhmm_fp32_dataset_avx512<256>( rslen, haplen, hap, rs, rslen16, haplen16, q, i, d, qual, finalsum, seq_size[7], 7*(batchsize/16 + 1), seqCounts);
    if(seq_size[8] != 0) total_time += pairhmm_fp32_dataset_avx512<288>( rslen, haplen, hap, rs, rslen16, haplen16, q, i, d, qual, finalsum, seq_size[8], 8*(batchsize/16 + 1), seqCounts);
    if(seq_size[9] != 0) total_time += pairhmm_fp32_dataset_avx512<320>( rslen, haplen, hap, rs, rslen16, haplen16, q, i, d, qual, finalsum, seq_size[9], 9*(batchsize/16 + 1), seqCounts);
    if(seq_size[10] != 0) total_time += pairhmm_fp32_dataset_avx512<352>( rslen, haplen, hap, rs, rslen16, haplen16, q, i, d, qual, finalsum, seq_size[10], 10*(batchsize/16 + 1), seqCounts);
    if(seq_size[11] != 0) total_time += pairhmm_fp32_dataset_avx512<384>( rslen, haplen, hap, rs, rslen16, haplen16, q, i, d, qual, finalsum, seq_size[11], 11*(batchsize/16 + 1), seqCounts);
    if(seq_size[12] != 0) total_time += pairhmm_fp32_dataset_avx512<416>( rslen, haplen, hap, rs, rslen16, haplen16, q, i, d, qual, finalsum, seq_size[12], 12*(batchsize/16 + 1), seqCounts);
    if(seq_size[13] != 0) total_time += pairhmm_fp32_dataset_avx512<448>( rslen, haplen, hap, rs, rslen16, haplen16, q, i, d, qual, finalsum, seq_size[13], 13*(batchsize/16 + 1), seqCounts);
    if(seq_size[14] != 0) total_time += pairhmm_fp32_dataset_avx512<480>( rslen, haplen, hap, rs, rslen16, haplen16, q, i, d, qual, finalsum, seq_size[14], 14*(batchsize/16 + 1), seqCounts);
    if(seq_size[15] != 0) total_time += pairhmm_fp32_dataset_avx512<512>( rslen, haplen, hap, rs, rslen16, haplen16, q, i, d, qual, finalsum, seq_size[15], 15*(batchsize/16 + 1), seqCounts);

    if(seq_size[16] != 0) total_time += pairhmm_fp32_dataset_avx512<544>( rslen, haplen, hap, rs, rslen16, haplen16, q, i, d, qual, finalsum, seq_size[16], 16*(batchsize/16 + 1), seqCounts);
    if(seq_size[17] != 0) total_time += pairhmm_fp32_dataset_avx512<576>( rslen, haplen, hap, rs, rslen16, haplen16, q, i, d, qual, finalsum, seq_size[17], 17*(batchsize/16 + 1), seqCounts);
    if(seq_size[18] != 0) total_time += pairhmm_fp32_dataset_avx512<608>( rslen, haplen, hap, rs, rslen16, haplen16, q, i, d, qual, finalsum, seq_size[18], 18*(batchsize/16 + 1), seqCounts);
    if(seq_size[19] != 0) total_time += pairhmm_fp32_dataset_avx512<640>( rslen, haplen, hap, rs, rslen16, haplen16, q, i, d, qual, finalsum, seq_size[19], 19*(batchsize/16 + 1), seqCounts);
    if(seq_size[20] != 0) total_time += pairhmm_fp32_dataset_avx512<672>( rslen, haplen, hap, rs, rslen16, haplen16, q, i, d, qual, finalsum, seq_size[20], 20*(batchsize/16 + 1), seqCounts);
    if(seq_size[21] != 0) total_time += pairhmm_fp32_dataset_avx512<704>( rslen, haplen, hap, rs, rslen16, haplen16, q, i, d, qual, finalsum, seq_size[21], 21*(batchsize/16 + 1), seqCounts);
    if(seq_size[22] != 0) total_time += pairhmm_fp32_dataset_avx512<736>( rslen, haplen, hap, rs, rslen16, haplen16, q, i, d, qual, finalsum, seq_size[22], 22*(batchsize/16 + 1), seqCounts);
    if(seq_size[23] != 0) total_time += pairhmm_fp32_dataset_avx512<768>( rslen, haplen, hap, rs, rslen16, haplen16, q, i, d, qual, finalsum, seq_size[23], 23*(batchsize/16 + 1), seqCounts);
    if(seq_size[24] != 0) total_time += pairhmm_fp32_dataset_avx512<800>( rslen, haplen, hap, rs, rslen16, haplen16, q, i, d, qual, finalsum, seq_size[24], 24*(batchsize/16 + 1), seqCounts);
    if(seq_size[25] != 0) total_time += pairhmm_fp32_dataset_avx512<832>( rslen, haplen, hap, rs, rslen16, haplen16, q, i, d, qual, finalsum, seq_size[25], 25*(batchsize/16 + 1), seqCounts);
    if(seq_size[26] != 0) total_time += pairhmm_fp32_dataset_avx512<864>( rslen, haplen, hap, rs, rslen16, haplen16, q, i, d, qual, finalsum, seq_size[26], 26*(batchsize/16 + 1), seqCounts);
    if(seq_size[27] != 0) total_time += pairhmm_fp32_dataset_avx512<896>( rslen, haplen, hap, rs, rslen16, haplen16, q, i, d, qual, finalsum, seq_size[27], 27*(batchsize/16 + 1), seqCounts);
    if(seq_size[28] != 0) total_time += pairhmm_fp32_dataset_avx512<928>( rslen, haplen, hap, rs, rslen16, haplen16, q, i, d, qual, finalsum, seq_size[28], 28*(batchsize/16 + 1), seqCounts);
    if(seq_size[29] != 0) total_time += pairhmm_fp32_dataset_avx512<960>( rslen, haplen, hap, rs, rslen16, haplen16, q, i, d, qual, finalsum, seq_size[29], 29*(batchsize/16 + 1), seqCounts);
    if(seq_size[30] != 0) total_time += pairhmm_fp32_dataset_avx512<992>( rslen, haplen, hap, rs, rslen16, haplen16, q, i, d, qual, finalsum, seq_size[30], 30*(batchsize/16 + 1), seqCounts);
    if(seq_size[31] != 0) total_time += pairhmm_fp32_dataset_avx512<1024>( rslen, haplen, hap, rs, rslen16, haplen16, q, i, d, qual, finalsum, seq_size[31], 31*(batchsize/16 + 1), seqCounts);

    if(seq_size[32] != 0) total_time += pairhmm_fp32_dataset_avx512<2048>( rslen, haplen, hap, rs, rslen16, haplen16, q, i, d, qual, finalsum, seq_size[32], 32*(batchsize/16 + 1), seqCounts);
    if(seq_size[33] != 0) total_time += pairhmm_fp32_dataset_avx512<4096>( rslen, haplen, hap, rs, rslen16, haplen16, q, i, d, qual, finalsum, seq_size[33], 33*(batchsize/16 + 1), seqCounts);
    if(seq_size[34] != 0) total_time += pairhmm_fp32_dataset_avx512<8192>( rslen, haplen, hap, rs, rslen16, haplen16, q, i, d, qual, finalsum, seq_size[34], 34*(batchsize/16 + 1), seqCounts);
    if(seq_size[35] != 0) total_time += pairhmm_fp32_dataset_avx512<16384>( rslen, haplen, hap, rs, rslen16, haplen16, q, i, d, qual, finalsum, seq_size[35], 35*(batchsize/16 + 1), seqCounts);

    return total_time;
}
