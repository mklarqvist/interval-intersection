#include <iostream>//out streams
#include <random>//random generator (c++11)
#include <chrono>//time (c++11)
#include <cassert>//assert
#include <cstring>//memset
#include <cstdint>//ctypes
#include <algorithm>//sort
#include <vector> //vector

#include "IntervalTree.h"

/*
interval ideas:
* Partition interval set S into ceil(max(S)/M)+1 bins. 
* Set bits in bitmap to true if ANY value overlaps that range
* Possibly duplicate values that overlap multiple bins.
* Perform overlap test in bins.
* Stop if any bin has a matching value.

if left hand is sorted:
* keep pointer to previous hit
* single check to see if the current position still overlap
*/

#include <bitset>

/*------ SIMD definitions --------*/
#if defined(_MSC_VER)
     /* Microsoft C/C++-compatible compiler */
     #include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
     /* GCC-compatible compiler, targeting x86/x86-64 */
     #include <x86intrin.h>
#elif defined(__GNUC__) && defined(__ARM_NEON__)
     /* GCC-compatible compiler, targeting ARM with NEON */
     #include <arm_neon.h>
#elif defined(__GNUC__) && defined(__IWMMXT__)
     /* GCC-compatible compiler, targeting ARM with WMMX */
     #include <mmintrin.h>
#elif (defined(__GNUC__) || defined(__xlC__)) && (defined(__VEC__) || defined(__ALTIVEC__))
     /* XLC or GCC-compatible compiler, targeting PowerPC with VMX/VSX */
     #include <altivec.h>
#elif defined(__GNUC__) && defined(__SPE__)
     /* GCC-compatible compiler, targeting PowerPC with SPE */
     #include <spe.h>
#endif

#if defined(__AVX512F__) && __AVX512F__ == 1
#define SIMD_AVAILABLE  1
#define SIMD_VERSION    6
#define SIMD_ALIGNMENT  64
#elif defined(__AVX2__) && __AVX2__ == 1
#define SIMD_AVAILABLE  1
#define SIMD_VERSION    5
#define SIMD_ALIGNMENT  32
#elif defined(__AVX__) && __AVX__ == 1
#define SIMD_AVAILABLE  1
#define SIMD_VERSION    4
#define SIMD_ALIGNMENT  16
#elif defined(__SSE4_1__) && __SSE4_1__ == 1
#define SIMD_AVAILABLE  1
#define SIMD_VERSION    3
#define SIMD_ALIGNMENT  16
#elif defined(__SSE2__) && __SSE2__ == 1
#define SIMD_AVAILABLE  0 // unsupported version
#define SIMD_VERSION    0
#define SIMD_ALIGNMENT  16
#elif defined(__SSE__) && __SSE__ == 1
#define SIMD_AVAILABLE  0 // unsupported version
#define SIMD_VERSION    0
#define SIMD_ALIGNMENT  16
#else
#define SIMD_AVAILABLE  0
#define SIMD_VERSION    0
#define SIMD_ALIGNMENT  16
#endif

#define DEBUG 0

#ifdef _mm_popcnt_u64
#define PIL_POPCOUNT _mm_popcnt_u64
#else
#define PIL_POPCOUNT __builtin_popcountll
#endif


#ifndef PIL_POPCOUNT_AVX2
#define PIL_POPCOUNT_AVX2(A, B) {                  \
    A += PIL_POPCOUNT(_mm256_extract_epi64(B, 0)); \
    A += PIL_POPCOUNT(_mm256_extract_epi64(B, 1)); \
    A += PIL_POPCOUNT(_mm256_extract_epi64(B, 2)); \
    A += PIL_POPCOUNT(_mm256_extract_epi64(B, 3)); \
}
#endif

/*------ Support functions --------*/

inline void* aligned_malloc(size_t size, size_t align) {
    void* result;
#ifdef _MSC_VER 
    result = _aligned_malloc(size, align);
#else 
     if(posix_memalign(&result, align, size)) result = 0;
#endif
    return result;
}

inline void aligned_free(void* ptr) {
#ifdef _MSC_VER 
      _aligned_free(ptr);
#else 
      free(ptr);
#endif
}

/*------ Support structures --------*/

struct bitmap_helper {
    std::vector<uint32_t> intervals;
};

/*------ Functions --------*/

typedef bool(*overlap_func)(const uint32_t, const ssize_t, const uint32_t*);
typedef bool(*overlap_func_squash)(const uint32_t, const ssize_t,
                                   const uint32_t*, const uint64_t*, 
                                   const uint32_t, const uint32_t, 
                                   const std::vector<bitmap_helper>&);

/**
 * Using only start and end values of two intervals [a0,b0] and [a1,b1] the overlap can be performed as follows:
 * a0 < b1 AND a1 > b0
 * 
 * @param query 
 * @param n 
 * @param ranges 
 * @return true 
 * @return false 
 */
 __attribute__((optimize("no-tree-vectorize")))
bool overlap_scalar_nosimd(const uint32_t query, const ssize_t n, const uint32_t* ranges) {
    uint32_t overlaps = 0;
    for (int i = 0; i < n; i += 2) {
        overlaps += (query < ranges[i+1] && query > ranges[i]);
#if DEBUG == 1
        if ((query < ranges[i+1] && query > ranges[i])) {
            std::cout << query << " overlaps with: " << ranges[i] << "," << ranges[i+1] << std::endl;
        }
#endif
    }

#if DEBUG == 1
    std::cout << "scalar=" << overlaps << std::endl;
#endif
    return overlaps;
}

bool overlap_scalar(const uint32_t query, const ssize_t n, const uint32_t* ranges) {
    uint32_t overlaps = 0;
    for (int i = 0; i < n; i += 2) {
        overlaps += (query < ranges[i+1] && query > ranges[i]);
    }
    return overlaps;
}

bool overlap_scalar_break(const uint32_t query, const ssize_t n, const uint32_t* ranges) {
    uint32_t overlaps = 0;
    for (int i = 0; i < n; i += 2) {
        overlaps += (query < ranges[i+1] && query > ranges[i]);
        if(overlaps) break;
    }

    return overlaps;
}

bool overlap_simd(const uint32_t query, const ssize_t n, const uint32_t* ranges) {
    uint32_t n_cycles = n / (sizeof(__m128i)/sizeof(uint32_t));
    __m128i* vec = (__m128i*)(ranges);
    __m128i q = _mm_set1_epi32(query);

    uint32_t tot = 0;
    for (int i = 0; i < n_cycles; ++i) {
        __m128i lt = _mm_cmplt_epi32(q, vec[i]);
        __m128i gt = _mm_cmpgt_epi32(q, vec[i]);
        __m128i shuffle1 = _mm_shuffle_epi32(lt, _MM_SHUFFLE(2,3,0,1));
        __m128i collapse = _mm_and_si128(shuffle1, gt);
        tot += __builtin_popcountll(_mm_cvtsi128_si64(collapse));
        tot += __builtin_popcountll(_mm_cvtsi128_si64(_mm_srli_si128(collapse, 8)));

#if DEBUG == 1

        uint32_t* b = (uint32_t*)(&lt);
        std::cerr << "LT " << std::bitset<32>(b[0]) << " " << std::bitset<32>(b[1]) << " " << std::bitset<32>(b[2]) << " " << std::bitset<32>(b[3]) << std::endl;
        b = (uint32_t*)(&gt);
        std::cerr << "GT " << std::bitset<32>(b[0]) << " " << std::bitset<32>(b[1]) << " " << std::bitset<32>(b[2]) << " " << std::bitset<32>(b[3]) << std::endl;
        b = (uint32_t*)(&shuffle1);
        std::cerr << "SH " << std::bitset<32>(b[0]) << " " << std::bitset<32>(b[1]) << " " << std::bitset<32>(b[2]) << " " << std::bitset<32>(b[3]) << std::endl;
        std::cerr << std::endl;
        
        if (__builtin_popcountll(_mm_cvtsi128_si64(collapse))) {
            std::cerr << "SIMD: " << query << " overlaps with " << ranges[4*i] << "," << ranges[4*i+1] << std::endl;
        }

        if (__builtin_popcountll(_mm_cvtsi128_si64(_mm_srli_si128(collapse, 8)))) {
            std::cerr << "SIMD: " << query << " overlaps with " << ranges[4*i+2] << "," << ranges[4*i+3] << std::endl;
        }
#endif
    }
#if DEBUG == 1
    std::cout << tot << std::endl;
#endif

    return tot;
}

bool overlap_simd_add(const uint32_t query, const ssize_t n, const uint32_t* ranges) {
    uint32_t n_cycles = n / (sizeof(__m128i)/sizeof(uint32_t));
    __m128i* vec = (__m128i*)(ranges);
    __m128i q = _mm_set1_epi32(query);
    __m128i add = _mm_set1_epi32(0);

    uint32_t tot = 0;
    for (int i = 0; i < n_cycles; ++i) {
        __m128i v0 = _mm_loadu_si128(vec + i + 0);
        __m128i lt = _mm_cmplt_epi32(q, v0);
        __m128i gt = _mm_cmpgt_epi32(q, v0);
        __m128i shuffle1 = _mm_shuffle_epi32(lt, _MM_SHUFFLE(2,3,0,1));
        __m128i collapse = _mm_and_si128(shuffle1, gt);
        add = _mm_add_epi32(add, _mm_and_si128(collapse, _mm_set1_epi32(1)));
    }

    for (int i = 0; i < 4; ++i) tot += _mm_extract_epi32(add, i);

    return tot;
}

bool overlap_simd_add_unroll2(const uint32_t query, const ssize_t n, const uint32_t* ranges) {
    uint32_t n_cycles = n / (sizeof(__m128i)/sizeof(uint32_t));
    __m128i* vec = (__m128i*)(ranges);
    __m128i q = _mm_set1_epi32(query);
    __m128i add = _mm_set1_epi32(0);

    uint32_t tot = 0;
    int i = 0;
    for (/**/; i + 2 < n_cycles; i += 2) {
        __m128i v0 = _mm_loadu_si128(vec + i + 0);
        __m128i v1 = _mm_loadu_si128(vec + i + 1);

        __m128i lt = _mm_cmplt_epi32(q, v0);
        __m128i gt = _mm_cmpgt_epi32(q, v0);
        __m128i lt2 = _mm_cmplt_epi32(q, v1);
        __m128i gt2 = _mm_cmpgt_epi32(q, v1);

        __m128i shuffle1 = _mm_shuffle_epi32(lt, _MM_SHUFFLE(2,3,0,1));
        __m128i shuffle2 = _mm_shuffle_epi32(lt2, _MM_SHUFFLE(2,3,0,1));
        
        __m128i collapse = _mm_and_si128(shuffle1, gt);
        __m128i collapse2 = _mm_and_si128(shuffle2, gt2);
        
        add = _mm_add_epi32(add, _mm_and_si128(collapse, _mm_set1_epi32(1)));
        add = _mm_add_epi32(add, _mm_and_si128(collapse2, _mm_set1_epi32(1)));
    }

    for (/**/; i < n_cycles; ++i) {
        __m128i v0 = _mm_loadu_si128(vec + i + 0);
        __m128i lt = _mm_cmplt_epi32(q, v0);
        __m128i gt = _mm_cmpgt_epi32(q, v0);
        __m128i shuffle1 = _mm_shuffle_epi32(lt, _MM_SHUFFLE(2,3,0,1));
        __m128i collapse = _mm_and_si128(shuffle1, gt);
        add = _mm_add_epi32(add, _mm_and_si128(collapse, _mm_set1_epi32(1)));
    }

    for (int i = 0; i < 4; ++i) tot += _mm_extract_epi32(add, i);

    return tot;
}

bool overlap_simd_add_unroll4(const uint32_t query, const ssize_t n, const uint32_t* ranges) {
    uint32_t n_cycles = n / (sizeof(__m128i)/sizeof(uint32_t));
    __m128i* vec = (__m128i*)(ranges);
    __m128i q = _mm_set1_epi32(query);
    __m128i add = _mm_set1_epi32(0);

    uint32_t tot = 0;
    int i = 0;
    for (/**/; i + 4 < n_cycles; i += 4) {
        __m128i v0 = _mm_loadu_si128(vec + i + 0);
        __m128i v1 = _mm_loadu_si128(vec + i + 1);
        __m128i v2 = _mm_loadu_si128(vec + i + 2);
        __m128i v3 = _mm_loadu_si128(vec + i + 3);

        __m128i lt  = _mm_cmplt_epi32(q, v0);
        __m128i gt  = _mm_cmpgt_epi32(q, v0);
        __m128i lt2 = _mm_cmplt_epi32(q, v1);
        __m128i gt2 = _mm_cmpgt_epi32(q, v1);
        __m128i lt3 = _mm_cmplt_epi32(q, v2);
        __m128i gt3 = _mm_cmpgt_epi32(q, v2);
        __m128i lt4 = _mm_cmplt_epi32(q, v3);
        __m128i gt4 = _mm_cmpgt_epi32(q, v3);

        __m128i shuffle1 = _mm_shuffle_epi32(lt,  _MM_SHUFFLE(2,3,0,1));
        __m128i shuffle2 = _mm_shuffle_epi32(lt2, _MM_SHUFFLE(2,3,0,1));
        __m128i shuffle3 = _mm_shuffle_epi32(lt3, _MM_SHUFFLE(2,3,0,1));
        __m128i shuffle4 = _mm_shuffle_epi32(lt4, _MM_SHUFFLE(2,3,0,1));
        
        __m128i collapse = _mm_and_si128(shuffle1,  gt);
        __m128i collapse2 = _mm_and_si128(shuffle2, gt2);
        __m128i collapse3 = _mm_and_si128(shuffle3, gt3);
        __m128i collapse4 = _mm_and_si128(shuffle4, gt4);
        
        add = _mm_add_epi32(add, _mm_and_si128(collapse,  _mm_set1_epi32(1)));
        add = _mm_add_epi32(add, _mm_and_si128(collapse2, _mm_set1_epi32(1)));
        add = _mm_add_epi32(add, _mm_and_si128(collapse3, _mm_set1_epi32(1)));
        add = _mm_add_epi32(add, _mm_and_si128(collapse4, _mm_set1_epi32(1)));
    }

    for (/**/; i + 2 < n_cycles; i += 2) {
        __m128i v0 = _mm_loadu_si128(vec + i + 0);
        __m128i v1 = _mm_loadu_si128(vec + i + 1);

        __m128i lt = _mm_cmplt_epi32(q, v0);
        __m128i gt = _mm_cmpgt_epi32(q, v0);
        __m128i lt2 = _mm_cmplt_epi32(q, v1);
        __m128i gt2 = _mm_cmpgt_epi32(q, v1);

        __m128i shuffle1 = _mm_shuffle_epi32(lt, _MM_SHUFFLE(2,3,0,1));
        __m128i shuffle2 = _mm_shuffle_epi32(lt2, _MM_SHUFFLE(2,3,0,1));
        
        __m128i collapse = _mm_and_si128(shuffle1, gt);
        __m128i collapse2 = _mm_and_si128(shuffle2, gt2);
        
        add = _mm_add_epi32(add, _mm_and_si128(collapse, _mm_set1_epi32(1)));
        add = _mm_add_epi32(add, _mm_and_si128(collapse2, _mm_set1_epi32(1)));
    }

    for (/**/; i < n_cycles; ++i) {
        __m128i v0 = _mm_loadu_si128(vec + i + 0);
        __m128i lt = _mm_cmplt_epi32(q, v0);
        __m128i gt = _mm_cmpgt_epi32(q, v0);
        __m128i shuffle1 = _mm_shuffle_epi32(lt, _MM_SHUFFLE(2,3,0,1));
        __m128i collapse = _mm_and_si128(shuffle1, gt);
        add = _mm_add_epi32(add, _mm_and_si128(collapse, _mm_set1_epi32(1)));
    }

    for (int i = 0; i < 4; ++i) tot += _mm_extract_epi32(add, i);

    return tot;
}

bool overlap_avx2(const uint32_t query, const ssize_t n, const uint32_t* ranges) {
    if (n < 16)
        return overlap_scalar(query, n, ranges);

    uint32_t n_cycles = n / (sizeof(__m256i)/sizeof(uint32_t));
    __m256i* vec = (__m256i*)(ranges);
    __m256i q = _mm256_set1_epi32(query);

    uint32_t tot = 0;
    for (int i = 0; i < n_cycles; ++i) {
        __m256i v0 = _mm256_loadu_si256(vec + i + 0);
        __m256i lt = _mm256_cmpgt_epi32(v0, q); // hack for missing cmplt
        __m256i gt = _mm256_cmpgt_epi32(q, vec[i]);
        __m256i shuffle1 = _mm256_shuffle_epi32(lt, _MM_SHUFFLE(2,3,0,1));
        __m256i collapse = _mm256_and_si256(shuffle1, gt);
        PIL_POPCOUNT_AVX2(tot, collapse);
    }

    return tot;
}

bool overlap_ekg_itree(IntervalTree<uint32_t,uint32_t>& itree, std::vector< Interval<uint32_t,uint32_t> >& results, const uint32_t query, const ssize_t n, const uint32_t* ranges) {
    uint32_t overlaps = 0;
    itree.findOverlapping(query, query, results);
    overlaps += results.size();
    return overlaps;
}

bool overlap_scalar_binary(const uint32_t query, const ssize_t n, const uint32_t* ranges) {
    if (n == 0) return false;

    if (n < 4)
        return overlap_scalar(query, n, ranges);

    if (query < ranges[0]) return false; // if first interval [A,B] start after the query then never overlap
    if (query > ranges[n-1]) return false; // if the query starts after the last interval then never overlap
    
    struct test {
        uint32_t a, b;
    };

    test* tt = (test*)(ranges);

    // Binary search
    uint32_t from = 0, to = n/2, mid = 0;
    while (true) {
        mid = (to + from) / 2;
        __builtin_prefetch(&tt[(mid + 1 + to)/2], 0, 1);
        __builtin_prefetch(&tt[(from + mid - 1)/2], 0, 1);

        if(tt[mid].b <= query) from = mid + 1;
        else if(tt[mid].a >= query) to = mid - 1;
        else {
            //std::cerr << "match=" << tt[mid].a << "-" << tt[mid].b << " with q=" << query << std::endl;
            //assert(query < tt[mid].b && query > tt[mid].a);
            return true;
        }

        if(to - from <= 32) break; // region is small
    }

    //std::cerr << "here= " << from << "-" << to << " for " << to-from << std::endl;

    uint32_t overlaps = 0;
    for (int i = from; i <= to; ++i) {
        overlaps += (query < tt[i].b && query > tt[i].a);
    }

    //if(overlaps) std::cerr << "overlaps" << std::endl;
    return overlaps;
}

bool overlap_simd_add_binary(const uint32_t query, const ssize_t n, const uint32_t* ranges) {
    if (n < 4)
        return overlap_scalar(query, n, ranges);

    struct test {
        uint32_t a, b;
    };

    test* tt = (test*)(ranges);

    // Binary search
    uint32_t from = 0, to = n/2, mid = 0;
    while (true) {
        mid = (to + from) / 2;
        __builtin_prefetch(&tt[(mid + 1 + to)/2], 0, 1);
        __builtin_prefetch(&tt[(from + mid - 1)/2], 0, 1);

        if(tt[mid].b <= query) from = mid + 1;
        else if(tt[mid].a >= query) to = mid - 1;
        else return true;

        if(to - from <= 32) break; // region is small
    }

    uint32_t tot = 0;


    uint32_t target_range = 2*((to - from) + 1);
    uint32_t n_cycles = target_range / (sizeof(__m128i)/sizeof(uint32_t));
    __m128i* vec = (__m128i*)(&tt[from]);
    
    const __m128i q = _mm_set1_epi32(query);

    int i = 0;
    for (/**/; i < n_cycles; ++i) {
        __m128i v = _mm_loadu_si128(&vec[i]);
        __m128i lt = _mm_cmplt_epi32(q, v);
        __m128i gt = _mm_cmpgt_epi32(q, v);
        __m128i shuffle1 = _mm_shuffle_epi32(lt, _MM_SHUFFLE(2,3,0,1));
        __m128i collapse = _mm_and_si128(shuffle1, gt);
        tot += __builtin_popcountll(_mm_cvtsi128_si64(collapse));
        tot += __builtin_popcountll(_mm_cvtsi128_si64(_mm_srli_si128(collapse, 8)));
    }

    i = from + n_cycles*(sizeof(__m128i)/sizeof(uint32_t));
    for (/**/; i <= to; ++i) {
        tot += (query < tt[i].b && query > tt[i].a);
    }

    return tot;
}

bool overlap_scalar_binary_skipsquash(const uint32_t query, const ssize_t n, 
                                      const uint32_t* ranges, const uint64_t* bitmaps, 
                                      const uint32_t n_bitmaps, const uint32_t bin_size, 
                                      const std::vector<bitmap_helper>& bitmap_data) 
{
    if (n == 0) return false;

    if (n < 4)
        return overlap_scalar(query, n, ranges);

    if (query < ranges[0])   return false; // if first interval [A,B] start after the query then never overlap
    if (query > ranges[n-1]) return false; // if the query starts after the last interval then never overlap

    const uint32_t target_bin = query / bin_size; // single position / bin_size
    if ((bitmaps[target_bin / 64] & (1ULL << (target_bin % 64))) == 0) 
        return false; // squash

    //int target_bucket = __builtin_ctzll(bitmaps[target_bin / 64] & (1ULL << (target_bin % 64))) + 1;
    //const uint32_t target_bucket = target_bin % 64;
    //std::cerr << "trailing = " << target_bucket - 1  << " " << std::bitset<64>(bitmaps[target_bin / 64] & (1ULL << (target_bin % 64))) << std::endl;
    //std::cerr << target_bin << "->" << target_bucket << " or " << __builtin_ctzll(bitmaps[target_bin / 64] & (1ULL << (target_bin % 64))) << std::endl;

    //std::cerr << "QUERY=" << query << std::endl;
    //for (int i = 0; i < bitmap_data[target_bucket].intervals.size(); ++i) {
    //    std::cerr << " " << bitmap_data[target_bucket].intervals[i];
    //}
    //std::cerr << std::endl;

    // temp
    struct test {
        uint32_t a, b;
    };

    uint32_t overlaps = 0;

    const std::vector<uint32_t>& d = bitmap_data[target_bin].intervals;
    if (query < d.front()) return false; // if first interval [A,B] start after the query then never overlap
    if (query > d.back())  return false; // if the query starts after the last interval then never overlap

    if (d.size() < 32) {
        for (int i = 0; i + 2 <= d.size(); i += 2) {
            overlaps += (query < d[i+1] && query > d[i]);
        }
        return overlaps;
        //return (overlap_scalar(query, d.size(), &d[0]));       
    } else {
        test* tt = (test*)(&d[0]);

        // Binary search
        uint32_t from = 0, to = d.size() / 2 - 1, mid = 0;
        while (true) {
            mid = (to + from) / 2;
            __builtin_prefetch(&tt[(mid + 1 + to) / 2],   0, 1);
            __builtin_prefetch(&tt[(from + mid - 1) / 2], 0, 1);

            if(tt[mid].b <= query) from = mid + 1;
            else if(tt[mid].a >= query) to = mid - 1;
            else {
                //std::cerr << "match=" << tt[mid].a << "-" << tt[mid].b << " with q=" << query << std::endl;
                assert(query < tt[mid].b && query > tt[mid].a);
                return true;
            }

            if(to - from <= 32) break; // region is small
        }
        

        //std::cerr << "here= " << from << "-" << to << " for " << to-from << " lim=" << d.size() << std::endl;
        //for (int i = 0; i + 2 <= d.size(); i += 2) {
        //    overlaps += (query < d[i+1] && query > d[i]);
        //}
        //return overlaps;

        
        for (int i = from; i <= to; ++i) {
            overlaps += (query < tt[i].b && query > tt[i].a);
        }
        

        //std::cerr << "overlaps=" << overlaps << std::endl;
        //if(overlaps) std::cerr << "overlaps MOTHERFUKKKKA" << std::endl;

        //if(overlaps) std::cerr << "overlaps" << std::endl;
    }

    return overlaps;
}

// Definition for microsecond timer.
typedef std::chrono::high_resolution_clock hclock;
typedef hclock::time_point clockdef;

bool debug_bench() {
    uint32_t n_ranges = 8;
    uint32_t min_range = 0, max_range = 16;
    
    uint32_t* ranges = new uint32_t[n_ranges];
    
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator
    std::uniform_int_distribution<uint32_t> distr(min_range, max_range); // right inclusive

    uint32_t query = 9;

    for (int c = 0; c < 10000; ++c) {
        std::vector< Interval<uint32_t,uint32_t> > intervals;
        intervals.resize(n_ranges / 2); // stupid because intervaltree corrupts data

        for (int i = 0; i < n_ranges; ++i) {
            ranges[i] = distr(eng);
            //std::cout << " " << ranges[i];
        }
        //std::cout << std::endl;

        std::sort(ranges, &ranges[n_ranges]);
        //for (int i = 0; i < n_ranges; ++i) {
        //    std::cout << " " << ranges[i];
        //}
        //std::cout << std::endl;

        for (int i = 0, j = 0; i + 2 <= n_ranges; i += 2, ++j) {
            intervals[j].start = ranges[i+0];
            intervals[j].stop  = ranges[i+1];
        }

        //for (int i = 0; i < n_ranges/2; ++i) {
        //    std::cout << " " << intervals[i].start << "," << intervals[i].stop;
        //}
        //std::cout << std::endl;

        // Start timer.

        IntervalTree<uint32_t,uint32_t> itree(std::move(intervals));
        std::vector< Interval<uint32_t,uint32_t> > results;
        results.reserve(10);
        //std::cerr << "done build" << std::endl;
        
        bool a = overlap_scalar(query, n_ranges, ranges);
        bool b = overlap_simd(query, n_ranges, ranges);
        bool i_c = overlap_ekg_itree(itree, results, query, n_ranges, ranges);
        
        //std::cout << a << "," << b << "," << i_c << std::endl;
        assert(a == b);
        assert(i_c == a);
    }

    delete[] ranges;
    return true;
}

uint64_t wrapper(overlap_func f, ssize_t n_queries, uint32_t* __restrict__ queries, ssize_t n_ranges, const uint32_t* __restrict__ ranges, uint64_t& timing) {
    // Start timer.
    clockdef t1 = hclock::now();

    uint64_t n_overlaps = 0;
    for (int i = 0; i < n_queries; ++i) {
        n_overlaps += (*f)(queries[i], n_ranges, ranges);
    }

    // End timer and update times.
    clockdef t2 = hclock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    timing = time_span.count();

    // queries / second
    std::cerr << time_span.count() << " " << (uint64_t)((double)n_queries/time_span.count()*1e6) << std::endl;

    return n_overlaps;
}

uint64_t wrapper_tree(ssize_t n_queries, uint32_t* __restrict__ queries, ssize_t n_ranges, const uint32_t* __restrict__ ranges, uint64_t& timing) {
    // Construction phase -- do not add this to timer
    std::vector< Interval<uint32_t,uint32_t> > intervals;
    intervals.resize(n_ranges / 2); // stupid because intervaltree corrupts data

    for (int i = 0, j = 0; i + 2 <= n_ranges; i += 2, ++j) {
        intervals[j].start = ranges[i+0];
        intervals[j].stop  = ranges[i+1];
    }
    
    // Start timer.
    clockdef t1 = hclock::now();

    // Add construction of tree to timer.
    IntervalTree<uint32_t,uint32_t> itree(std::move(intervals));
    std::vector< Interval<uint32_t,uint32_t> > results;
    results.reserve(n_ranges);

    // Predicate phase
    uint64_t n_overlaps = 0;
    for (int i = 0; i < n_queries; ++i) {
        n_overlaps += overlap_ekg_itree(itree, results, queries[i], n_ranges, ranges);
        results.clear();
    }

    // End timer and update times.
    clockdef t2 = hclock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    timing = time_span.count();

    // queries / second
    std::cerr << time_span.count() << " " << (uint64_t)((double)n_queries/time_span.count()*1e6) << std::endl;

    return n_overlaps;
}

uint64_t wrapper_listsquash(overlap_func_squash f, 
                            ssize_t n_queries, uint32_t* __restrict__ queries, 
                            ssize_t n_ranges, const uint32_t* __restrict__ ranges,
                            uint64_t* bitmaps, const uint32_t n_bitmaps, 
                            const uint32_t bin_size, 
                            const std::vector<bitmap_helper>& bitmap_data,
                            uint64_t& timing) 
{
    // Start timer.
    clockdef t1 = hclock::now();

    uint64_t n_overlaps = 0;
    for (int i = 0; i < n_queries; ++i) {
        n_overlaps += (*f)(queries[i], n_ranges, ranges, bitmaps, n_bitmaps, bin_size, bitmap_data);
    }

    // End timer and update times.
    clockdef t2 = hclock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    timing = time_span.count();

    // queries / second
    std::cerr << time_span.count() << " " << (uint64_t)((double)n_queries/time_span.count()*1e6) << std::endl;

    return n_overlaps;
}


bool bench() {
    // Some numbers:
    // @see https://www.ncbi.nlm.nih.gov/genome/annotation_euk/Homo_sapiens/106/
    // 233,785 exons and 207,344 introns from 20,246 annotated genes (hg38)
    std::vector<uint32_t> n_ranges = {8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576,2097152,4194304,8388608,16777216,33554432,67108864,134217728,268435456};

    for (int r = 0; r < n_ranges.size(); ++r) {
        uint32_t min_range = 0, max_range = 100e6;
        uint32_t n_queries = 10e6;
        //uint32_t* ranges   = new uint32_t[n_ranges[r]];
        //uint32_t* queries  = new uint32_t[n_queries];

        // Memory align input data.
        uint32_t* ranges  = (uint32_t*)aligned_malloc(n_ranges[r]*sizeof(uint32_t), SIMD_ALIGNMENT);
        uint32_t* queries = (uint32_t*)aligned_malloc(n_queries*sizeof(uint32_t), SIMD_ALIGNMENT);
        
        std::random_device rd; // obtain a random number from hardware
        std::mt19937 eng(rd()); // seed the generator
        std::uniform_int_distribution<uint32_t> distr(min_range, max_range); // right inclusive

        for (int c = 0; c < 50; ++c) {
            // Generate queries
            for (int i = 0; i < n_ranges[r]; ++i) {
                ranges[i] = distr(eng);
            }

            // SIMD methods require that the ranges are sorted.
            std::sort(ranges, &ranges[n_ranges[r]]);

            ///////////////////////////////////////////////////////////////
            // Max value is back
            uint32_t n_bins = std::min(n_ranges[r], std::min((uint32_t)n_ranges[r]/8, (uint32_t)131072)); // choose this such that no more than 50% of bits are ever set
            uint32_t step_size = ranges[n_ranges[r]-1] / n_bins + 1;
            std::cerr << "max=" << ranges[n_ranges[r]-1] << " -> " << n_bins << "," << step_size << std::endl;
            //std::cerr << "bins=" << std::endl;

            uint32_t n_bitmaps = n_bins/64 + 1;
            uint64_t* bitmaps = new uint64_t[n_bitmaps];
            memset(bitmaps,0,sizeof(uint64_t)*n_bitmaps);

            std::vector<bitmap_helper> bitmaps_pointers(n_bins + 1);

            // Map interval to segments and update segmental bitmap.
            for (int i = 0; i + 2 <= n_ranges[r]; i += 2) {
                uint32_t from = ranges[i+0] / step_size;
                uint32_t to   = ranges[i+1] / step_size;
                for (int k = from; k <= to; ++k) {
                    // Set bit for segment.
                    bitmaps[k / 64] |= 1ULL << (k % 64);
                   
                    // Debug
                    if(k/64 >= n_bitmaps) {
                        std::cerr << "illegal bitmap k=" << k << ":" << k/64 << "/" << n_bitmaps << std::endl;
                        exit(1);
                    }

                    // Debug
                    if(k >= bitmaps_pointers.size()) {
                        std::cerr << "out of bounds:" << k << "/" << bitmaps_pointers.size() << std::endl;
                    }

                    // Copy data into segment.
                    // These are sorted so added in-order.
                    bitmaps_pointers[k].intervals.push_back(ranges[i+0]);
                    bitmaps_pointers[k].intervals.push_back(ranges[i+1]);
                }
            }

            /*
            for (int i = 0; i < bitmaps_pointers.size(); ++i) {
                if(bitmaps_pointers[i].intervals.size()) {
                    std::cerr << "pointers=" << bitmaps_pointers[i].intervals.size() << std::endl;
                    for (int k = 0; k + 2 <= bitmaps_pointers[i].intervals.size(); k += 2) {
                        std::cerr << "," << bitmaps_pointers[i].intervals[k+0] << "-" << bitmaps_pointers[i].intervals[k+1];
                    }
                    std::cerr << std::endl;
                }                    
            }
            */

            //std::cerr << std::bitset<64>(bitmaps[0]) << " " << std::bitset<64>(bitmaps[1]) << std::endl;
            ///////////////////////////////////////////////////////////////

            //for (int i = 1; i + 2 < n_ranges; i += 2) {
            //    ranges[i] = std::numeric_limits<uint32_t>::max() >> 1;
            //}

            // Generate points as queries.
            for (int i = 0; i < n_queries; ++i) {
                queries[i] = distr(eng);
            }

            /*
            for (int i = 0; i < n_queries; ++i) {
                bool a = overlap_scalar(queries[i], n_ranges, ranges);
                bool b = overlap_simd(queries[i], n_ranges, ranges);
                assert(a == b);
            }
            */

            uint64_t timings[64] = {0};

            if (n_ranges[r] <= 8196) {
                uint64_t n_scalar_nosimd = wrapper(&overlap_scalar_nosimd, n_queries, queries, n_ranges[r], ranges, timings[0]);
                uint64_t n_simd = wrapper(&overlap_simd, n_queries, queries, n_ranges[r], ranges, timings[3]);
                uint64_t n_avx2 = wrapper(&overlap_avx2, n_queries, queries, n_ranges[r], ranges, timings[4]);
                uint64_t n_simd_add = wrapper(&overlap_simd_add, n_queries, queries, n_ranges[r], ranges, timings[5]);
                uint64_t n_simd_add2 = wrapper(&overlap_simd_add_unroll2, n_queries, queries, n_ranges[r], ranges, timings[6]);
                uint64_t n_simd_add4 = wrapper(&overlap_simd_add_unroll4, n_queries, queries, n_ranges[r], ranges, timings[7]);
                uint64_t n_scalar = wrapper(&overlap_scalar, n_queries, queries, n_ranges[r], ranges, timings[8]);
                uint64_t n_scalar_break = wrapper(&overlap_scalar_break, n_queries, queries, n_ranges[r], ranges, timings[9]);
                std::cerr << "done cycle " << c << " -> " << n_scalar_nosimd << " " << n_simd << "," << n_avx2 << "," << n_simd_add << "," << n_simd_add2 << "," << n_simd_add4 << "," << n_scalar << "," << n_scalar_break << std::endl;
            }
            
            uint64_t n_binary = wrapper(&overlap_scalar_binary, n_queries, queries, n_ranges[r], ranges, timings[1]);
            uint64_t n_simd_binary = wrapper(&overlap_simd_add_binary, n_queries, queries, n_ranges[r], ranges, timings[2]);
            uint64_t n_ekg_tree = wrapper_tree(n_queries, queries, n_ranges[r], ranges, timings[10]);
            //uint64_t n_ekg_tree = 0;
            std::cerr << "BINARY=" << n_binary << " and " << n_simd_binary << " EKG=" << n_ekg_tree << std::endl;

            uint64_t n_squash = wrapper_listsquash(&overlap_scalar_binary_skipsquash, 
                                                   n_queries, queries, 
                                                   n_ranges[r], ranges, 
                                                   bitmaps, bitmaps_pointers.size(), step_size, bitmaps_pointers, 
                                                   timings[11]);
            
            std::cerr << "squash=" << n_squash << std::endl;

            std::cout << n_ranges[r] << "\t" << n_queries << "\t" << c;
            for (int i = 0; i < 12; ++i) {
                std::cout << "\t" << timings[i];
            }
            std::cout << std::endl;
        
            delete[] bitmaps;
        }

        //delete[] queries;
        //delete[] ranges;
        aligned_free(queries);
        aligned_free(ranges);
    }

    return true;
}

int main(int argc, char **argv) {
    debug_bench();
    std::cerr << "done" << std::endl;
    bench();
    std::cerr << "done bench" << std::endl;
    return EXIT_SUCCESS;
}