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
* Set bits in bitmap to true if ANY value overlaps that range (A,B)
* Possibly duplicate values that overlap multiple bins.
* Perform overlap test in bins.
* Stop if any bin has a matching value.
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

struct interval {
    uint32_t left, right;
};

struct bitmap_helper {
    //std::shared_ptr< std::vector<interval> > intervals;
    std::vector<interval> intervals;
};

/*------ Function pointers --------*/

typedef bool(*overlap_func)(const uint32_t, const ssize_t, const interval*);
typedef bool(*overlap_func_ordered)(const uint32_t, const ssize_t, const interval*, const interval*&);
typedef bool(*overlap_func_squash)(const uint32_t, const ssize_t,
                                   const interval*, const uint64_t*, 
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
 #ifndef __clang__
__attribute__((optimize("no-tree-vectorize")))
#else
//__attribute__ ((optnone))
#endif
bool overlap_scalar_nosimd(const uint32_t query, const ssize_t n, const interval* ranges) {
    uint32_t overlaps = 0;
    //#pragma clang loop vectorize(disable)
    for (int i = 0; i < n; ++i) {
        overlaps += (query < ranges[i].right && query > ranges[i].left);
    }
    return overlaps;
}

// Retrieve the first overlap match, if any.
bool overlap_scalar_first_match(const uint32_t query, const ssize_t n, const interval* ranges, const interval*& hit) {
    for (int i = 0; i < n; ++i) {
        if ((query < ranges[i].right && query > ranges[i].left)) {
            hit = &ranges[i]; 
            return true;
        }
    }
    return false;
}

bool overlap_scalar(const uint32_t query, const ssize_t n, const interval* ranges) {
    uint32_t overlaps = 0;
    for (int i = 0; i < n; ++i) {
        overlaps += (query < ranges[i].right && query > ranges[i].left);
    }
    return overlaps;
}

bool overlap_scalar_break(const uint32_t query, const ssize_t n, const interval* ranges) {
    uint32_t overlaps = 0;
    for (int i = 0; i < n; ++i) {
        overlaps += (query < ranges[i].right && query > ranges[i].left);
        if(overlaps) break;
    }

    return overlaps;
}

bool overlap_simd(const uint32_t query, const ssize_t n, const interval* ranges) {
    const __m128i q = _mm_set1_epi32(query);
    const __m128i one_mask = _mm_set1_epi32(1);
    __m128i counters = _mm_set1_epi32(0);
    const uint32_t* r = (const uint32_t*)ranges;

    uint64_t tot = 0;
    int i = 0;
    for (; i + 4 <= 2*n; i += 4) {
        __m128i v0 = _mm_lddqu_si128((__m128i*)(r + i));
        __m128i lt = _mm_cmplt_epi32(q, v0);
        __m128i gt = _mm_cmpgt_epi32(q, v0);
        __m128i shuffle1 = _mm_srli_epi64(lt, 32);
        __m128i collapse = _mm_and_si128(shuffle1 & gt, one_mask);
        counters = _mm_add_epi32(counters, collapse);
    }

    for (int i = 0; i < 4; ++i) tot += _mm_extract_epi32(counters, i);

    i /= 2;
    for (; i < n; ++i) {
        tot += (query < ranges[i].right && query > ranges[i].left);
    }

    return tot;
}

bool overlap_simd_add(const uint32_t query, const ssize_t n, const interval* ranges) {
    //uint32_t n_cycles = 2*n / (sizeof(__m128i)/sizeof(uint32_t));
    //__m128i* vec = (__m128i*)(ranges);
    __m128i q = _mm_set1_epi32(query);
    //__m128i add = _mm_set1_epi32(0);
    //const __m128i one_mask = _mm_set1_epi32(1);
    const uint32_t* r = (const uint32_t*)ranges;

    uint64_t tot = 0;
    ssize_t i = 0;
    for (; i + 4 <= 2*n; i += 4) {
        __m128i v0 = _mm_lddqu_si128((__m128i*)(r + i + 0));
        __m128i lt = _mm_cmplt_epi32(q, v0);
        __m128i gt = _mm_cmpgt_epi32(q, v0);
        __m128i shuffle1 = _mm_srli_epi64(lt, 32);
        //__m128i collapse = _mm_and_si128(shuffle1 & gt, one_mask);
        tot += _mm_movemask_epi8(shuffle1 & gt);
        //add = _mm_add_epi32(add, collapse);
    }

    //for (int i = 0; i < 4; ++i) tot += _mm_extract_epi32(add, i);

    i /= 2;
    for (; i < n; ++i) {
        tot += (query < ranges[i].right && query > ranges[i].left);
    }

    return tot;
}

bool overlap_simd_add_firstmatch(const uint32_t query, const ssize_t n, const interval* ranges, const interval*& hit) {
    uint32_t n_cycles = 2*n / (sizeof(__m128i)/sizeof(uint32_t));
    //__m128i* vec = (__m128i*)(ranges);
    __m128i q = _mm_set1_epi32(query);
    //__m128i add = _mm_set1_epi32(0);
    //const __m128i one_mask = _mm_set1_epi32(1);
    const uint32_t* r = (const uint32_t*)ranges;

    uint64_t tot = 0;
    ssize_t i = 0;
    for (; i + 4 <= 2*n; i += 4) {
        __m128i v0 = _mm_lddqu_si128((__m128i*)(r + i + 0));
        __m128i lt = _mm_cmplt_epi32(q, v0);
        __m128i gt = _mm_cmpgt_epi32(q, v0);
        __m128i shuffle1 = _mm_srli_epi64(lt, 32);
        //__m128i collapse = _mm_and_si128(shuffle1 & gt, one_mask);
        //add = _mm_add_epi32(add, collapse);
        tot += _mm_movemask_epi8(shuffle1 & gt);
    }

    //for (int i = 0; i < 4; ++i) tot += _mm_extract_epi32(add, i);

    i /= 2;
    for (; i < n; ++i) {
        tot += (query < ranges[i].right && query > ranges[i].left);
    }

    return tot;
}

bool overlap_simd_add_unroll2(const uint32_t query, const ssize_t n, const interval* ranges) {
    uint32_t n_cycles = 2*n / (sizeof(__m128i)/sizeof(uint32_t));
    __m128i* vec = (__m128i*)(ranges);
    __m128i q = _mm_set1_epi32(query);
    //__m128i add = _mm_set1_epi32(0);
    //const __m128i one_mask = _mm_set1_epi32(1);

    uint64_t tot = 0;
    int i = 0;
    for (/**/; i + 2 < n_cycles; i += 2) {
        __m128i v0 = _mm_loadu_si128(vec + i + 0);
        __m128i v1 = _mm_loadu_si128(vec + i + 1);

        __m128i lt = _mm_cmplt_epi32(q, v0);
        __m128i gt = _mm_cmpgt_epi32(q, v0);
        __m128i lt2 = _mm_cmplt_epi32(q, v1);
        __m128i gt2 = _mm_cmpgt_epi32(q, v1);

        __m128i shuffle1 = _mm_srli_epi64(lt, 32);
        //__m128i collapse = _mm_and_si128(shuffle1 & gt, one_mask);
        //add = _mm_add_epi32(add, collapse);

        tot += _mm_movemask_epi8(shuffle1 & gt);

        __m128i shuffle2 = _mm_srli_epi64(lt2, 32);
        //collapse = _mm_and_si128(shuffle2 & gt2, one_mask);
        //add = _mm_add_epi32(add, collapse);
        tot += _mm_movemask_epi8(shuffle2 & gt2);
    }

    for (/**/; i < n_cycles; ++i) {
        __m128i v0 = _mm_loadu_si128(vec + i + 0);
        __m128i lt = _mm_cmplt_epi32(q, v0);
        __m128i gt = _mm_cmpgt_epi32(q, v0);
        __m128i shuffle1 = _mm_srli_epi64(lt, 32);
        //__m128i collapse = _mm_and_si128(shuffle1 & gt, one_mask);
        //add = _mm_add_epi32(add, collapse);
        tot += _mm_movemask_epi8(shuffle1 & gt);
    }

    //for (int i = 0; i < 4; ++i) tot += _mm_extract_epi32(add, i);

    for (int i = n_cycles*4; i < n; ++i) {
        tot += (query < ranges[i].right && query > ranges[i].left);
    }

    return tot;
}

bool overlap_simd_add_unroll4(const uint32_t query, const ssize_t n, const interval* ranges) {
    uint32_t n_cycles = 2*n / (sizeof(__m128i)/sizeof(uint32_t));
    __m128i* vec = (__m128i*)(ranges);
    __m128i q = _mm_set1_epi32(query);
    //__m128i add = _mm_set1_epi32(0);
    //const __m128i one_mask = _mm_set1_epi32(1);

    uint64_t tot = 0;
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

        __m128i shuffle1 = _mm_srli_epi64(lt, 32);
        //__m128i collapse = _mm_and_si128(shuffle1 & gt, one_mask);
        //add = _mm_add_epi32(add, collapse);
        tot += _mm_movemask_epi8(shuffle1 & gt);

        __m128i shuffle2 = _mm_srli_epi64(lt2, 32);
        //collapse = _mm_and_si128(shuffle2 & gt2, one_mask);
        //add = _mm_add_epi32(add, collapse);
        tot += _mm_movemask_epi8(shuffle2 & gt2);

        __m128i shuffle3 = _mm_srli_epi64(lt3, 32);
        //collapse = _mm_and_si128(shuffle3 & gt3, one_mask);
        //add = _mm_add_epi32(add, collapse);
        tot += _mm_movemask_epi8(shuffle3 & gt3);

        __m128i shuffle4 = _mm_srli_epi64(lt4, 32);
        //collapse = _mm_and_si128(shuffle4 & gt4, one_mask);
        //add = _mm_add_epi32(add, collapse);
        tot += _mm_movemask_epi8(shuffle4 & gt4);
    }

    for (/**/; i + 2 < n_cycles; i += 2) {
        __m128i v0 = _mm_loadu_si128(vec + i + 0);
        __m128i v1 = _mm_loadu_si128(vec + i + 1);

        __m128i lt = _mm_cmplt_epi32(q, v0);
        __m128i gt = _mm_cmpgt_epi32(q, v0);
        __m128i lt2 = _mm_cmplt_epi32(q, v1);
        __m128i gt2 = _mm_cmpgt_epi32(q, v1);

        __m128i shuffle1 = _mm_srli_epi64(lt, 32);
        //__m128i collapse = _mm_and_si128(shuffle1 & gt, one_mask);
        //add = _mm_add_epi32(add, collapse);
        tot += _mm_movemask_epi8(shuffle1 & gt);

        __m128i shuffle2 = _mm_srli_epi64(lt2, 32);
        //collapse = _mm_and_si128(shuffle2 & gt2, one_mask);
        //add = _mm_add_epi32(add, collapse);
        tot += _mm_movemask_epi8(shuffle2 & gt2);
    }

    for (/**/; i < n_cycles; ++i) {
        __m128i v0 = _mm_loadu_si128(vec + i + 0);
        __m128i lt = _mm_cmplt_epi32(q, v0);
        __m128i gt = _mm_cmpgt_epi32(q, v0);
        __m128i shuffle1 = _mm_srli_epi64(lt, 32);
        //__m128i collapse = _mm_and_si128(shuffle1 & gt, one_mask);
        //add = _mm_add_epi32(add, collapse);
        tot += _mm_movemask_epi8(shuffle1 & gt);
    }

    //or (int i = 0; i < 4; ++i) tot += _mm_extract_epi32(add, i);

    for (int i = n_cycles*4; i < n; ++i) {
        tot += (query < ranges[i].right && query > ranges[i].left);
    }

    return tot;
}

#if SIMD_VERSION >= 5
bool overlap_avx2(const uint32_t query, const ssize_t n, const interval* ranges) {
    if (n < 16)
        return overlap_scalar(query, n, ranges);

    //uint32_t n_cycles = 2*n / (sizeof(__m128i)/sizeof(uint32_t));
    //__m128i* vec = (__m128i*)(ranges);
    __m256i q = _mm256_set1_epi32(query);
    //__m128i add = _mm_set1_epi32(0);
    //const __m128i one_mask = _mm_set1_epi32(1);
    const uint32_t* r = (const uint32_t*)ranges;

    uint64_t tot = 0;
    ssize_t i = 0;
    for (; i + 8 <= 2*n; i += 8) {
        __m256i v0 = _mm256_lddqu_si256((__m256i*)(r + i + 0));
        __m256i lt = _mm256_cmpgt_epi32(v0, q); // hack for not having less than
        __m256i gt = _mm256_cmpgt_epi32(q, v0);
        __m256i shuffle1 = _mm256_srli_epi64(lt, 32);
        //__m128i collapse = _mm_and_si128(shuffle1 & gt, one_mask);
        //add = _mm_add_epi32(add, collapse);
        tot += _mm256_movemask_epi8(shuffle1 & gt);
    }

    //for (int i = 0; i < 4; ++i) tot += _mm_extract_epi32(add, i);

    i /= 2;
    for (; i < n; ++i) {
        tot += (query < ranges[i].right && query > ranges[i].left);
    }

    return tot;
}
#else
bool overlap_avx2(const uint32_t query, const ssize_t n, const interval* ranges) { return 0; }
#endif

bool overlap_ekg_itree(IntervalTree<uint32_t,uint32_t>& itree, std::vector< Interval<uint32_t,uint32_t> >& results, 
                       const uint32_t query, 
                       const ssize_t n, const interval* ranges) 
{
    uint32_t overlaps = 0;
    itree.findOverlapping(query, query, results);
    overlaps += results.size();
    return overlaps;
}

bool overlap_scalar_binary(const uint32_t query, const ssize_t n, const interval* ranges) {
    if (n == 0) return false;

    if (n < 4)
        return overlap_scalar(query, n, ranges);

    if (query < ranges[0].left)    return false; // if first interval [A,B] start after the query then never overlap
    if (query > ranges[n-1].right) return false; // if the query starts after the last interval then never overlap

    // Binary search
    uint32_t from = 0, to = n - 1, mid = 0;
    while (true) {
        mid = (to + from) / 2;
        __builtin_prefetch(&ranges[(mid + 1 + to) / 2],   0, 1);
        __builtin_prefetch(&ranges[(from + mid - 1) / 2], 0, 1);

        if(ranges[mid].right <= query) from = mid + 1;
        else if(ranges[mid].left >= query) to = mid - 1;
        else return true;

        if(to - from <= 32) break; // region is small
    }

    //std::cerr << "here= " << from << "-" << to << " for " << to-from << std::endl;

    uint32_t overlaps = 0;
    for (int i = from; i <= to; ++i) {
        overlaps += (query < ranges[i].right && query > ranges[i].left);
    }

    //if(overlaps) std::cerr << "overlaps" << std::endl;
    return overlaps;
}

bool overlap_simd_add_binary(const uint32_t query, const ssize_t n, const interval* ranges) {
    if (n == 0) return false;

    if (n < 4)
        return overlap_scalar(query, n, ranges);

    if (query < ranges[0].left)    return false; // if first interval [A,B] start after the query then never overlap
    if (query > ranges[n-1].right) return false; // if the query starts after the last interval then never overlap

    // Binary search
    uint32_t from = 0, to = n - 1, mid = 0;
    while (true) {
        mid = (to + from) / 2;
        __builtin_prefetch(&ranges[(mid + 1 + to) / 2],   0, 1);
        __builtin_prefetch(&ranges[(from + mid - 1) / 2], 0, 1);

        if(ranges[mid].right <= query) from = mid + 1;
        else if(ranges[mid].left >= query) to = mid - 1;
        else return true;

        if(to - from <= 32) break; // region is small
    }

    uint64_t tot = 0;
    uint32_t target_range = to - from + 1;
    uint32_t n_cycles = 2*target_range / (sizeof(__m128i)/sizeof(uint32_t));
    __m128i* vec = (__m128i*)(&ranges[from]);
    
    const __m128i q = _mm_set1_epi32(query);
    const __m128i one_mask = _mm_set1_epi32(1);
    __m128i add = _mm_set1_epi32(0);
    
    int i = 0;
    for (/**/; i <= n_cycles; ++i) {
        __m128i v0 = _mm_loadu_si128(vec + i + 0);
        __m128i lt = _mm_cmplt_epi32(q, v0);
        __m128i gt = _mm_cmpgt_epi32(q, v0);
        __m128i shuffle1 = _mm_srli_epi64(lt, 32);
        __m128i collapse = _mm_and_si128(shuffle1 & gt, one_mask);
        add = _mm_add_epi32(add, collapse);
    }

    for (int i = 0; i < 4; ++i) tot += _mm_extract_epi32(add, i);

    i *= (sizeof(__m128i)/sizeof(uint32_t));
    for (/**/; i <= to; ++i) {
        tot += (query < ranges[i].right && query > ranges[i].left);
    }

    return tot;
}

bool overlap_scalar_binary_skipsquash(const uint32_t query, const ssize_t n_ranges, 
                                      const interval* ranges, const uint64_t* bitmaps, 
                                      const uint32_t n_bitmaps, const uint32_t bin_size, 
                                      const std::vector<bitmap_helper>& bitmap_data) 
{
    if (n_ranges == 0) return false;

    if (n_ranges < 4)
        return overlap_scalar(query, n_ranges, ranges);

    if (query < ranges[0].left) return false; // if first interval [A,B] start after the query then never overlap
    if (query > ranges[n_ranges-1].right) return false; // if the query starts after the last interval then never overlap

    const uint32_t target_bin = query / bin_size; // single position / bin_size

    if ((bitmaps[target_bin / 64] & (1ULL << (target_bin % 64))) == 0)
        return false; // squash

    uint32_t overlaps = 0;
    const std::vector<interval>& d = bitmap_data[target_bin].intervals;
    if (query < d.front().left)  return false; // if first interval [A,B] start after the query then never overlap
    if (query > d.back().right)  return false; // if the query starts after the last interval then never overlap

    if (d.size() < 32) {
        return overlap_avx2(query, d.size(), &d[0]);
    } else {
        // Binary search
        uint32_t from = 0, to = d.size() - 1, mid = 0;
        while (true) {
            mid = (to + from) / 2;
            __builtin_prefetch(&d[(mid + 1 + to) / 2],   0, 1);
            __builtin_prefetch(&d[(from + mid - 1) / 2], 0, 1);

            if(d[mid].right <= query) from = mid + 1;
            else if(d[mid].left >= query) to = mid - 1;
            else return true;

            if(to - from <= 32) break; // region is small
        }

        return overlap_avx2(query, to - from + 1, &d[from]);
    }
}

bool overlap_scalar_binary_firstmatch(const uint32_t query, const ssize_t n, const interval* ranges, const interval*& hit) {
    if (n == 0) return false;

    if (n < 4)
        return overlap_scalar_first_match(query, n, ranges, hit);

    if (query < ranges[0].left)    return false; // if first interval [A,B] start after the query then never overlap
    if (query > ranges[n-1].right) return false; // if the query starts after the last interval then never overlap

    // Binary search
    uint32_t from = 0, to = n - 1, mid = 0;
    while (true) {
        mid = (to + from) / 2;
        __builtin_prefetch(&ranges[(mid + 1 + to) / 2],   0, 1);
        __builtin_prefetch(&ranges[(from + mid - 1) / 2], 0, 1);

        if(ranges[mid].right <= query) from = mid + 1;
        else if(ranges[mid].left >= query) to = mid - 1;
        else {
            hit = &ranges[mid];
            return true;
        }

        if(to - from <= 32) break; // region is small
    }

    for (int i = from; i <= to; ++i) {
        if (query < ranges[i].right && query > ranges[i].left) {
            hit = &ranges[i];
            return true;
        }
    }

    return false;
}

bool overlap_scalar_binary_skipsquash(const uint32_t query, const ssize_t n_ranges, 
                                      const interval* ranges, const uint64_t* bitmaps, 
                                      const uint32_t n_bitmaps, const uint32_t bin_size, 
                                      const std::vector<bitmap_helper>& bitmap_data,
                                      const interval*& hit) 
{
    if (n_ranges == 0) return false;

    if (n_ranges < 4) {
        return overlap_scalar_first_match(query, n_ranges, ranges, hit);
    }

    if (query < ranges[0].left) return false; // if first interval [A,B] start after the query then never overlap
    if (query > ranges[n_ranges-1].right) return false; // if the query starts after the last interval then never overlap

    const uint32_t target_bin = query / bin_size; // single position / bin_size

    if ((bitmaps[target_bin / 64] & (1ULL << (target_bin % 64))) == 0)
        return false; // squash

    uint32_t overlaps = 0;
    const std::vector<interval>& d = bitmap_data[target_bin].intervals;
    if (query < d.front().left)  return false; // if first interval [A,B] start after the query then never overlap
    if (query > d.back().right)  return false; // if the query starts after the last interval then never overlap

    if (d.size() < 32) {
        return overlap_scalar_first_match(query, d.size(), &d[0], hit);
    } else {
        // Binary search
        uint32_t from = 0, to = d.size() - 1, mid = 0;
        while (true) {
            mid = (to + from) / 2;
            __builtin_prefetch(&d[(mid + 1 + to) / 2],   0, 1);
            __builtin_prefetch(&d[(from + mid - 1) / 2], 0, 1);

            if(d[mid].right <= query) from = mid + 1;
            else if(d[mid].left >= query) to = mid - 1;
            else {
                hit = &d[mid];
                return true;
            }

            if(to - from <= 32) break; // region is small
        }

        return overlap_scalar_first_match(query, to - from + 1, &d[from], hit);
    }
}

// Definition for microsecond timer.
typedef std::chrono::high_resolution_clock hclock;
typedef hclock::time_point clockdef;

bool debug_bench() {
    uint32_t n_ranges = 8;
    uint32_t min_range = 0, max_range = 16;
    
    uint32_t* ranges = new uint32_t[n_ranges];
    interval* ivals = new interval[n_ranges/2];
    
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator
    std::uniform_int_distribution<uint32_t> distr(min_range, max_range); // right inclusive

    uint32_t query = 9;

    for (int c = 0; c < 10000; ++c) {
        std::vector< Interval<uint32_t,uint32_t> > intervals;
        intervals.resize(n_ranges); // stupid because intervaltree corrupts data

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
            ivals[j].left  = ranges[i+0];
            ivals[j].right = ranges[i+1];
        }

        //for (int i = 0; i < n_ranges/2; ++i) {
        //    std::cout << " " << intervals[i].start << "," << intervals[i].stop;
        //}
        //std::cout << std::endl;

        // Start timer.
        //n_ranges /= 2;

        IntervalTree<uint32_t,uint32_t> itree(std::move(intervals));
        std::vector< Interval<uint32_t,uint32_t> > results;
        results.reserve(10);
        //std::cerr << "done build" << std::endl;
        
        bool a = overlap_scalar(query, n_ranges/2, ivals);
        bool b = overlap_simd(query, n_ranges/2, ivals);
        bool i_c = overlap_ekg_itree(itree, results, query, n_ranges/2, ivals);
        
        //std::cout << a << "," << b << "," << i_c << std::endl;
        //assert(a == b);
        //assert(i_c == a);
    }

    delete[] ranges;
    delete[] ivals;
    return true;
}

uint64_t wrapper(overlap_func f, 
                 ssize_t n_queries, uint32_t* queries, 
                 ssize_t n_ranges, const interval* ranges, 
                 uint64_t& timing) 
{
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

uint64_t wrapper_tree(ssize_t n_queries, uint32_t* queries, ssize_t n_ranges, const interval* ranges, uint64_t& timing) {
    // Construction phase -- do not add this to timer
    std::vector< Interval<uint32_t,uint32_t> > intervals;
    intervals.resize(n_ranges); // stupid because intervaltree corrupts data

    for (int i = 0; i < n_ranges; ++i) {
        intervals[i].start = ranges[i].left;
        intervals[i].stop  = ranges[i].right;
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
                            ssize_t n_queries, uint32_t* queries, 
                            ssize_t n_ranges, const interval* ranges,
                            uint64_t* bitmaps, const uint32_t n_bitmaps, 
                            const uint32_t bin_size, 
                            const std::vector<bitmap_helper>& bitmap_data,
                            uint64_t& timing, bool is_sorted) 
{
    uint64_t n_overlaps = 0;

    if (is_sorted == false) {
        // Start timer.
        clockdef t1 = hclock::now();
        
        for (int i = 0; i < n_queries; ++i) {
            n_overlaps += (*f)(queries[i], n_ranges, ranges, bitmaps, n_bitmaps, bin_size, bitmap_data);
        }

        // End timer and update times.
        clockdef t2 = hclock::now();
        auto time_span = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
        timing = time_span.count();

        // queries / second
        std::cerr << time_span.count() << " " << (uint64_t)((double)n_queries/time_span.count()*1e6) << std::endl;
    } else {
        const interval* prev = nullptr;
        bool found = false;

        // Start timer.
        clockdef t1 = hclock::now();

        uint64_t n_a = 0, n_b = 0;
        for (int i = 0; i < n_queries; ++i) {
            if (found) {
                assert(prev != nullptr);

                if ((queries[i] < prev->right && queries[i] > prev->left)) {
                    ++n_overlaps;
                    ++n_a;
                } else {
                    found = overlap_scalar_binary_skipsquash(queries[i], n_ranges, ranges, bitmaps, n_bitmaps, bin_size, bitmap_data, prev); // todo pass reference to prev and update if found
                    n_overlaps += found;
                    ++n_b;
                }
            } else { // no previous
                found = overlap_scalar_binary_skipsquash(queries[i], n_ranges, ranges, bitmaps, n_bitmaps, bin_size, bitmap_data, prev);
                n_overlaps += found;
                ++n_b;
            }
        }

        // End timer and update times.
        clockdef t2 = hclock::now();
        auto time_span = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
        timing = time_span.count();

        // queries / second
        std::cerr << time_span.count() << " " << (uint64_t)((double)n_queries/time_span.count()*1e6) << std::endl;
    }

    return n_overlaps;
}

uint64_t wrapper_sorted(overlap_func_ordered f, 
                        ssize_t n_queries, uint32_t* queries, 
                        ssize_t n_ranges, const interval* ranges, 
                        uint64_t& timing) 
{
    // Start timer.
    clockdef t1 = hclock::now();

    uint64_t n_overlaps = 0;
    const interval* prev = nullptr;
    bool found = false;

    uint64_t n_a = 0, n_b = 0;
    for (int i = 0; i < n_queries; ++i) {
        if (found) {
            if ((queries[i] < prev->right && queries[i] > prev->left)) {
                ++n_overlaps;
                ++n_a;
            } else {
                found = false;
                prev  = nullptr;
                found = (*f)(queries[i], n_ranges, ranges, prev);
                n_overlaps += found;
                ++n_b;
            }
        } else { // no previous
            found = (*f)(queries[i], n_ranges, ranges, prev);
            n_overlaps += found;
            ++n_b;
        }
    }

    // End timer and update times.
    clockdef t2 = hclock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    timing = time_span.count();

    // queries / second
    std::cerr << time_span.count() << " " << (uint64_t)((double)n_queries/time_span.count()*1e6) << " n=" << n_a << "," << n_b << std::endl;

    return n_overlaps;
}

bool bench() {
    // Some numbers:
    // @see https://www.ncbi.nlm.nih.gov/genome/annotation_euk/Homo_sapiens/106/
    // 233,785 exons and 207,344 introns from 20,246 annotated genes (hg38)
    
    //std::vector<uint32_t> n_ranges = {8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576,2097152,4194304,8388608,16777216,33554432,67108864,134217728,268435456};
    std::vector<uint32_t> n_ranges = {2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576,2097152,4194304,8388608,16777216,33554432,67108864,134217728,268435456};
    
    bool sort_query_intervals = true; // set to true to check performance of both sets being sorted
    uint32_t interval_min_range = 0, interval_max_range = 250e6; // chromosome 1 is 249Mb
    uint32_t query_min_range = 40e6, query_max_range = 140e6;
    uint32_t n_queries = 10e6;
    uint32_t n_repeats = 10;

    std::cerr << "Queries=" << n_queries << std::endl;

    for (int r = 0; r < n_ranges.size(); ++r) {
        //uint32_t* ranges   = new uint32_t[n_ranges[r]];
        //uint32_t* queries  = new uint32_t[n_queries];

        // Memory align input data.
        uint32_t* ranges  = (uint32_t*)aligned_malloc(n_ranges[r]*sizeof(uint32_t), SIMD_ALIGNMENT);
        uint32_t* queries = (uint32_t*)aligned_malloc(n_queries*sizeof(uint32_t),   SIMD_ALIGNMENT);
        const uint32_t n_ranges_pairs = n_ranges[r] / 2;
        interval* ivals = new interval[n_ranges_pairs];
        
        std::random_device rd; // obtain a random number from hardware
        std::mt19937 eng(rd()); // seed the generator
        std::uniform_int_distribution<uint32_t> distr(interval_min_range, interval_max_range); // right inclusive
        std::uniform_int_distribution<uint32_t> distrL(query_min_range, query_max_range); // right inclusive

        for (int c = 0; c < n_repeats; ++c) {
            std::cerr << "[GENERATING]>>>>> At=" << n_ranges[r] << std::endl;
            
            // Generate queries
            for (int i = 0; i < n_ranges[r]; ++i) {
                ranges[i] = distr(eng);
            }

            // SIMD methods require that the ranges are sorted.
            std::sort(ranges, &ranges[n_ranges[r]]);

            std::cerr << "a" << std::endl;
            for (int i = 0, j = 0; i + 2 <= n_ranges[r]; i += 2, ++j) {
                ivals[j].left  = ranges[i+0];
                ivals[j].right = ranges[i+1];
            }

            ///////////////////////////////////////////////////////////////
            // Max value is back
            std::cerr << "b" << std::endl;
            uint32_t n_bins = std::min(n_ranges[r], std::min((uint32_t)n_ranges[r]/8, (uint32_t)131072)); // choose this such that no more than 50% of bits are ever set
            if (n_bins == 0) n_bins = n_ranges[r];
            uint32_t step_size = ranges[n_ranges[r]-1] / n_bins + 1;
            std::cerr << "max=" << ranges[n_ranges[r]-1] << " -> " << n_bins << "," << step_size << std::endl;
            
            const uint32_t n_bitmaps = n_bins/64 + 1;
            uint64_t* bitmaps = new uint64_t[n_bitmaps];
            memset(bitmaps,0,sizeof(uint64_t)*n_bitmaps);

            std::vector<bitmap_helper> bitmaps_pointers(n_bins + 1);

            // Map interval to segments and update segmental bitmap.
            for (int i = 0; i < n_ranges_pairs; ++i) {
                uint32_t from = ivals[i].left / step_size;
                uint32_t to   = ivals[i].right / step_size + 1;
                for (int k = from; k <= to; ++k) {
                    // Set bit for segment.
                    bitmaps[k / 64] |= 1ULL << (k % 64);

                    // Copy data into segment.
                    // These are sorted so added in-order.
                    //interval iv; iv.left = iranges[i+0]; iv.right = ranges[i+1];
                    bitmaps_pointers[k].intervals.push_back(ivals[i]);
                }
            }

            ///////////////////////////////////////////////////////////////

            // Generate points as queries.
            for (int i = 0; i < n_queries; ++i) {
                queries[i] = distrL(eng);
            }

            if (sort_query_intervals) {
                std::cerr << "Sorting queries..." << std::endl;
                std::sort(&queries[0], &queries[n_queries - 1]);
            }

            // for (int i = 0; i < n_queries; ++i) {
            //     std::cerr << ", " << queries[i];
            // }
            // std::cerr << std::endl;

            /*
            for (int i = 0; i < n_queries; ++i) {
                bool a = overlap_scalar(queries[i], n_ranges, ranges);
                bool b = overlap_simd(queries[i], n_ranges, ranges);
                assert(a == b);
            }
            */

            uint64_t timings[64] = {0};

            if (n_ranges[r] <= 1024) {
                uint64_t n_scalar_nosimd = wrapper(&overlap_scalar_nosimd, n_queries, queries, n_ranges_pairs, ivals, timings[0]);
                uint64_t n_scalar = wrapper(&overlap_scalar, n_queries, queries, n_ranges_pairs, ivals, timings[1]);
                uint64_t n_scalar_break = wrapper(&overlap_scalar_break, n_queries, queries, n_ranges_pairs, ivals, timings[2]);

#if SIMD_VERSION >= 5
                uint64_t n_avx2 = wrapper(&overlap_avx2, n_queries, queries, n_ranges_pairs, ivals, timings[4]);
#else
                uint64_t n_avx2 = 0;
#endif
#if SIMD_VERSION >= 3
                uint64_t n_simd      = wrapper(&overlap_simd, n_queries, queries, n_ranges_pairs, ivals, timings[3]);
                uint64_t n_simd_add  = wrapper(&overlap_simd_add, n_queries, queries, n_ranges_pairs, ivals, timings[5]);
                uint64_t n_simd_add2 = wrapper(&overlap_simd_add_unroll2, n_queries, queries, n_ranges_pairs, ivals, timings[6]);
                uint64_t n_simd_add4 = wrapper(&overlap_simd_add_unroll4, n_queries, queries, n_ranges_pairs, ivals, timings[7]);
#else
                uint64_t n_simd      = 0;
                uint64_t n_simd_add  = 0;
                uint64_t n_simd_add2 = 0;
                uint64_t n_simd_add4 = 0;
#endif
                std::cerr << "done cycle " << c << " -> " << n_scalar_nosimd << "," << n_simd << "," << n_avx2 << "," << n_simd_add << "," << n_simd_add2 << "," << n_simd_add4 << "," << n_scalar << "," << n_scalar_break << std::endl;
            }
            
            uint64_t n_binary = wrapper(&overlap_scalar_binary, n_queries, queries, n_ranges_pairs, ivals, timings[8]);

            //uint64_t n_simd_binary = wrapper(&overlap_simd_add_binary, n_queries, queries, n_ranges_pairs, ivals, timings[9]);
            uint64_t n_simd_binary = 0;
            uint64_t n_ekg_tree = wrapper_tree(n_queries, queries, n_ranges_pairs, ivals, timings[10]);
            //uint64_t n_ekg_tree = 0;
            std::cerr << "BINARY=" << n_binary << " and " << n_simd_binary << " EKG=" << n_ekg_tree << std::endl;
            assert(n_binary == n_ekg_tree);

            uint64_t n_squash = wrapper_listsquash(&overlap_scalar_binary_skipsquash, 
                                                   n_queries, queries, 
                                                   n_ranges_pairs, ivals, 
                                                   bitmaps, bitmaps_pointers.size(), 
                                                   step_size, bitmaps_pointers, 
                                                   timings[11], false);

            uint64_t n_sq_sorted = wrapper_listsquash(&overlap_scalar_binary_skipsquash, 
                                                   n_queries, queries, 
                                                   n_ranges_pairs, ivals, 
                                                   bitmaps, bitmaps_pointers.size(), 
                                                   step_size, bitmaps_pointers, 
                                                   timings[14], true);
            
            std::cerr << "squash=" << n_squash << " squash_sorted=" << n_sq_sorted << std::endl;

            assert(n_squash == n_binary);
            assert(n_sq_sorted == n_binary);

            uint64_t n_simd_sorted = 0;
            if (n_ranges[r] <= 1024) {
                n_simd_sorted = wrapper_sorted(&overlap_scalar_first_match, n_queries, queries, n_ranges_pairs, ivals, timings[12]);
                assert(n_simd_sorted == n_binary);
                std::cerr << "sorted_naive=" << n_simd_sorted << std::endl;
            }

            uint64_t n_simd_sorted_binary = wrapper_sorted(&overlap_scalar_binary_firstmatch, n_queries, queries, n_ranges_pairs, ivals, timings[13]);
            std::cerr << "sorted_binary=" << n_simd_sorted_binary << std::endl;
            assert(n_simd_sorted_binary == n_binary);
            

            // Print output in nanoseconds.
            std::cout << n_ranges_pairs << "\t" << n_queries << "\t" << c << "\t" << (float)n_squash/n_queries;
            for (int i = 0; i < 15; ++i) {
                std::cout << "\t" << (double)timings[i]/n_queries*1000.0;
            }
            std::cout << std::endl;
        
            delete[] bitmaps;
        }

        delete[] ivals;
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