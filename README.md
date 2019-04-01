# interval-intersection

These functions compute the interval intersection pairs of interval sets.

### Usage

Compile the test suite with: `make` and run `./ranges`.

Include `ranges.h` and `ranges.c` in your project. Then use the wrapper function for `pospopcnt`:
```c
intersect_intervals(TBD);
```

### History

These functions were developed for [pil](https://github.com/mklarqvist/pil) but can be applied to any interval intersection problem.

## Problem statement

See [paper](https://www.biorxiv.org/content/biorxiv/early/2019/01/11/517987.full.pdf).

Let  us  first  introduce  the  notation  we will use. Let $I_r$ denote  a ‘reference’ collection of intervals, and $I_q$ denote a ‘query’ collection of intervals. We use the space-counted, zero-start convention for genomic coordinates. Namely, we count the space between bases starting from 0 (the one before the first base) up to $g$ (the one after the last base), where $g$ denotes the length of the genomic region of interest. Thus, each interval is denoted by a pair of indices $(u_1,u_2)$ with $0 \leq u_1 \lt u_2 \leq g$,and is composed of the nucleotides between $u_1$ and $u_2$.  We use $i$ to index the intervals in query set $I$, which has total number of $n$ intervals, and designate $j$ to index the intervals in reference set $I_r$, which consists of $m$ intervals in total. The length of $i$-th query interval and $j$-th reference interval are represented by $l_i$ and $x_j$, respectively. Two intervals $(u_1,u_2)$ and $(v_1,v_2)$ overlap iff they share common nucleotide(s). A collection of intervals is non-overlapping if no pair of intervals in the collection overlap.

## Goals

* Evaluate set-membership predicates: is this given query interval in the reference set?
* Evaluate the set intersection count: how many reference intervals did my query interval overlaps with?
* Achieve high-performance on large arrays of values.
* Support machines without SIMD (scalar).
* Specialized algorithms for SSE2 up to AVX512.

## Technical approach

