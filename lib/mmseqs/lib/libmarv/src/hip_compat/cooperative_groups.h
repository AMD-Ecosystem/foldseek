#ifndef LIBMARV_HIP_COMPAT_COOPERATIVE_GROUPS_H
#define LIBMARV_HIP_COMPAT_COOPERATIVE_GROUPS_H

// HIP's cooperative_groups provides tiled_partition / thread_block_tile with
// shfl/shfl_down/shfl_xor, but not the free-function reduce() or the
// greater<>/less<>/plus<> reduction operators that libmarv uses. Add them here
// over namespace cooperative_groups so the CUDA-spelled sources compile
// unchanged. reduce() is a butterfly all-reduce over the tile (every lane gets
// the result, matching CUDA cg::reduce), valid for the commutative+associative
// max/add operators libmarv passes.

#include <hip/hip_cooperative_groups.h>

namespace cooperative_groups {

    template<class T>
    struct greater {
        __device__ T operator()(const T& a, const T& b) const { return a > b ? a : b; }
    };

    template<class T>
    struct less {
        __device__ T operator()(const T& a, const T& b) const { return a < b ? a : b; }
    };

    template<class T>
    struct plus {
        __device__ T operator()(const T& a, const T& b) const { return a + b; }
    };

    // Shuffle an arbitrary trivially-copyable value across a tile by relaying
    // it as 32-bit words, so reduce() works for half2/uint as well as the
    // int3 endpoint-pack the Smith-Waterman kernels reduce.
    template<unsigned int Size, class Parent, class T>
    __device__ T tile_shfl_xor(const thread_block_tile<Size, Parent>& g, T val, unsigned int laneMask){
        constexpr int nwords = (sizeof(T) + sizeof(int) - 1) / sizeof(int);
        union { T v; int w[nwords]; } src, dst;
        src.v = val;
        #pragma unroll
        for(int i = 0; i < nwords; ++i){
            dst.w[i] = g.shfl_xor(src.w[i], laneMask);
        }
        return dst.v;
    }

    template<unsigned int Size, class Parent, class T, class Op>
    __device__ T reduce(const thread_block_tile<Size, Parent>& g, T val, Op op){
        for(unsigned int offset = g.size() / 2; offset > 0; offset /= 2){
            T other = tile_shfl_xor(g, val, offset);
            val = op(val, other);
        }
        return val;
    }

} // namespace cooperative_groups

#endif
