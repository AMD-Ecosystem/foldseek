#ifndef LIBMARV_CUDA_TO_HIP_H
#define LIBMARV_CUDA_TO_HIP_H

// Compatibility shim that lets the libmarv CUDA sources build under ROCm/HIP
// while keeping the NVIDIA build byte-identical (this header is only force
// included on the HIP compile). It maps the CUDA runtime/Thrust/CUB spellings
// the library uses to their HIP equivalents and supplies the small set of
// device intrinsics ROCm 7.2.x does not provide (packed-fp16 max and the
// Hopper DPX 16-/32-bit SIMD ops). The DPX path is never selected at runtime
// on AMD (see the kernel-config selectors), but its template instantiations
// are still compiled, so the intrinsics must exist as valid device code.

// On Windows, <windows.h> defines min/max as function-like macros which break
// C++ standard library uses of std::numeric_limits<T>::max() etc. Suppress them.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif

// On ROCm 7.14+, <amd_hip_bf16.h> defines __shfl_*_sync bfloat16 overloads that
// conflict with this header's own warp-sync shuffle definitions, causing a clang
// "redefinition of default argument" error; HIP_DISABLE_WARP_SYNC_BUILTINS
// suppresses the runtime's builtins so ours win. On ROCm 7.2.x those overloads
// and __syncwarp (used in kernels.cuh) live behind the SAME guard, so defining
// it there would strip __syncwarp and break the build; the flag must be set on
// 7.14+ only. Key it on the HIP runtime version, not the OS.
#include <hip/hip_version.h>
#if defined(HIP_VERSION) && (HIP_VERSION_MAJOR > 7 || (HIP_VERSION_MAJOR == 7 && HIP_VERSION_MINOR >= 14))
#  define HIP_DISABLE_WARP_SYNC_BUILTINS
#endif

#include <cstring>
#include <cstdlib>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// CUDA kernel-parameter qualifier with no HIP equivalent; the by-value
// parameter passing it optimizes is the default on HIP, so map it to nothing.
#ifndef __grid_constant__
#define __grid_constant__
#endif

// HIP __shfl_*_sync require a 64-bit lane mask (one bit per wavefront lane);
// the CUDA sources use the 32-bit 0xFFFFFFFF full-warp mask. Additionally,
// when HIP_DISABLE_WARP_SYNC_BUILTINS is set (ROCm 7.14+) the masked-sync
// builtins are suppressed entirely; map them to the maskless HIP equivalents.
#define WARP_FULL_MASK 0xFFFFFFFFFFFFFFFFull
typedef unsigned long long WarpMaskT;

#define __shfl_sync(mask, ...)      __shfl(__VA_ARGS__)
#define __shfl_up_sync(mask, ...)   __shfl_up(__VA_ARGS__)
#define __shfl_down_sync(mask, ...) __shfl_down(__VA_ARGS__)
#define __shfl_xor_sync(mask, ...)  __shfl_xor(__VA_ARGS__)

// CUB's RAII device switcher has no hipCUB equivalent; provide it in namespace
// cub for the single use site.
namespace cub {
    struct SwitchDevice {
        int oldDevice = 0;
        explicit SwitchDevice(int device){
            hipGetDevice(&oldDevice);
            hipSetDevice(device);
        }
        ~SwitchDevice(){ hipSetDevice(oldDevice); }
        SwitchDevice(const SwitchDevice&) = delete;
        SwitchDevice& operator=(const SwitchDevice&) = delete;
    };
}

// ---- runtime API -----------------------------------------------------------
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetLastError hipGetLastError
#define cudaGetErrorString hipGetErrorString
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceProp hipDeviceProp_t
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDeviceSetCacheConfig hipDeviceSetCacheConfig
#define cudaDeviceGetDefaultMemPool hipDeviceGetDefaultMemPool
#define cudaFuncCachePreferShared hipFuncCachePreferShared

#define cudaDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMinor hipDeviceAttributeComputeCapabilityMinor
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define cudaDevAttrPageableMemoryAccessUsesHostPageTables hipDeviceAttributePageableMemoryAccessUsesHostPageTables
#define cudaDeviceAttrReadOnlyHostRegisterSupported hipDeviceAttributeCanUseHostPointerForRegisteredMem

#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyToSymbol hipMemcpyToSymbol
#define cudaMemcpyToSymbolAsync hipMemcpyToSymbolAsync
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemGetInfo hipMemGetInfo
#define cudaMemAdvise hipMemAdvise
#define cudaMemAdviseSetAccessedBy hipMemAdviseSetAccessedBy
#define cudaMemcpyKind hipMemcpyKind
#define cudaMemcpyDefault hipMemcpyDefault
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToHost hipMemcpyHostToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice

#define cudaMalloc hipMalloc
#define cudaMallocHost hipHostMalloc
#define cudaMallocManaged hipMallocManaged
#define cudaMallocAsync hipMallocAsync
#define cudaFree hipFree
#define cudaFreeHost hipHostFree
#define cudaFreeAsync hipFreeAsync
#define cudaHostRegister hipHostRegister
#define cudaHostRegisterDefault hipHostRegisterDefault

#define cudaMemPool_t hipMemPool_t
#define cudaMemPoolSetAttribute hipMemPoolSetAttribute
#define cudaMemPoolAttrReleaseThreshold hipMemPoolAttrReleaseThreshold

#define cudaStream_t hipStream_t
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamQuery hipStreamQuery
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaStreamLegacy ((hipStream_t)1)
#define cudaStreamPerThread ((hipStream_t)2)

#define cudaEvent_t hipEvent_t
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventQuery hipEventQuery
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventDisableTiming hipEventDisableTiming

#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
// hipFuncSetAttribute takes const void*; cast the typed kernel function pointer.
#define cudaFuncSetAttribute(func, attr, value) hipFuncSetAttribute((const void*)(func), (attr), (value))
#define cudaLaunchHostFunc hipLaunchHostFunc

#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define cudaDeviceDisablePeerAccess hipDeviceDisablePeerAccess
#define cudaErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#define cudaErrorPeerAccessNotEnabled hipErrorPeerAccessNotEnabled

#define cudaIpcMemHandle_t hipIpcMemHandle_t
#define cudaIpcGetMemHandle hipIpcGetMemHandle
#define cudaIpcOpenMemHandle hipIpcOpenMemHandle
#define cudaIpcMemLazyEnablePeerAccess hipIpcMemLazyEnablePeerAccess

// rocThrust exposes the device execution policy under thrust::hip, not
// thrust::cuda. Expanded at the call sites (after the thrust headers are
// included), so the forward reference is fine.
#define THRUST_DEVICE_PAR_NOSYNC thrust::hip::par_nosync

// ---- vector-type warp shuffles --------------------------------------------
// HIP supplies __shfl* for scalars and __half2 but not short2, which the
// gapless/Smith-Waterman kernels shuffle. Provide short2 overloads that relay
// the value as a 32-bit word. Declared before <hip/hip_cooperative_groups.h>
// (pulled in via the compat cooperative_groups.h) so the tile shfl_* members
// resolve to them. No default width: the tile members always pass it.
#if defined(__HIPCC__)

__device__ __forceinline__ short2 __shfl(short2 v, int src, int w){
    unsigned int u; memcpy(&u, &v, 4); u = __shfl((int)u, src, w); memcpy(&v, &u, 4); return v;
}
__device__ __forceinline__ short2 __shfl_up(short2 v, unsigned int d, int w){
    unsigned int u; memcpy(&u, &v, 4); u = __shfl_up((int)u, d, w); memcpy(&v, &u, 4); return v;
}
__device__ __forceinline__ short2 __shfl_down(short2 v, unsigned int d, int w){
    unsigned int u; memcpy(&u, &v, 4); u = __shfl_down((int)u, d, w); memcpy(&v, &u, 4); return v;
}
__device__ __forceinline__ short2 __shfl_xor(short2 v, int m, int w){
    unsigned int u; memcpy(&u, &v, 4); u = __shfl_xor((int)u, m, w); memcpy(&v, &u, 4); return v;
}

// int2 (two 32-bit words): used to shuffle the packed M/E score columns.
__device__ __forceinline__ int2 __shfl(int2 v, int src, int w){
    v.x = __shfl(v.x, src, w); v.y = __shfl(v.y, src, w); return v;
}
__device__ __forceinline__ int2 __shfl_up(int2 v, unsigned int d, int w){
    v.x = __shfl_up(v.x, d, w); v.y = __shfl_up(v.y, d, w); return v;
}
__device__ __forceinline__ int2 __shfl_down(int2 v, unsigned int d, int w){
    v.x = __shfl_down(v.x, d, w); v.y = __shfl_down(v.y, d, w); return v;
}
__device__ __forceinline__ int2 __shfl_xor(int2 v, int m, int w){
    v.x = __shfl_xor(v.x, m, w); v.y = __shfl_xor(v.y, m, w); return v;
}

// float2 (two 32-bit words): used to shuffle the packed M/E score columns.
__device__ __forceinline__ float2 __shfl(float2 v, int src, int w){
    v.x = __shfl(v.x, src, w); v.y = __shfl(v.y, src, w); return v;
}
__device__ __forceinline__ float2 __shfl_up(float2 v, unsigned int d, int w){
    v.x = __shfl_up(v.x, d, w); v.y = __shfl_up(v.y, d, w); return v;
}
__device__ __forceinline__ float2 __shfl_down(float2 v, unsigned int d, int w){
    v.x = __shfl_down(v.x, d, w); v.y = __shfl_down(v.y, d, w); return v;
}
__device__ __forceinline__ float2 __shfl_xor(float2 v, int m, int w){
    v.x = __shfl_xor(v.x, m, w); v.y = __shfl_xor(v.y, m, w); return v;
}

#endif // __HIPCC__

// ---- device intrinsics ROCm 7.2.x does not provide -------------------------
// Packed-fp16 max/min: scalar __hmax/__hmin exist on ROCm, the packed forms do
// not. Emulate pairwise so the portable half2 gapless path matches CUDA.
#if defined(__HIPCC__)

__device__ __forceinline__ __half2 __hmax2(const __half2 a, const __half2 b){
    return __halves2half2(__hmax(__low2half(a), __low2half(b)),
                          __hmax(__high2half(a), __high2half(b)));
}

__device__ __forceinline__ __half2 __hmin2(const __half2 a, const __half2 b){
    return __halves2half2(__hmin(__low2half(a), __low2half(b)),
                          __hmin(__high2half(a), __high2half(b)));
}

// Hopper DPX 16-bit SIMD intrinsics, packed as two int16 lanes in a uint32.
// Only reached when the DPX kernel templates are instantiated; the runtime
// kernel-config selector keeps dpx=0 on AMD, so these never execute, but they
// must compile as valid device code.
__device__ __forceinline__ unsigned int __vadd2(unsigned int a, unsigned int b){
    short2 av, bv, rv;
    memcpy(&av, &a, 4); memcpy(&bv, &b, 4);
    rv.x = short(av.x + bv.x);
    rv.y = short(av.y + bv.y);
    unsigned int r; memcpy(&r, &rv, 4); return r;
}

__device__ __forceinline__ unsigned int __vmaxs2(unsigned int a, unsigned int b){
    short2 av, bv, rv;
    memcpy(&av, &a, 4); memcpy(&bv, &b, 4);
    rv.x = av.x > bv.x ? av.x : bv.x;
    rv.y = av.y > bv.y ? av.y : bv.y;
    unsigned int r; memcpy(&r, &rv, 4); return r;
}

__device__ __forceinline__ unsigned int __vibmax_u16x2(unsigned int a, unsigned int b, bool* pa, bool* pb){
    unsigned short av0 = (unsigned short)(a & 0xFFFFu), av1 = (unsigned short)(a >> 16);
    unsigned short bv0 = (unsigned short)(b & 0xFFFFu), bv1 = (unsigned short)(b >> 16);
    bool xge = av0 >= bv0;
    bool yge = av1 >= bv1;
    *pa = xge;
    *pb = yge;
    unsigned short r0 = xge ? av0 : bv0;
    unsigned short r1 = yge ? av1 : bv1;
    return (unsigned int)r0 | ((unsigned int)r1 << 16);
}

__device__ __forceinline__ unsigned int __vimax3_s16x2(unsigned int a, unsigned int b, unsigned int c){
    short2 av, bv, cv, rv;
    memcpy(&av, &a, 4); memcpy(&bv, &b, 4); memcpy(&cv, &c, 4);
    short mx0 = av.x > bv.x ? av.x : bv.x; rv.x = mx0 > cv.x ? mx0 : cv.x;
    short mx1 = av.y > bv.y ? av.y : bv.y; rv.y = mx1 > cv.y ? mx1 : cv.y;
    unsigned int r; memcpy(&r, &rv, 4); return r;
}

__device__ __forceinline__ unsigned int __vimax3_s16x2_relu(unsigned int a, unsigned int b, unsigned int c){
    short2 av, bv, cv, rv;
    memcpy(&av, &a, 4); memcpy(&bv, &b, 4); memcpy(&cv, &c, 4);
    short mx0 = av.x > bv.x ? av.x : bv.x; mx0 = mx0 > cv.x ? mx0 : cv.x; rv.x = mx0 > 0 ? mx0 : 0;
    short mx1 = av.y > bv.y ? av.y : bv.y; mx1 = mx1 > cv.y ? mx1 : cv.y; rv.y = mx1 > 0 ? mx1 : 0;
    unsigned int r; memcpy(&r, &rv, 4); return r;
}

__device__ __forceinline__ unsigned int __viaddmax_s16x2(unsigned int a, unsigned int b, unsigned int c){
    short2 av, bv, cv, rv;
    memcpy(&av, &a, 4); memcpy(&bv, &b, 4); memcpy(&cv, &c, 4);
    short s0 = short(av.x + bv.x); rv.x = s0 > cv.x ? s0 : cv.x;
    short s1 = short(av.y + bv.y); rv.y = s1 > cv.y ? s1 : cv.y;
    unsigned int r; memcpy(&r, &rv, 4); return r;
}

__device__ __forceinline__ unsigned int __viaddmax_s16x2_relu(unsigned int a, unsigned int b, unsigned int c){
    short2 av, bv, cv, rv;
    memcpy(&av, &a, 4); memcpy(&bv, &b, 4); memcpy(&cv, &c, 4);
    short s0 = short(av.x + bv.x); s0 = s0 > cv.x ? s0 : cv.x; rv.x = s0 > 0 ? s0 : 0;
    short s1 = short(av.y + bv.y); s1 = s1 > cv.y ? s1 : cv.y; rv.y = s1 > 0 ? s1 : 0;
    unsigned int r; memcpy(&r, &rv, 4); return r;
}

// 32-bit DPX scalar ops used by the integer Smith-Waterman DPX path.
__device__ __forceinline__ int __vibmax_s32(int a, int b, bool* pa){
    bool ge = a >= b;
    *pa = ge;
    return ge ? a : b;
}

__device__ __forceinline__ int __vimax3_s32(int a, int b, int c){
    int m = a > b ? a : b;
    return m > c ? m : c;
}

__device__ __forceinline__ int __viaddmax_s32(int a, int b, int c){
    int s = a + b;
    return s > c ? s : c;
}

__device__ __forceinline__ int __viaddmax_s32_relu(int a, int b, int c){
    int s = a + b;
    int m = s > c ? s : c;
    return m > 0 ? m : 0;
}

#endif // __HIPCC__

#endif // LIBMARV_CUDA_TO_HIP_H
