cmake_minimum_required(VERSION 3.8)

set(USE_HIP 0 CACHE BOOL "Build the GPU aligner with HIP for AMD ROCm")

if(USE_HIP)
    project(CUDA_INCLUDE_DIRS LANGUAGES CXX HIP)
    enable_language(HIP)
else()
    project(CUDA_INCLUDE_DIRS LANGUAGES CXX CUDA)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

set(LIBRARY_ONLY 0 CACHE BOOL "Do not build executables")

set(NVCC_FLAGS)
set(HIP_FLAGS)
if(USE_HIP)
    # The .cu sources keep their CUDA spelling; cuda_to_hip.h (force included)
    # and the hip_compat/ shim headers (cuda_fp16.h, cooperative_groups{,/reduce}.h)
    # bridge to HIP. -fgpu-rdc matches the CUDA separable-compilation build.
    list(APPEND HIP_FLAGS -fgpu-rdc)
    list(APPEND HIP_FLAGS -include ${CMAKE_CURRENT_SOURCE_DIR}/cuda_to_hip.h)
    list(APPEND HIP_FLAGS -I${CMAKE_CURRENT_SOURCE_DIR}/hip_compat)
else()
    list(APPEND NVCC_FLAGS --extended-lambda)
    list(APPEND NVCC_FLAGS --expt-relaxed-constexpr)
    list(APPEND NVCC_FLAGS -rdc=true)
endif()
string(TOLOWER "${CMAKE_BUILD_TYPE}" BUILD_TYPE_LOWER)
if(BUILD_TYPE_LOWER STREQUAL "debug" OR BUILD_TYPE_LOWER STREQUAL "relwithdebinfo")
    list(APPEND NVCC_FLAGS -lineinfo)
endif()

set(NVCC_FLAGS_OMP ${NVCC_FLAGS})
list(APPEND NVCC_FLAGS_OMP -Xcompiler -fopenmp)


set(ALIGN_SOURCES
    kernels.cuh
    blosum.cu
    pssm.cuh
    pssmkernels_gapless.cuh
    pssmkernels_gapless_instantiation_dpx.cu
    pssmkernels_gapless_instantiation_half2.cu
    pssmkernels_gapless_instantiation_dpx_kernelparamzero.cu
    pssmkernels_gapless_instantiation_half2_kernelparamzero.cu
    pssmkernels_smithwaterman.cuh
    pssmkernels_smithwaterman_instantiation_float.cu
    pssmkernels_smithwaterman_instantiation_dpx.cu
    dbdata.cpp
)

if(USE_HIP)
    # Build as a shared library so the HIP device-link of the -fgpu-rdc objects
    # happens inside this target (driven by the HIP compiler); the rest of the
    # project then links libmarv as an ordinary host shared object.
    add_library(marv SHARED ${ALIGN_SOURCES} marv.cu marv.h)
    # Only the .cu translation units are compiled as HIP; the .cuh files are
    # headers pulled in by them (compiling a .cuh standalone would duplicate the
    # __constant__ globals it defines and break the device link).
    set_source_files_properties(
        pssmkernels_gapless_instantiation_dpx.cu
        pssmkernels_gapless_instantiation_half2.cu
        pssmkernels_gapless_instantiation_dpx_kernelparamzero.cu
        pssmkernels_gapless_instantiation_half2_kernelparamzero.cu
        pssmkernels_smithwaterman_instantiation_float.cu
        pssmkernels_smithwaterman_instantiation_dpx.cu
        blosum.cu marv.cu
        PROPERTIES LANGUAGE HIP
    )
    set_source_files_properties(
        kernels.cuh pssm.cuh pssmkernels_gapless.cuh pssmkernels_smithwaterman.cuh
        PROPERTIES HEADER_FILE_ONLY ON
    )
    set_target_properties(marv PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_options(marv PRIVATE $<$<COMPILE_LANGUAGE:HIP>:${HIP_FLAGS}>)
    # On Windows (MSVC-ABI clang), CMake adds -fuse-ld=lld-link to the HIP
    # link command via CMAKE_HIP_USING_LINKER_DEFAULT. clang rejects that flag
    # when --hip-link is present (clang-linker-wrapper takes over from the
    # driver). Clear the default so clang-linker-wrapper picks lld-link on its own.
    if(WIN32)
        set(CMAKE_HIP_USING_LINKER_DEFAULT "")
    endif()
    target_link_options(marv PRIVATE -fgpu-rdc --hip-link)
else()
    add_library(marv ${ALIGN_SOURCES} marv.cu marv.h)
    target_compile_options(marv PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_FLAGS}>)
    set_target_properties(marv
        PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()
target_compile_definitions(marv PRIVATE NO_NVTOOLSEXT $<$<BOOL:${WIN32}>:MARV_BUILDING_DLL>)

if (NOT LIBRARY_ONLY)
    find_package(CUDAToolkit REQUIRED)
    find_package(ZLIB REQUIRED)
    find_package(OpenMP REQUIRED)

    add_executable(align ${ALIGN_SOURCES} main.cu options.cpp sequence_io.cpp)
    target_compile_options(align PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_FLAGS_OMP}>)
    target_link_libraries(align ZLIB::ZLIB CUDA::nvToolsExt OpenMP::OpenMP_CXX)
    set_target_properties(align PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    
    add_executable(makedb makedb.cpp sequence_io.cpp dbdata.cpp)
    target_compile_options(makedb PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_FLAGS_OMP}>)
    target_link_libraries(makedb ZLIB::ZLIB OpenMP::OpenMP_CXX)
    set_target_properties(makedb PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
