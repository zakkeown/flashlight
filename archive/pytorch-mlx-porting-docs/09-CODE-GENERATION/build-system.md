# PyTorch Build System Architecture

## Overview

PyTorch uses **CMake** as its primary build system to manage a highly complex, multi-language, multi-backend codebase. The build system handles C++17 compilation, CUDA/ROCm GPU backends, Metal (MPS) on macOS, Python extension building, code generation integration, and dependency management for 50+ third-party libraries.

**Key Characteristics**:
- **Massive Scale**: 400+ CMake options, 63k+ lines in Dependencies.cmake
- **Multi-Backend**: CPU, CUDA, ROCm, MPS (Metal), XPU, Vulkan
- **Code Generation**: Integrates torchgen output into build pipeline
- **Cross-Compilation**: Supports ARM64 (Apple Silicon), x86_64, RISC-V, PowerPC
- **Python Integration**: Builds Python C++ extensions via setuptools + CMake

**Location**: Root `CMakeLists.txt` (900+ lines), `cmake/` directory (25+ files)

---

## 1. Build System Architecture

### 1.1 Build Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│  python setup.py build (Python entry point)                 │
│  - Calls CMake with configuration flags                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  CMakeLists.txt (Root)                                       │
│  - Set build options (400+ options)                          │
│  - Detect platform (macOS, Linux, Windows)                  │
│  - Detect CPU architecture (x86_64, ARM64, etc.)            │
│  - Find dependencies (CUDA, BLAS, Python, etc.)             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  cmake/Dependencies.cmake (63k lines!)                       │
│  - CUDA/ROCm detection and configuration                    │
│  - BLAS library selection (MKL, OpenBLAS, Eigen, etc.)      │
│  - Protobuf, ONNX, gRPC                                      │
│  - 50+ third-party libraries                                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  cmake/Codegen.cmake                                         │
│  - Run torchgen code generation (native_functions.yaml)     │
│  - Generate C++ operator implementations                    │
│  - Generate Python bindings                                  │
│  - Generate autograd derivatives                            │
│  - Output: 1000s of .cpp/.h files in build/aten/src/ATen/   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Build Targets                                               │
│  - torch_cpu (core CPU library)                              │
│  - torch_cuda (CUDA backend)                                 │
│  - torch_python (Python bindings)                            │
│  - c10 (core library)                                        │
│  - ATen operators                                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Installation                                                │
│  - Copy .so/.dylib/.dll to torch/lib/                        │
│  - Copy Python modules to torch/                             │
│  - Install headers to include/                               │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Key CMake Files

| File | Lines | Purpose |
|------|-------|---------|
| `CMakeLists.txt` | 900+ | Root build configuration, options, platform detection |
| `cmake/Dependencies.cmake` | 63,000+ | Dependency management (CUDA, BLAS, protobuf, etc.) |
| `cmake/Codegen.cmake` | 600+ | Code generation integration (torchgen) |
| `cmake/public/cuda.cmake` | 1,500+ | CUDA configuration and nvcc flags |
| `cmake/BuildVariables.cmake` | 100+ | Build variable management |
| `cmake/Summary.cmake` | 400+ | Configuration summary printing |
| `cmake/Metal.cmake` | 200+ | Metal/MPS backend configuration |
| `cmake/ProtoBuf.cmake` | 300+ | Protobuf build configuration |
| `caffe2/CMakeLists.txt` | 1,000+ | Caffe2 library build |
| `torch/CMakeLists.txt` | 800+ | PyTorch Python extension build |

---

## 2. CMake Options and Configuration

### 2.1 Build Options (Selected Subset)

PyTorch has **400+ CMake options**. Key categories:

#### Core Build Options

```cmake
option(BUILD_PYTHON "Build Python bindings" ON)
option(BUILD_SHARED_LIBS "Build libcaffe2.so" ON)
option(BUILD_TEST "Build C++ test binaries" ON)
option(BUILD_BINARY "Build C++ binaries" OFF)
```

#### Backend Options

```cmake
option(USE_CUDA "Use CUDA" ON)
option(USE_ROCM "Use ROCm (AMD GPUs)" ON)
option(USE_MPS "Use MPS for macOS build" ON)  # Metal Performance Shaders
option(USE_XPU "Use XPU (Intel GPUs)" ON)
option(USE_VULKAN "Use Vulkan GPU backend" ON)  # For Android

cmake_dependent_option(USE_CUDNN "Use cuDNN" ON "USE_CUDA" OFF)
cmake_dependent_option(USE_CUSPARSELT "Use cuSPARSELt" ON "USE_CUDA" OFF)
cmake_dependent_option(USE_NCCL "Use NCCL" ON "USE_DISTRIBUTED;USE_CUDA" OFF)
```

**`cmake_dependent_option`**: Only available if condition is met

#### BLAS Options

```cmake
set(BLAS "MKL" CACHE STRING "Selected BLAS library")
set_property(CACHE BLAS PROPERTY STRINGS
  "ATLAS;BLIS;Eigen;MKL;OpenBLAS;vecLib;APL")

option(USE_MKLDNN "Use MKLDNN (oneDNN)" ON)  # Intel MKL-DNN
option(USE_STATIC_MKL "Prefer static MKL linkage" OFF)
```

#### Distributed Options

```cmake
option(USE_DISTRIBUTED "Use distributed training" ON)
cmake_dependent_option(USE_GLOO "Use Gloo" ON "USE_DISTRIBUTED" OFF)
cmake_dependent_option(USE_MPI "Use MPI" ON "USE_DISTRIBUTED" OFF)
cmake_dependent_option(USE_TENSORPIPE "Use TensorPipe" ON "USE_DISTRIBUTED" OFF)
```

#### Optimization Options

```cmake
option(USE_OPENMP "Use OpenMP for parallel code" ON)
option(USE_NATIVE_ARCH "Use -march=native" OFF)
option(USE_PRECOMPILED_HEADERS "Use pre-compiled headers" OFF)
option(USE_CCACHE "Attempt using CCache" ON)
```

#### Sanitizers

```cmake
option(USE_ASAN "Use Address+Undefined Sanitizers" OFF)
option(USE_TSAN "Use Thread Sanitizer" OFF)
option(USE_LSAN "Use Leak Sanitizer" OFF)
```

#### Code Generation

```cmake
option(USE_PER_OPERATOR_HEADERS "Generate separate headers per operator" ON)
```

**Purpose**: Reduces compile times by allowing parallel compilation of operator files

### 2.2 Platform Detection

```cmake
# Operating system
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  set(LINUX TRUE)
else()
  set(LINUX FALSE)
endif()

# CPU architecture
set(CPU_AARCH64 OFF)
set(CPU_INTEL OFF)
set(CPU_POWER OFF)  # PowerPC
set(CPU_RISCV OFF)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(AMD64|x86_64)")
  set(CPU_INTEL ON)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)")
  set(CPU_AARCH64 ON)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(ppc64le)")
  set(CPU_POWER ON)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(riscv64)")
  set(CPU_RISCV ON)
endif()

# Architecture-dependent defaults
cmake_dependent_option(
  USE_MKLDNN "Use MKLDNN" "${CPU_INTEL}"
  "CPU_INTEL OR CPU_AARCH64 OR CPU_POWER OR CPU_RISCV" OFF)
```

### 2.3 Apple Silicon and Metal (MPS) Detection

```cmake
if(APPLE)
  # Detect macOS SDK version
  execute_process(
    COMMAND bash -c "xcrun --sdk macosx --show-sdk-version"
    OUTPUT_VARIABLE _macosx_sdk_version
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  # MPS requires macOS 12.3+
  if(_macosx_sdk_version VERSION_GREATER_EQUAL 12.3)
    set(_MPS_supported_os_version ON)
  endif()

  # Find MetalPerformanceShadersGraph framework
  execute_process(
    COMMAND bash -c "xcrun --sdk macosx --show-sdk-path"
    OUTPUT_VARIABLE _macosx_sdk_path
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  set(_SDK_SEARCH_PATH "${_macosx_sdk_path}/System/Library/Frameworks/")

  find_library(_MPS_fwrk_path_
    NAMES MetalPerformanceShadersGraph
    PATHS ${_SDK_SEARCH_PATH}
    NO_DEFAULT_PATH)

  if(_MPS_supported_os_version AND _MPS_fwrk_path_)
    set(MPS_FOUND ON)
    message(STATUS "MPSGraph framework found")
  endif()
endif()

cmake_dependent_option(USE_MPS "Use MPS for macOS build" ON "MPS_FOUND" OFF)
```

**Significance**: MPS is PyTorch's Metal backend for Apple Silicon GPUs (M1/M2/M3)

---

## 3. Code Generation Integration

### 3.1 Codegen.cmake Overview

**Purpose**: Integrate torchgen (Python-based code generator) into CMake build

**Challenge**: CMake doesn't natively handle dynamic outputs. Torchgen can generate different files depending on configuration.

**Solution**: Two-stage approach:
1. **Dry-run**: Run torchgen with `--dry-run` to get list of outputs
2. **Generate**: Run torchgen as `add_custom_command` with outputs known

### 3.2 Code Generation Command

```cmake
# Determine flags based on configuration
set(GEN_ROCM_FLAG)
if(USE_ROCM)
  set(GEN_ROCM_FLAG --rocm)
endif()

set(GEN_MPS_FLAG)
if(USE_MPS)
  set(GEN_MPS_FLAG --mps)
endif()

set(GEN_XPU_FLAG)
if(USE_XPU)
  set(GEN_XPU_FLAG --xpu)
endif()

set(GEN_PER_OPERATOR_FLAG)
if(USE_PER_OPERATOR_HEADERS)
  list(APPEND GEN_PER_OPERATOR_FLAG "--per-operator-headers")
endif()

# Main code generation command
set(GEN_COMMAND
  "${Python_EXECUTABLE}" -m torchgen.gen
  --source-path ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen
  --install_dir ${CMAKE_BINARY_DIR}/aten/src/ATen
  ${GEN_PER_OPERATOR_FLAG}
  ${GEN_ROCM_FLAG}
  ${GEN_MPS_FLAG}
  ${GEN_XPU_FLAG}
  ${CUSTOM_BUILD_FLAGS}
)
```

### 3.3 Dynamic Output Discovery

Torchgen outputs are **dynamic** (change based on config). CMake requires knowing outputs upfront.

**Workaround**: Generate a `.cmake` file listing outputs

```cmake
# For each output type (headers, sources, declarations)
foreach(gen_type "headers" "sources" "declarations_yaml")
  set("GEN_COMMAND_${gen_type}"
    ${GEN_COMMAND}
    --generate ${gen_type}
    --output-dependencies ${CMAKE_BINARY_DIR}/aten/src/ATen/generated_${gen_type}.cmake
  )

  # Dry-run to get output list
  execute_process(
    COMMAND ${GEN_COMMAND_${gen_type}} --dry-run
    RESULT_VARIABLE RETURN_VALUE
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/..
  )

  if(NOT RETURN_VALUE EQUAL 0)
    message(FATAL_ERROR "Failed to get generated_${gen_type} list")
  endif()

  # Include generated .cmake file with output variables
  include("${CMAKE_BINARY_DIR}/aten/src/ATen/generated_${gen_type}.cmake")
  include("${CMAKE_BINARY_DIR}/aten/src/ATen/core_generated_${gen_type}.cmake")
  include("${CMAKE_BINARY_DIR}/aten/src/ATen/cpu_vec_generated_${gen_type}.cmake")
  include("${CMAKE_BINARY_DIR}/aten/src/ATen/cuda_generated_${gen_type}.cmake")
  include("${CMAKE_BINARY_DIR}/aten/src/ATen/ops_generated_${gen_type}.cmake")
endforeach()
```

**Generated .cmake Files** (example `generated_headers.cmake`):

```cmake
# Auto-generated by torchgen
set(generated_headers
  ${CMAKE_BINARY_DIR}/aten/src/ATen/Functions.h
  ${CMAKE_BINARY_DIR}/aten/src/ATen/NativeFunctions.h
  ${CMAKE_BINARY_DIR}/aten/src/ATen/RedispatchFunctions.h
  ${CMAKE_BINARY_DIR}/aten/src/ATen/Operators.h
  # ... 500+ more files
)
```

**Why This Works**: If `generated_headers.cmake` changes (different outputs), CMake re-runs automatically because it's an `include()`d file.

### 3.4 Custom Command for Generation

```cmake
add_custom_command(
  COMMENT "Generating ATen headers"
  OUTPUT ${generated_headers}
  COMMAND ${GEN_COMMAND_headers}
  DEPENDS
    ${all_python}  # All torchgen/*.py files
    ${headers_templates}  # Template files
    ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/native/native_functions.yaml
    ${CMAKE_CURRENT_LIST_DIR}/../aten/src/ATen/native/tags.yaml
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/..
)
```

**Dependency Tracking**: Code regenerates if:
- Any torchgen Python file changes
- Any template file changes
- `native_functions.yaml` changes
- `tags.yaml` changes

### 3.5 Generated Sources as Build Targets

```cmake
# Create library from generated sources
add_library(torch_cpu SHARED
  ${generated_sources}
  ${core_generated_sources}
  ${cpu_vec_generated_sources}
  # ... manually written sources
)

# Ensure headers are generated before compiling
add_dependencies(torch_cpu aten_headers_target)
```

---

## 4. Dependency Management

### 4.1 CUDA Configuration

```cmake
# From cmake/public/cuda.cmake

if(USE_CUDA)
  # Find CUDA toolkit
  find_package(CUDAToolkit REQUIRED)

  # Determine CUDA architectures to build for
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES
      50;60;61;70;75;80;86;89;90  # Pascal to Hopper
      CACHE STRING "CUDA architectures to build for")
  endif()

  # Set nvcc flags
  string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler -fPIC")
  string(APPEND CMAKE_CUDA_FLAGS " --expt-relaxed-constexpr")
  string(APPEND CMAKE_CUDA_FLAGS " --expt-extended-lambda")

  # cuDNN
  if(USE_CUDNN)
    find_package(CUDNN REQUIRED)
    target_link_libraries(torch_cuda PRIVATE torch::cudnn)
  endif()

  # NCCL (for distributed)
  if(USE_NCCL)
    find_package(NCCL REQUIRED)
    target_link_libraries(torch_cuda PRIVATE torch::nccl)
  endif()
endif()
```

**CUDA Architecture Handling**: Build for multiple GPU generations simultaneously

```cmake
# Example: Build for V100 (sm_70) and A100 (sm_80)
set(CMAKE_CUDA_ARCHITECTURES "70;80")
```

Generates PTX/cubin for each architecture, bloating binary size but ensuring compatibility.

### 4.2 BLAS Library Selection

PyTorch supports **9 different BLAS libraries**:

```cmake
set(BLAS "MKL" CACHE STRING "Selected BLAS library")
set_property(CACHE BLAS PROPERTY STRINGS
  "ATLAS;BLIS;Eigen;MKL;OpenBLAS;vecLib;APL;NVPL;FlexiBLAS")

if(BLAS STREQUAL "MKL")
  find_package(MKL QUIET)
  if(MKL_FOUND)
    include_directories(AFTER SYSTEM ${MKL_INCLUDE_DIR})
    list(APPEND Caffe2_PUBLIC_DEPENDENCY_LIBS caffe2::mkl)
    set(CAFFE2_USE_MKL ON)
  else()
    message(WARNING "MKL not found. Defaulting to Eigen")
    set(CAFFE2_USE_EIGEN_FOR_BLAS ON)
  endif()

elseif(BLAS STREQUAL "OpenBLAS")
  find_package(OpenBLAS REQUIRED)
  include_directories(SYSTEM ${OpenBLAS_INCLUDE_DIR})
  list(APPEND Caffe2_DEPENDENCY_LIBS ${OpenBLAS_LIB})

elseif(BLAS STREQUAL "Eigen")
  # Eigen is header-only
  set(CAFFE2_USE_EIGEN_FOR_BLAS ON)

# ... 6 more BLAS backends
endif()
```

**Default Strategy**:
- Desktop: Try MKL first, fallback to Eigen
- Mobile: Always use Eigen (header-only, lightweight)

### 4.3 Protobuf Handling

PyTorch builds its own protobuf to avoid version conflicts:

```cmake
# cmake/ProtoBuf.cmake

if(BUILD_CUSTOM_PROTOBUF)
  # Download and build protobuf from third_party/
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../third_party/protobuf)

  # Hide protobuf symbols (don't export them)
  if(BUILD_SHARED_LIBS AND CAFFE2_LINK_LOCAL_PROTOBUF)
    target_compile_definitions(libprotobuf PRIVATE LIBPROTOBUF_EXPORTS=0)
  endif()
else()
  # Use system protobuf
  find_package(Protobuf REQUIRED)
endif()
```

**Why Build Custom**: Avoid conflicts with system/user-installed protobuf versions

### 4.4 Third-Party Libraries

PyTorch vendors 50+ libraries in `third_party/`:

```
third_party/
├── benchmark/       # Google Benchmark
├── cub/             # CUDA utilities
├── cudnn_frontend/  # cuDNN API wrapper
├── cutlass/         # CUDA GEMM templates
├── eigen/           # Linear algebra
├── fbgemm/          # Facebook GEMM (quantization)
├── fmt/             # String formatting
├── foxi/            # ONNX importer
├── gloo/            # Distributed backend
├── googletest/      # Testing framework
├── ideep/           # Intel MKL-DNN wrapper
├── kineto/          # Profiling library
├── nccl/            # NVIDIA collective ops
├── onnx/            # ONNX protobuf definitions
├── pocketfft/       # FFT implementation
├── protobuf/        # Protocol buffers
├── psimd/           # Portable SIMD
├── pthreadpool/     # Thread pool
├── pybind11/        # Python bindings
├── sleef/           # SIMD math library
├── tensorpipe/      # Distributed transport
├── xnnpack/         # Neural net operators (mobile)
└── ... 30+ more
```

Each added via `add_subdirectory()` or ExternalProject

---

## 5. Build Targets and Libraries

### 5.1 Major Build Targets

```cmake
# Core library (C10)
add_library(c10 SHARED
  c10/core/Allocator.cpp
  c10/core/CPUAllocator.cpp
  c10/core/Device.cpp
  # ... 50+ core files
)

# CPU operators library
add_library(torch_cpu SHARED
  ${generated_sources}  # From torchgen
  aten/src/ATen/Context.cpp
  aten/src/ATen/Tensor.cpp
  # ... 1000+ operator implementations
)
target_link_libraries(torch_cpu PUBLIC c10)

# CUDA library (if USE_CUDA)
if(USE_CUDA)
  add_library(torch_cuda SHARED
    ${cuda_generated_sources}
    aten/src/ATen/cuda/CUDAContext.cpp
    aten/src/ATen/cuda/CUDAGeneratorImpl.cpp
    # ... CUDA kernels
  )
  target_link_libraries(torch_cuda PUBLIC torch_cpu CUDA::cudart)
endif()

# Python bindings
add_library(torch_python MODULE
  torch/csrc/Module.cpp
  torch/csrc/autograd/python_variable.cpp
  ${python_generated_sources}
  # ... Python binding code
)
target_link_libraries(torch_python PRIVATE torch_cpu)
```

### 5.2 Library Dependency Graph

```
                 ┌──────────────┐
                 │ torch_python │ (Python extension)
                 │  (_C.so)     │
                 └──────┬───────┘
                        │
         ┌──────────────┴──────────────┐
         │                             │
    ┌────▼─────┐              ┌────────▼────────┐
    │torch_cpu │              │   torch_cuda    │
    │  (.so)   │              │     (.so)       │
    └────┬─────┘              └────────┬────────┘
         │                             │
         └─────────────┬───────────────┘
                       │
                  ┌────▼────┐
                  │   c10   │ (Core primitives)
                  │  (.so)  │
                  └────┬────┘
                       │
          ┌────────────┴──────────────┬─────────────┐
          │                           │             │
    ┌─────▼──────┐            ┌───────▼──────┐  ┌──▼──────┐
    │   BLAS     │            │  protobuf    │  │  pthreads│
    │(MKL/OpenBL)│            │              │  │          │
    └────────────┘            └──────────────┘  └──────────┘
```

### 5.3 Per-Operator Headers (Optional)

When `USE_PER_OPERATOR_HEADERS=ON`:

```cmake
# Instead of single monolithic Functions.h:
#include <ATen/Functions.h>

# Use per-operator headers:
#include <ATen/ops/add.h>
#include <ATen/ops/mul.h>
```

**Benefits**:
- Faster incremental builds (change one operator → rebuild fewer files)
- Parallelizable compilation

**Tradeoff**:
- Slower full builds (more headers to parse)
- Higher disk usage

---

## 6. Build Workflows

### 6.1 Standard Python Build

```bash
# User-facing build command
python setup.py install

# What happens internally:
# 1. setup.py calls CMake with detected options
# 2. CMake configures build (find dependencies, run codegen)
# 3. Ninja/Make compiles C++ code
# 4. Python extension installed to site-packages
```

**Environment Variables** (influence CMake options):

```bash
USE_CUDA=1          # Enable CUDA
USE_CUDNN=1         # Enable cuDNN
USE_MKL=1           # Prefer MKL for BLAS
MAX_JOBS=8          # Parallel compilation jobs
DEBUG=1             # Debug build
REL_WITH_DEB_INFO=1 # Release with debug symbols
```

### 6.2 Developer Build

```bash
# CMake-based build (developers)
mkdir build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DUSE_CUDA=ON \
  -DUSE_CUDNN=ON \
  -DUSE_MKL=ON \
  -DBUILD_TEST=ON \
  -DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')

make -j$(nproc)
```

**Advantages**:
- Full control over options
- Easier debugging
- Faster incremental builds

### 6.3 Mobile/Cross-Compilation Build

```bash
# Android build (uses Gradle + CMake)
cd android
./gradlew :pytorch_android:assembleRelease

# iOS build
cd ios
xcodebuild -project TestApp.xcodeproj -sdk iphoneos

# Both invoke CMake with mobile-specific flags:
# -DINTERN_BUILD_MOBILE=ON
# -DUSE_CUDA=OFF
# -DUSE_MKL=OFF
# -DBLAS=Eigen
# -DBUILD_PYTHON=OFF
```

---

## 7. Build Optimizations

### 7.1 ccache Integration

```cmake
option(USE_CCACHE "Attempt using CCache" ON)

if(USE_CCACHE)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
  endif()
endif()
```

**Speedup**: 10-50x faster rebuilds for unchanged files

### 7.2 Precompiled Headers

```cmake
option(USE_PRECOMPILED_HEADERS "Use pre-compiled headers" OFF)

if(USE_PRECOMPILED_HEADERS)
  target_precompile_headers(torch_cpu PRIVATE
    <torch/csrc/python_headers.h>
    <ATen/ATen.h>
    <c10/core/TensorImpl.h>
  )
endif()
```

**Benefit**: Faster compilation for frequently-included headers

**Why OFF by default**: Can cause issues with incremental builds

### 7.3 Unity Builds

```cmake
set_target_properties(torch_cpu PROPERTIES
  UNITY_BUILD ON
  UNITY_BUILD_BATCH_SIZE 16
)
```

**Concept**: Combine multiple `.cpp` files into single translation unit

**Benefit**: Faster full builds (fewer compiler invocations)

**Tradeoff**: Slower incremental builds, higher memory usage

### 7.4 Link-Time Optimization (LTO)

```cmake
cmake_policy(SET CMP0069 NEW)  # Enable LTO policy

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  include(CheckIPOSupported)
  check_ipo_supported(RESULT ipo_supported)

  if(ipo_supported)
    set_property(TARGET torch_cpu PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  endif()
endif()
```

**Effect**: 10-20% performance improvement, significantly longer link times

---

## 8. Platform-Specific Build Details

### 8.1 macOS Metal (MPS) Build

```cmake
# cmake/Metal.cmake

if(USE_MPS)
  # Find Metal framework
  find_library(METAL_LIBRARY Metal)
  find_library(METALPERFORMANCESHADERS_LIBRARY MetalPerformanceShaders)
  find_library(METALPERFORMANCESHADERSGRAPH_LIBRARY MetalPerformanceShadersGraph)

  # Compile Metal shaders (.metal → .metallib)
  set(METAL_SOURCES
    aten/src/ATen/native/mps/kernels/Activations.metal
    aten/src/ATen/native/mps/kernels/BinaryOps.metal
    # ... 50+ Metal shaders
  )

  foreach(metal_src ${METAL_SOURCES})
    get_filename_component(metal_name ${metal_src} NAME_WE)
    set(metallib_file "${CMAKE_BINARY_DIR}/mps_kernels/${metal_name}.metallib")

    add_custom_command(
      OUTPUT ${metallib_file}
      COMMAND xcrun -sdk macosx metal -c ${metal_src} -o ${metal_name}.air
      COMMAND xcrun -sdk macosx metallib ${metal_name}.air -o ${metallib_file}
      DEPENDS ${metal_src}
    )

    list(APPEND METALLIB_FILES ${metallib_file})
  endforeach()

  # Embed Metal libraries in binary
  add_custom_target(mps_metallibs ALL DEPENDS ${METALLIB_FILES})
  add_dependencies(torch_cpu mps_metallibs)
endif()
```

**Metal Compilation**: `.metal` (source) → `.air` (intermediate) → `.metallib` (library)

### 8.2 CUDA Compilation

```cmake
# Special handling for CUDA files
set_source_files_properties(
  aten/src/ATen/native/cuda/Activation.cu
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON
)

# Per-file architecture flags (for advanced CUDA features)
function(_BUILD_FOR_ADDITIONAL_ARCHS file archs)
  torch_cuda_get_nvcc_gencode_flag(_existing_arch_flags)

  set(_file_compile_flags "")
  foreach(_arch ${archs})
    if("${_arch}" STREQUAL "90a")  # Hopper with FP8
      if(_existing_arch_flags MATCHES ".*compute_90.*")
        list(APPEND _file_compile_flags
          "-gencode;arch=compute_90a,code=sm_90a")
      endif()
    endif()
  endforeach()

  set_source_files_properties(${file}
    PROPERTIES COMPILE_FLAGS "${_file_compile_flags}")
endfunction()

# Example: Build FlashAttention for Hopper
_BUILD_FOR_ADDITIONAL_ARCHS(
  "aten/src/ATen/native/transformers/cuda/attention.cu"
  "90a")
```

**Rationale**: Some CUDA features (e.g., FP8) are architecture-specific

### 8.3 Cross-Compilation for Apple Silicon

```cmake
# Building for ARM64 on x86_64 Mac (or vice versa)
if(APPLE)
  set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Target architecture")
  # OR for universal binary:
  # set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64")

  # Ensure MPS is only enabled for ARM64
  if(CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
    set(MPS_ARCH_SUPPORTED ON)
  endif()
endif()
```

---

## 9. Build Artifacts and Installation

### 9.1 Generated Directory Structure

```
build/
├── aten/src/ATen/
│   ├── Functions.h                   # Generated operator declarations
│   ├── NativeFunctions.h             # Native function declarations
│   ├── Operators.h                   # Dispatcher operator declarations
│   ├── ops/                          # Per-operator headers (if enabled)
│   │   ├── add.h
│   │   ├── mul.h
│   │   └── ... 2,600+ operator headers
│   ├── RegisterCPU.cpp               # CPU dispatcher registration
│   ├── RegisterCUDA.cpp              # CUDA dispatcher registration
│   └── ... 1,000+ generated files
├── lib/
│   ├── libc10.so
│   ├── libtorch_cpu.so
│   ├── libtorch_cuda.so (if USE_CUDA)
│   └── _C.cpython-3.11-x86_64-linux-gnu.so  # Python extension
├── torch/
│   ├── _C.so -> ../lib/_C.cpython-...so
│   ├── lib/
│   │   ├── libc10.so
│   │   ├── libtorch_cpu.so
│   │   └── libtorch_cuda.so
│   └── include/
│       ├── ATen/
│       ├── c10/
│       └── torch/
└── bin/
    └── torch_shm_manager  # Shared memory utility
```

### 9.2 Installation Layout

```bash
# After `make install` or `python setup.py install`

/path/to/site-packages/torch/
├── __init__.py
├── _C.so                          # Main Python extension
├── lib/
│   ├── libc10.so
│   ├── libtorch_cpu.so
│   ├── libtorch_cuda.so
│   ├── libtorch_python.so
│   └── libnvfuser_codegen.so
├── include/                       # C++ headers (for extensions)
│   ├── ATen/
│   ├── c10/
│   └── torch/csrc/
├── share/cmake/                   # CMake config for downstream projects
│   └── Torch/
│       ├── TorchConfig.cmake
│       └── TorchConfigVersion.cmake
└── bin/
    └── torch_shm_manager
```

### 9.3 CMake Config Files

For downstream C++ projects:

```cmake
# TorchConfig.cmake (generated)
find_package(Torch REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app "${TORCH_LIBRARIES}")
set_property(TARGET my_app PROPERTY CXX_STANDARD 17)
```

Allows users to build against installed PyTorch:

```bash
cmake -DCMAKE_PREFIX_PATH=/path/to/site-packages/torch ..
```

---

## 10. Common Build Issues and Solutions

### 10.1 CUDA Architecture Mismatch

**Problem**: Running on GPU with architecture not built for

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**Solution**: Build for correct architecture

```bash
# For A100 (Ampere, sm_80)
CMAKE_CUDA_ARCHITECTURES=80 python setup.py install

# For multiple architectures
CMAKE_CUDA_ARCHITECTURES="70;75;80;86;89;90" python setup.py install
```

### 10.2 Out-of-Memory During Compilation

**Problem**: CUDA compilation uses too much RAM

```bash
c++: fatal error: Killed signal terminated program cc1plus
```

**Solution**: Reduce parallelism

```bash
MAX_JOBS=4 python setup.py install
```

Or disable CUDA:

```bash
USE_CUDA=0 python setup.py install
```

### 10.3 Code Generation Failures

**Problem**: Torchgen fails during build

**Solution**: Ensure Python environment is correct

```bash
# Install torchgen dependencies
pip install pyyaml typing_extensions

# Clean build
rm -rf build/
python setup.py clean
python setup.py install
```

### 10.4 Missing Dependencies

**Problem**: `find_package()` fails for BLAS/CUDA/etc.

**Solution**: Set `CMAKE_PREFIX_PATH`

```bash
# For MKL
export CMAKE_PREFIX_PATH=/opt/intel/mkl:$CMAKE_PREFIX_PATH

# For CUDA
export CMAKE_PREFIX_PATH=/usr/local/cuda:$CMAKE_PREFIX_PATH

python setup.py install
```

---

## 11. MLX Porting Recommendations

### 11.1 What to Port

**ADOPT**:

1. **CMake Structure**: Use CMake for build system
   - Industry standard for C++ projects
   - Good cross-platform support
   - Integration with Python via setuptools/scikit-build

2. **Code Generation Integration**: Integrate code generation into build
   - Use `add_custom_command()` pattern
   - Generate `.cmake` files listing outputs
   - Track dependencies (templates, YAML) properly

3. **BLAS Selection**: Support multiple BLAS backends
   - Apple Accelerate (for macOS)
   - OpenBLAS (for Linux)
   - Eigen (fallback)

4. **Metal Shader Compilation**: Compile `.metal` shaders in build
   - Use `xcrun metal` for compilation
   - Embed `.metallib` files in binary or ship alongside

5. **ccache Support**: Enable ccache for faster rebuilds

**SIMPLIFY**:

1. **Fewer Options**: MLX doesn't need 400 options
   - Start with 10-20 core options (USE_METAL, BUILD_PYTHON, BUILD_TESTS)
   - Add more as needed

2. **Fewer Backends**: MLX is Metal-focused
   - No CUDA/ROCm/XPU complexity
   - CPU fallback only

3. **Simpler Dependencies**: MLX has fewer third-party deps
   - No protobuf, gRPC, distributed libs
   - Minimal: pybind11/nanobind, Metal frameworks

### 11.2 What NOT to Port

**SKIP**:

1. **Multi-Backend Complexity**: CUDA/ROCm/Vulkan/XPU logic
2. **Distributed Infrastructure**: Gloo, NCCL, MPI, TensorPipe
3. **Mobile Builds**: Android/iOS complexity (unless needed)
4. **Legacy Compatibility**: Caffe2 remnants, old build modes

### 11.3 Recommended MLX Build Architecture

```cmake
# Simplified MLX CMakeLists.txt

cmake_minimum_required(VERSION 3.21)
project(MLX CXX)

set(CMAKE_CXX_STANDARD 17)

# Core options
option(BUILD_PYTHON "Build Python bindings" ON)
option(USE_METAL "Enable Metal backend" ON)
option(BUILD_TESTS "Build C++ tests" OFF)

# Find dependencies
find_package(Python REQUIRED COMPONENTS Interpreter Development)

if(USE_METAL)
  find_library(METAL_LIBRARY Metal REQUIRED)
  find_library(METAL_KIT MetalKit REQUIRED)
endif()

# Code generation
set(GEN_COMMAND
  "${Python_EXECUTABLE}" -m mlxgen.gen
  --source-path ${CMAKE_SOURCE_DIR}/mlx
  --install-dir ${CMAKE_BINARY_DIR}/mlx/generated
)

execute_process(
  COMMAND ${GEN_COMMAND} --dry-run
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

include(${CMAKE_BINARY_DIR}/mlx/generated/sources.cmake)

# Core library
add_library(mlx SHARED
  ${GENERATED_SOURCES}
  mlx/array.cpp
  mlx/ops.cpp
  # ... manual sources
)

if(USE_METAL)
  target_sources(mlx PRIVATE mlx/backend/metal/ops.mm)
  target_link_libraries(mlx PRIVATE ${METAL_LIBRARY})
endif()

# Python bindings (if enabled)
if(BUILD_PYTHON)
  add_subdirectory(python)
endif()
```

**Key Simplifications**:
- ~100 lines vs PyTorch's 900+
- 3 options vs 400+
- Single backend (Metal) vs 6+
- No complex dependency resolution

### 11.4 Metal Shader Build Pattern

```cmake
# Compile Metal shaders
file(GLOB_RECURSE METAL_SOURCES mlx/backend/metal/kernels/*.metal)

foreach(metal_src ${METAL_SOURCES})
  get_filename_component(metal_name ${metal_src} NAME_WE)
  set(metallib "${CMAKE_BINARY_DIR}/metal/${metal_name}.metallib")

  add_custom_command(
    OUTPUT ${metallib}
    COMMAND xcrun -sdk macosx metal -c ${metal_src} -o ${metal_name}.air
    COMMAND xcrun -sdk macosx metallib ${metal_name}.air -o ${metallib}
    DEPENDS ${metal_src}
    COMMENT "Compiling Metal shader ${metal_name}"
  )

  list(APPEND METAL_LIBS ${metallib})
endforeach()

# Embed Metal libraries (or install alongside .so)
add_custom_target(metal_shaders ALL DEPENDS ${METAL_LIBS})
add_dependencies(mlx metal_shaders)
```

---

## Summary

PyTorch's build system is **extremely complex** due to:
- Multi-backend support (6+ hardware backends)
- Cross-platform requirements (Linux/Windows/macOS/iOS/Android)
- Massive codebase (2.5M+ lines C++)
- Complex dependencies (50+ third-party libraries)
- Code generation integration (1000s of generated files)

**For MLX**:
- Adopt core CMake patterns (code generation, Metal shaders)
- Drastically simplify configuration (10-20 options max)
- Focus on macOS + Apple Silicon
- Leverage Apple Accelerate for BLAS
- Use Metal exclusively for GPU acceleration

The build system should be **simple, fast, and maintainable**—not PyTorch's 63k-line behemoth.
