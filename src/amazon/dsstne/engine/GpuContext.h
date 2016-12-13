/*
   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#ifndef GPU_CONTEXT_H
#define GPU_CONTEXT_H

#include <curand.h>
#include <memory>

#include "GpuTypes.h"

using std::unique_ptr;

struct GpuContext {
    enum SM_VERSION
    {
        SM_3X,
        SM_5X,
    };

    // Memory parameters
    GpuData                             _data;                      // All GPU data accessible from kernels (mostly device memory pointers)
    bool                                _bECCSupport;               // Flag for ECC support to detect Tesla versus consumer GPU
    bool                                _bCanMapHostMemory;         // Flag for pinned memory support
    aligned_lli                         _totalMemory;               // Total memory on GPU
    aligned_lli                         _totalCPUMemory;            // Approximate total allocated CPU memory
    aligned_lli                         _totalGPUMemory;            // Approximate total allocated CPU memory

    // SM/SMX parameters
    SM_VERSION                          _sm_version;                // SM revision
    unsigned int                        _threadsPerBlock;           // Default threads per block to launch
    unsigned int                        _warpSize;                  // Warp size (probably 32 but may change some day)
    unsigned int                        _warpBits;                  // Warp bit count
    unsigned int                        _warpMask;                  // Masks bits within a warp
    int                                 _numprocs;                  // Number of total processors in run
    int                                 _id;                        // Process ID
    int                                 _device;                    // Device ID

    // Fast sparse kernel limits
    uint32_t                            _maxSparse;                 // Maximum sparse boolean datapoints for sparse input layers
    uint32_t                            _maxSparseAnalog;           // Maximum sparse analog datapoints for sparse input layers

    // cuBLAS parameters
    cublasHandle_t                      _cuBLASHandle;              // Handle for cuBLAS state

    // cuRand parameters
    curandGenerator_t                   _RNG;                       // Handle for random number generator

    // cuDNN parameters
    cudnnHandle_t                       _cuDNNHandle;               // handle for cuDNN library

    // Neural network parameters
    NNNetwork*                          _pNetwork;                  // Pointer to current neural network
    unique_ptr<GpuBuffer<unsigned long long int>> _pbAccumulator;   // Pointer to per-kernel fix point accumulator
    bool                                _bCPUValidate;              // Should CPU validate GPU calculations?
    float                               _acceptableError;           // Acceptable error between CPU and GPU

    // Single-node multi-gpu parameters
    bool                                _bSingleNode;               // Flag to indicate MPI run is all on one node
    bool                                _bP2P;                      // Flag to indicate P2P connectivity between all processes

    cudaStream_t _streams[10];
    size_t _currentStream;

    // Methods
    GpuContext();
    ~GpuContext();
    void GetMemoryUsage(int* gpuMemory, int* cpuMemory);
    void SetRandomSeed(unsigned long seed);
    void SetNeuralNetwork(NNNetwork* pNetwork);
    void Startup(int argc, char** argv);
    void Shutdown();
    void CopyConstants();
    void SetCPUValidate(bool bCPUValidate);

    // Static methods
    static unsigned int Pad(unsigned int x);

    cudaStream_t getStream();
};

extern struct GpuContext& getGpu();

#endif // GPU_CONTEXT_H
