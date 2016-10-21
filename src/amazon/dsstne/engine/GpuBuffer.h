/*
   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#ifndef GPU_BUFFER_H
#define GPU_BUFFER_H

#include <cstring>

#include "GpuContext.h"

template <typename T>
struct GpuBuffer
{
    unsigned long long int  _length;
    bool                    _bSysMem;
    bool                    _bPinned;
    T*                      _pSysData;
    T*                      _pDevData;
    GpuBuffer(int length, bool bSysMem = false, bool bPinned = false);
    GpuBuffer(unsigned int length, bool bSysMem = false, bool bPinned = false);
    GpuBuffer(unsigned long long int length, bool bSysMem = false, bool bPinned = false);
    GpuBuffer(size_t length, bool bSysMem = false, bool bPinned = false);
    virtual ~GpuBuffer();
    void Allocate();
    void Deallocate();
    void Upload(T* pBuff = NULL);
    void Download(T * pBuff = NULL);
    void Copy(T* pBuff);
};

template <typename T>
GpuBuffer<T>::GpuBuffer(int length, bool bSysMem, bool bPinned) : _length(length), _bSysMem(bSysMem), _bPinned(bPinned), _pSysData(NULL), _pDevData(NULL)
{
    Allocate();
}

template <typename T>
GpuBuffer<T>::GpuBuffer(unsigned int length, bool bSysMem, bool bPinned) : _length(length), _bSysMem(bSysMem), _bPinned(bPinned), _pSysData(NULL), _pDevData(NULL)
{
    Allocate();
}

template <typename T>
GpuBuffer<T>::GpuBuffer(unsigned long long int length, bool bSysMem, bool bPinned) : _length(length), _bSysMem(bSysMem), _bPinned(bPinned), _pSysData(NULL), _pDevData(NULL)
{
    Allocate();
}

template <typename T>
GpuBuffer<T>::GpuBuffer(size_t length, bool bSysMem, bool bPinned) : _length(length), _bSysMem(bSysMem), _bPinned(bPinned), _pSysData(NULL), _pDevData(NULL)
{
    Allocate();
}

template <typename T>
GpuBuffer<T>::~GpuBuffer()
{
    Deallocate();
}

template <typename T>
void GpuBuffer<T>::Allocate()
{
#ifdef MEMTRACKING
    printf("Allocating %llu bytes of GPU memory", _length * sizeof(T));
    if (!_bSysMem && !_bPinned)
        printf(", unshadowed");
    else if (_bPinned)
        printf(", pinned");
    printf("\n");
#endif
    cudaError_t status;
    if (_bPinned)
    {
        status = cudaHostAlloc(reinterpret_cast<void **>(&_pSysData), _length * sizeof(T), cudaHostAllocMapped);
        RTERROR(status, "cudaHostalloc GpuBuffer::Allocate failed");
        getGpu()._totalCPUMemory                    += _length * sizeof(T);
        getGpu()._totalGPUMemory                    += _length * sizeof(T);
        status = cudaHostGetDevicePointer(reinterpret_cast<void **>(&_pDevData), reinterpret_cast<void *>(_pSysData), 0);
        RTERROR(status, "cudaGetDevicePointer GpuBuffer::failed to get device pointer");
        memset(_pSysData, 0, _length * sizeof(T));
    }
    else
    {
        if (_bSysMem)
        {
            _pSysData =     new T[_length];
            getGpu()._totalCPUMemory            +=  _length * sizeof(T);
            memset(_pSysData, 0, _length * sizeof(T));
        }

        status = cudaMalloc(reinterpret_cast<void **>(&_pDevData), _length * sizeof(T));
        getGpu()._totalGPUMemory                +=  _length * sizeof(T);
        RTERROR(status, "cudaMalloc GpuBuffer::Allocate failed");
        status = cudaMemset(reinterpret_cast<void *>(_pDevData), 0, _length * sizeof(T));
        RTERROR(status, "cudaMemset GpuBuffer::Allocate failed");
    }
#ifdef MEMTRACKING
    printf("Mem++: %llu %llu\n", getGpu()._totalGPUMemory, getGpu()._totalCPUMemory);
#endif
}

template <typename T>
void GpuBuffer<T>::Deallocate()
{
    cudaError_t status;
    if (_bPinned)
    {
        status = cudaFreeHost(_pSysData);
        getGpu()._totalCPUMemory                -=  _length * sizeof(T);
        getGpu()._totalGPUMemory                -=  _length * sizeof(T);
    }
    else
    {
        if (_bSysMem)
        {
            delete[] _pSysData;
            getGpu()._totalCPUMemory            -=  _length * sizeof(T);
        }
        status = cudaFree(_pDevData);
        getGpu()._totalGPUMemory                -=  _length * sizeof(T);
    }
    RTERROR(status, "cudaFree GpuBuffer::Deallocate failed");
    _pSysData = NULL;
    _pDevData = NULL;
#ifdef MEMTRACKING
    printf("Mem--: %lld %lld\n", getGpu()._totalGPUMemory, getGpu()._totalCPUMemory);
#endif
}

template <typename T>
void GpuBuffer<T>::Copy(T* pBuff)
{
    cudaError_t status;
    status = cudaMemcpy(_pDevData, pBuff, _length * sizeof(T), cudaMemcpyDeviceToDevice);
    RTERROR(status, "cudaMemcpy GpuBuffer::Upload failed");
}

template <typename T>
void GpuBuffer<T>::Upload(T* pBuff)
{
    if (pBuff)
    {
        cudaError_t status;
        status = cudaMemcpy(_pDevData, pBuff, _length * sizeof(T), cudaMemcpyHostToDevice);
        RTERROR(status, "cudaMemcpy GpuBuffer::Upload failed");
    }
    else if (_bSysMem)
    {
        cudaError_t status;
        status = cudaMemcpy(_pDevData, _pSysData, _length * sizeof(T), cudaMemcpyHostToDevice);
        RTERROR(status, "cudaMemcpy GpuBuffer::Upload failed");
    }
}

template <typename T>
void GpuBuffer<T>::Download(T* pBuff)
{
    if (pBuff)
    {
        cudaError_t status;
        status = cudaMemcpy(pBuff, _pDevData, _length * sizeof(T), cudaMemcpyDeviceToHost);
        RTERROR(status, "cudaMemcpy GpuBuffer::Download failed");
    }
    else if (_bSysMem)
    {
        cudaError_t status;
        status = cudaMemcpy(_pSysData, _pDevData, _length * sizeof(T), cudaMemcpyDeviceToHost);
        RTERROR(status, "cudaMemcpy GpuBuffer::Download failed");
    }
}

#endif // GPU_BUFFER_H
