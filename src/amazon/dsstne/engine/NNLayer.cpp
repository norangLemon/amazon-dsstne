/*


   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#include "NNLayer.h"

#include "GpuTypes.h"
#include "NNDataSetBase.h"
#include "NNLayerDescriptor.h"
#include "NNNetwork.h"
#include "NNWeight.h"
#include "kernels.h"
#define __STDC_FORMAT_MACROS
#include <inttypes.h>

using namespace netCDF;
using namespace netCDF::exceptions;

// NNLayer functions

NNLayer::NNLayer(NNLayerDescriptor& d, uint32_t batch) :
_name(d._name),
_kind(d._kind),
_type(d._type),
_poolingFunction(d._poolingFunction),
_dataSet(d._dataSet),
_pDataSet(NULL),
_vSource(d._vSource),
_vSkip(d._vSkip),
_pbUnit(NULL),
_pbDelta(NULL),
_pbDropout(NULL),
_Nx(d._Nx),
_Ny(d._Ny),
_Nz(d._Nz),
_Nw(d._Nw),
_dimensions(d._dimensions),
_weightInit(d._weightInit),
_weightInitScale(d._weightInitScale),
_biasInit(d._biasInit),
_kernelX(d._kernelX),
_kernelY(d._kernelY),
_kernelZ(d._kernelZ),
_kernelStrideX(d._kernelStrideX),
_kernelStrideY(d._kernelStrideY),
_kernelStrideZ(d._kernelStrideZ),
_kernelPaddingX(d._kernelPaddingX),
_kernelPaddingY(d._kernelPaddingY),
_kernelPaddingZ(d._kernelPaddingZ),
_kernelDimensions(d._kernelDimensions),
_weightNorm(d._weightNorm),
_deltaNorm(d._deltaNorm),
_pDropout(d._pDropout),
_activation(d._activation),
_oddBatch(0),
_bSparse(d._attributes & NNLayer::Attributes::Sparse),
_sparsenessPenalty_p(d._sparsenessPenalty_p),
_sparsenessPenalty_beta(d._sparsenessPenalty_beta),
_bDenoising(d._attributes & NNLayer::Attributes::Denoising),
_bFastSparse(false),
_bDirty(true),
_priority(-1),
_deltaUpdateCount(0),
_unitUpdateCount(0),
_batch(batch),
_localBatch(batch)
{  
    _stride                         = _Nx * _Ny * _Nz * _Nw;
    _parallelization                = Serial;

    // Model parallel settings
    _minX                           = ((size_t)_Nx * (size_t)getGpu()._id) / (size_t)getGpu()._numprocs;
    _maxX                           = ((size_t)_Nx * (size_t)(getGpu()._id + 1)) / (size_t)getGpu()._numprocs;
    _localStride                    = (_maxX - _minX) * _Ny * _Nz * _Nw;
    _maxLocalStride                 = (((size_t)_Nx + getGpu()._numprocs - 1) / (size_t)getGpu()._numprocs) * _Ny * _Nz * _Nw;
    
    // Allocate cuDNN tensor data if convolutional or pooling layer
    if ((_type == NNLayer::Type::Pooling) || (_type == NNLayer::Type::Convolutional))
    {
        cudnnStatus_t cudnnStatus   = cudnnCreateTensorDescriptor(&_tensorDescriptor);
        CUDNNERROR(cudnnStatus, "NNLayer::NNLayer: unable to create _tensordescriptor");
        cudnnStatus                 = cudnnCreateTensorDescriptor(&_oddBatchTensorDescriptor);
        CUDNNERROR(cudnnStatus, "NNLayer::NNLayer: unable to create _oddBatchTensordescriptor");        
    }
    
    // Allocate cuDNN pooling descriptor for pooling layers
    if (_type == NNLayer::Type::Pooling)
    {
        cudnnStatus_t cudnnStatus = cudnnCreatePoolingDescriptor(&_poolingDescriptor);
        CUDNNERROR(cudnnStatus, "NNLayer::NNLayer: unable to create pooling descriptor");
        vector<int> vKernel(3);
        vector<int> vKernelPadding(3);
        vector<int> vKernelStride(3);
        vKernel[0]                  = _kernelX;
        vKernel[1]                  = _kernelY;
        vKernel[2]                  = _kernelZ;
        vKernelPadding[0]           = _kernelPaddingX;
        vKernelPadding[1]           = _kernelPaddingY;
        vKernelPadding[2]           = _kernelPaddingZ;
        vKernelStride[0]            = _kernelStrideX;
        vKernelStride[1]            = _kernelStrideY;
        vKernelStride[2]            = _kernelStrideZ;
        
        switch (_poolingFunction)
        {
            case PoolingFunction::Max:
                cudnnSetPoolingNdDescriptor(_poolingDescriptor,
                                           CUDNN_POOLING_MAX,
                                           CUDNN_PROPAGATE_NAN,
                                           _kernelDimensions,
                                           vKernel.data(),
                                           vKernelPadding.data(),
                                           vKernelStride.data());
                CUDNNERROR(cudnnStatus, "NNLayer::NNLayer: unable to set max pooling descriptor");
                break;

            case PoolingFunction::Average:
                cudnnSetPoolingNdDescriptor(_poolingDescriptor,
                                           CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
                                           CUDNN_PROPAGATE_NAN,
                                           _kernelDimensions,
                                           vKernel.data(),
                                           vKernelPadding.data(),
                                           vKernelStride.data());
                CUDNNERROR(cudnnStatus, "NNLayer::NNLayer: unable to set average pooling descriptor");
                
            case PoolingFunction::LRN:
                cudnnStatus         = cudnnCreateLRNDescriptor(&_LRNDescriptor);
                CUDNNERROR(cudnnStatus, "NNLayer::NNLayer: unable to create LRN descriptor");
                break;
        }        
    }
}

NNLayer::~NNLayer()
{
    Deallocate();
    // Deallocate cuDNN tensor data if convolutional or pooling layer
    if ((_type == NNLayer::Type::Pooling) || (_type == NNLayer::Type::Convolutional))
    {
        // Delete tensor descriptors
        cudnnStatus_t cudnnStatus       = cudnnDestroyTensorDescriptor(_tensorDescriptor);
        CUDNNERROR(cudnnStatus, "NNLayer::~NNLayer: unable to delete _tensorDescriptor");        
        cudnnStatus                     = cudnnDestroyTensorDescriptor(_oddBatchTensorDescriptor);
        CUDNNERROR(cudnnStatus, "NNLayer::~NNLayer: unable to delete _oddBatchTensorDescriptor");  
    }
    
    // Delete pooling layer-specific stuff
    if (_type == NNLayer::Type::Pooling)
    {
        cudnnStatus_t cudnnStatus       = cudnnDestroyPoolingDescriptor(_poolingDescriptor);
        CUDNNERROR(cudnnStatus, "NNLayer::~NNLayer: unable to destroy _poolingDescriptor");
        
        if (_poolingFunction == PoolingFunction::LRN)
        {
            // Delete LRN descriptor
            cudnnStatus_t cudnnStatus   = cudnnDestroyLRNDescriptor(_LRNDescriptor);
            CUDNNERROR(cudnnStatus, "NNLayer::~NNLayer: unable to delete _LRNDescriptor");
        }
    }
}

void NNLayer::Deallocate()
{
    if (getGpu()._id == 0)
        printf("NNLayer::Allocate: Deallocating all data for layer %s\n", _name.c_str());

    // Recklessly delete everything because the standard says you can...
    delete _pbUnit;
    _pbUnit                     = NULL;
    delete _pbDelta;
    _pbDelta                    = NULL;
    delete _pbDropout;
    _pbDropout                  = NULL;
}

cudnnTensorDescriptor_t NNLayer::getTensorDescriptor(uint32_t batch)
{
    // Return usual tensor descriptor if regular batch
    if (batch == _batch)
    {
        return _tensorDescriptor;
    }
    
    // Return odd batch for ends of epochs or resize
    // here for a one-shot
    else if (batch != _oddBatch)
    {
        cudnnStatus_t cudnnStatus;
        vector<int> vDimensions(5, 1);
        vector<int> vStride(5, 1);
        switch (_dimensions)
        {
            case 2:
                vDimensions[0]      = batch;
                vDimensions[1]      = _Ny;
                vDimensions[2]      = _Nx;
                vStride[2]          = 1;
                vStride[1]          = _Nx;
                vStride[0]          = _Nx * _Ny;                
                cudnnStatus         = cudnnSetTensorNdDescriptor(_oddBatchTensorDescriptor, CUDNN_DATA_FLOAT, _dimensions + 1, vDimensions.data(), vStride.data());
                break;
            
            case 3:
                cudnnStatus         = cudnnSetTensor4dDescriptor(_oddBatchTensorDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, _Nz, _Ny, _Nx);
                break;
                
            case 4:
                vDimensions[0]      = batch;
                vDimensions[1]      = _Nw;
                vDimensions[2]      = _Nz;
                vDimensions[3]      = _Ny;
                vDimensions[4]      = _Nx;
                vStride[4]          = 1;
                vStride[3]          = _Nx;
                vStride[2]          = _Nx * _Ny;
                vStride[1]          = _Nx * _Ny * _Nz;
                vStride[0]          = _Nx * _Ny * _Nz * _Nw;                                             
                cudnnStatus         = cudnnSetTensorNdDescriptor(_tensorDescriptor, CUDNN_DATA_FLOAT, _dimensions + 1, vDimensions.data(), vStride.data());
                break;
        }
        CUDNNERROR(cudnnStatus, "NNLayer::Allocate: Unable to set oddBatchTensorDescriptor");
        _oddBatch = batch;
    }

    return _oddBatchTensorDescriptor;
}

tuple<uint32_t, uint32_t, uint32_t, uint32_t> NNLayer::GetDimensions()
{
    return make_tuple(_Nx, _Ny, _Nz, _Nw);
}

tuple<uint32_t, uint32_t, uint32_t, uint32_t> NNLayer::GetLocalDimensions()
{
    return make_tuple(_maxX - _minX, _Ny, _Nz, _Nw);
}

tuple<uint32_t, uint32_t, uint32_t> NNLayer::GetKernelDimensions()
{
    return make_tuple(_kernelX, _kernelY, _kernelZ);
}

tuple<uint32_t, uint32_t, uint32_t> NNLayer::GetKernelStride()
{
    return make_tuple(_kernelStrideX, _kernelStrideY, _kernelStrideZ);
}


static void DumpTensor(cudnnTensorDescriptor_t t)
{
    cudnnDataType_t dataType;
    int ndims;
    vector<int> vDim(16);
    vector<int> vStride(16);
    cudnnStatus_t cudnnStatus = cudnnGetTensorNdDescriptor(t, 8, &dataType, &ndims, vDim.data(), vStride.data());
    CUDNNERROR(cudnnStatus, "cudnnGetTensorNdDescriptor error");    
    cout << "Tensor:   " << ndims << " dimensions" << endl;
    cout << "DataType: " << dataType << endl;
    for (int i = 0; i < ndims; i++)
        cout << i << " " << vDim[i] << " " << vStride[i] << endl;
    cout << endl;
    
}

void NNLayer::Allocate(bool validate)
{
    Deallocate();
    uint64_t size                   = (uint64_t)_maxLocalStride * (uint64_t)_localBatch; 
        
    // Set tensor descriptor if pooling or convolutional layer
    if ((_type == NNLayer::Type::Pooling) || (_type == NNLayer::Type::Convolutional))
    {
        cudnnStatus_t cudnnStatus;
        vector<int> vDimensions(5, 1);
        vector<int> vStride(5, 1);
        switch (_dimensions)
        {
            case 2:
                vDimensions[0]      = _localBatch;
                vDimensions[1]      = _Ny;
                vDimensions[2]      = _Nx;
                vStride[2]          = 1;
                vStride[1]          = _Nx;
                vStride[0]          = _Nx * _Ny;                
                cudnnStatus         = cudnnSetTensorNdDescriptor(_tensorDescriptor, CUDNN_DATA_FLOAT, _dimensions + 1, vDimensions.data(), vStride.data());
                break;
            
            case 3:
                cudnnStatus         = cudnnSetTensor4dDescriptor(_tensorDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, _localBatch, _Nz, _Ny, _Nx);
                break;
                
            case 4:
                vDimensions[0]      = _localBatch;
                vDimensions[1]      = _Nw;
                vDimensions[2]      = _Nz;
                vDimensions[3]      = _Ny;
                vDimensions[4]      = _Nx;
                vStride[4]          = 1;
                vStride[3]          = _Nx;
                vStride[2]          = _Nx * _Ny;
                vStride[1]          = _Nx * _Ny * _Nz;
                vStride[0]          = _Nx * _Ny * _Nz * _Nw;                           
                cudnnStatus         = cudnnSetTensorNdDescriptor(_tensorDescriptor, CUDNN_DATA_FLOAT, _dimensions + 1, vDimensions.data(), vStride.data());
                break;
        }
        CUDNNERROR(cudnnStatus, "NNLayer::Allocate: Unable to set tensor descriptor");
        DumpTensor(_tensorDescriptor);
    }
    
    // Allocate hidden unit data for hidden and output layers and for non-sparse input layers
    if (!_bSparse || !_bFastSparse || (_kind != Input)
        || (_bSparse && (_kind == Input) && validate) // only for validation
    )
    {
        _vUnit.resize(size);
        _pbUnit                     = new GpuBuffer<NNFloat>(size);    
        if (getGpu()._id == 0)
            printf("NNLayer::Allocate: Allocating %" PRIu64 " bytes (%u, %u) of unit data for layer %s\n", size * sizeof(NNFloat), _maxLocalStride, _localBatch, _name.c_str());
    }

    // Allocate delta data for non-input layers
    if (_kind != Input)
    {
        _vDelta.resize(size);
        _pbDelta                    = new GpuBuffer<NNFloat>(size);
        if (getGpu()._id == 0)       
            printf("NNLayer::Allocate: Allocating %" PRIu64 " bytes (%u, %u) of delta data for layer %s\n", size * sizeof(NNFloat), _maxLocalStride, _localBatch, _name.c_str());        
    }
    
    // Allocate dropout data if active
    if (_pDropout > (NNFloat)0.0)
    {
        _pbDropout                  = new GpuBuffer<NNFloat>(size);
        if (getGpu()._id == 0)        
            printf("NNLayer::Allocate: Allocating %" PRIu64 " bytes (%u, %u) of dropout data for layer %s\n", size * sizeof(NNFloat), _maxLocalStride, _localBatch, _name.c_str());
    } 
    _bDirty                         = false;
}

void NNLayer::SetBatch(uint32_t batch)
{
    if (batch != _batch)
    {
        _batch                      = batch;
        if (_parallelization == NNLayer::Parallelization::Data)
            _localBatch             = batch / getGpu()._numprocs;
        else
            _localBatch             = batch;
        _bDirty                     = true;
    }
}

void NNLayer::RefreshParallelization()
{
    uint32_t convolutionalInputs = 0;
    uint32_t fullyConnectedInputs = 0;
    uint32_t poolingInputs = 0;
    uint32_t convolutionalOutputs = 0;
    uint32_t fullyConnectedOutputs = 0;
    uint32_t poolingOutputs = 0;    
    
    // Count number of inputs and outputs of each type
    for (auto l : _vIncomingLayer)
    {
        switch (l->_type)
        {
            case NNLayer::Type::Pooling:
                poolingInputs++;
                break;
            
            case NNLayer::Type::FullyConnected:
                fullyConnectedInputs++;
                break;
                
            case NNLayer::Type::Convolutional:
                convolutionalInputs++;
                break;
        }
    }
    
    for (auto l : _vOutgoingLayer)
    {
        switch (l->_type)
        {
            case NNLayer::Type::Pooling:
                poolingOutputs++;
                break;
                
            case NNLayer::Type::FullyConnected:
                fullyConnectedOutputs++;
                break;
                
            case NNLayer::Type::Convolutional:
                convolutionalOutputs++;
                break;
        }
    }
    
    switch (_kind)
    {
        // Input layer parallelization based on outputs
        case NNLayer::Kind::Input:
            if (convolutionalOutputs > 0)
                _parallelization = NNLayer::Parallelization::Data;
            else
                _parallelization = NNLayer::Parallelization::Model;
            break;
    
        // Output layer parallelization based on inputs
        case NNLayer::Kind::Output:
            if (convolutionalInputs > 0)
                _parallelization = NNLayer::Parallelization::Data;
            else
                _parallelization = NNLayer::Parallelization::Model;
            break;
        
        // Hidden Layer based on convolution, pooling, or fully-connected type, with possible transition post-activation
        // and post-delta calculation if outputs are of the other type
        case NNLayer::Hidden:
            // Fully connected layers are always model-parallel with possible incoming transposition
            if (_type == NNLayer::Type::FullyConnected)
            {    
                _parallelization = NNLayer::Parallelization::Model;
                if (convolutionalOutputs > 0)
                    _bTransposeParallelization = true;
            }
            
            // Pooling layer based on inputs, with possible transition post-activation
            // and post-delta calculation if outputs are of the other type
            else if (_type == NNLayer::Type::Pooling)
            {
                if (convolutionalInputs > 0)
                {
                    _parallelization = NNLayer::Parallelization::Data;
                    if (fullyConnectedOutputs > 0)
                        _bTransposeParallelization = true;
                }
                else
                {
                    _parallelization = NNLayer::Parallelization::Model;
                    if (convolutionalOutputs > 0)
                        _bTransposeParallelization = true;                
                }
            }
            
            // Otherwise a convolution layer, data-parallel with possible incoming transposition
            else
            {
                _parallelization = NNLayer::Parallelization::Data;                
                 if (fullyConnectedOutputs > 0)
                    _bTransposeParallelization = true;
            }
            break;
    }
}

void NNLayer::RefreshState(NNNetwork* pNetwork, bool validate)
{
    if (_bDirty)
    {
        // First test for fast sparse kernel compatibility if sparse input layer
        _bFastSparse                = false;
        if ((_kind == Input) && (_pDataSet != NULL) && (_bSparse))
        {
            uint32_t maxSparse      = (_pDataSet->_attributes & NNDataSetEnums::Attributes::Boolean) ? getGpu()._maxSparse : getGpu()._maxSparseAnalog;
            if (_batch > maxSparse)
            {
                if (getGpu()._id == 0)
                    printf("NNLayer::RefreshState: Batch size (%u) is too high to use fast sparse kernels on input layer %s\n", _batch, _name.c_str());    
            }
            else if (_pDataSet->_maxSparseDatapoints > maxSparse)
            {
                if (getGpu()._id == 0)
                    printf("NNLayer::RefreshState: Maximum sparse datapoints per example (%u) is too high to use fast sparse kernels on input layer %s\n", _pDataSet->_maxSparseDatapoints, _name.c_str());  
            }
            else if (_pDataSet->_sparseDensity > (NNFloat)0.1)
            {
                 if (getGpu()._id == 0)
                    printf("NNLayer::RefreshState: Sparse density per (%.2f) is too high to use fast sparse kernels on input layer %s\n", _pDataSet->_sparseDensity, _name.c_str());                 
            }
            else
            {
                _bFastSparse        = true;
            }
        }
        
        // Determine parallelization strategy
        if (getGpu()._numprocs > 1)
            RefreshParallelization();

        Allocate(validate);

        // Shard data set if necessary
        if ((_kind != Hidden) && (_pDataSet != NULL))
        {
            if (_type == FullyConnected)
            {
                _pDataSet->Shard(NNDataSetEnums::Model);
            }
            else if (_type == Convolutional)
            {
                _pDataSet->Shard(NNDataSetEnums::Data);
            }
        }
        _bDirty                     = false;
    }

    // Turn on/off denoising if active for input layers
    if ((_kind == Input) && _pDataSet)
        _pDataSet->SetDenoising(_bDenoising);
        
    // Set up LRN descriptor if active
    if ((_type == NNLayer::Type::Pooling) && (_poolingFunction == PoolingFunction::LRN))
    {
        cudnnStatus_t status = cudnnSetLRNDescriptor(_LRNDescriptor,
                                                    pNetwork->_LRN_n,
                                                    pNetwork->_LRN_alpha,
                                                    pNetwork->_LRN_beta,
                                                    pNetwork->_LRN_k);
        CUDNNERROR(status, "NNLayer::RefreshState: unable to set LRN descriptor");
    }
}

void NNLayer::ClearUpdates()
{
    _unitUpdateCount                = 0;
    _deltaUpdateCount               = 0;
}

void NNLayer::LoadPredictionBatch(uint32_t position, uint32_t batch)
{

    if (_kind == Input)
    { 
        if (!_bSparse)
        {
            _pDataSet->LoadInputUnit(position, batch, _localStride, _pbUnit->_pDevData);
        }
        else if (!_bFastSparse)
        {
            _pDataSet->LoadSparseInputUnit(position, batch, _localStride, _pbUnit->_pDevData);
        }
    }
}

void NNLayer::LoadTrainingBatch(uint32_t position, uint32_t batch)
{
    if (_kind == Input)
    { 
        if (_bSparse)
        {
            if (_bFastSparse)
            {
                if (_bDenoising)
                {
                    _pDataSet->CalculateSparseTransposedDenoisedMatrix(position, batch, this);
                }
                else
                {
                    _pDataSet->CalculateSparseTransposedMatrix(position, batch, this);
                }
            }
            else
            {
                if (_bDenoising)
                {
                    _pDataSet->LoadSparseDenoisedInputUnit(position, batch, _localStride, _pbUnit->_pDevData);    
                }
                else
                {
                    _pDataSet->LoadSparseInputUnit(position, batch, _localStride, _pbUnit->_pDevData);  
                }               
            }
        }
        else
        {
            _pDataSet->LoadInputUnit(position, batch, _localStride, _pbUnit->_pDevData);
            
            // Apply dropout if active
            if (_pDropout > (NNFloat)0.0)
                CalculateDropout(batch);    
        }
    }
}

void NNLayer::LoadValidationBatch(uint32_t position, uint32_t batch)
{
    if (_kind == Input)
    { 
        if (_bSparse)
        {
            _pDataSet->LoadSparseInputUnit(position, batch, _localStride, _pbUnit->_pDevData);
            _pDataSet->CalculateSparseTransposedMatrix(position, batch, this);
        }
        else
        {
            _pDataSet->LoadInputUnit(position, batch, _localStride, _pbUnit->_pDevData);
        }
    }
}

void NNLayer::GenerateDenoisingData()
{
    if (_pDataSet)
        _pDataSet->GenerateDenoisingData();
}

void NNLayer::ForwardPropagate(uint32_t position, uint32_t batch, bool bTraining)
{
    
    // Will switch to class-based decision shortly once working
    switch (_type)
    {
        case FullyConnected:
            ForwardPropagateFullyConnected(position, batch, bTraining);
            break;
            
        case Convolutional:
            ForwardPropagateConvolutional(position, batch, bTraining);
            break;
            
        case Pooling:
            ForwardPropagatePooling(position, batch, bTraining);
            break;                        
        
    }
}
    
    
void NNLayer::ForwardPropagateFullyConnected(uint32_t position, uint32_t batch, bool bTraining)
{    
    // Single GPU is the simplest case
    if (getGpu()._numprocs == 1)
    {
        if (_kind != Input)
        {         
            // Initialize units to bias values
            switch (_vIncomingLayer.size())
            {
                case 0: // Only skip layers
                    cudaMemset(_pbUnit->_pDevData, 0, _stride * batch * sizeof(NNFloat));
                    break;
                    
                case 1:
                    kClearUnit(_pbUnit->_pDevData, _vIncomingWeight[0]->_pbBias->_pDevData, _stride, batch);
                    break; 
                    
                case 2:
                    kClearDualSourceUnit(_pbUnit->_pDevData, _vIncomingWeight[0]->_pbBias->_pDevData, 
                                                             _vIncomingWeight[1]->_pbBias->_pDevData, 
                                        _stride, batch);
                    break;                   
                    
                case 3:
                    kClearTripleSourceUnit(_pbUnit->_pDevData, _vIncomingWeight[0]->_pbBias->_pDevData, 
                                                               _vIncomingWeight[1]->_pbBias->_pDevData, 
                                                               _vIncomingWeight[2]->_pbBias->_pDevData, 
                                        _stride, batch);
                    break;      

                case 4:
                    kClearQuadSourceUnit(_pbUnit->_pDevData, _vIncomingWeight[0]->_pbBias->_pDevData, 
                                                             _vIncomingWeight[1]->_pbBias->_pDevData, 
                                                             _vIncomingWeight[2]->_pbBias->_pDevData, 
                                                             _vIncomingWeight[3]->_pbBias->_pDevData, 
                                        _stride, batch);
                    break;                  
                    
                default:
                    if (getGpu()._id == 0)
                        printf("NNLayer::ForwardPropagate: Too many input layers for network layer %s\n", _name.c_str());          
                    getGpu().Shutdown();
                    exit(-1);
                    break; 
            }
        
        
            const NNFloat sgemm_beta                = (NNFloat)1.0;
            for (uint32_t i = 0; i < _vIncomingLayer.size(); i++)
            {
                // Special case sparse input layers with sparse matrix * matrix kernel
                if (_vIncomingLayer[i]->_bFastSparse)
                {
                    NNFloat* pWeight                = _vIncomingWeight[i]->_bShared ? 
                                                      _vIncomingWeight[i]->_pSharedWeight->_pbWeight->_pDevData : 
                                                      _vIncomingWeight[i]->_pbWeight->_pDevData;
                    if (bTraining && _vIncomingLayer[i]->_bDenoising)
                        _vIncomingLayer[i]->_pDataSet->CalculateSparseDenoisedZ(position, batch, _stride, pWeight, _pbUnit->_pDevData, sgemm_beta);  
                    else
                        _vIncomingLayer[i]->_pDataSet->CalculateSparseZ(position, batch, _stride, pWeight, _pbUnit->_pDevData, sgemm_beta);
                }
                else      
                {
                    const NNFloat sgemm_alpha       = (NNFloat)1.0;
                    cublasStatus_t cstatus;
                    NNFloat* pA                     = _vIncomingLayer[i]->_pbUnit->_pDevData;
                    NNFloat* pB                     = _vIncomingWeight[i]->_bShared ? 
                                                      _vIncomingWeight[i]->_pSharedWeight->_pbWeight->_pDevData : 
                                                      _vIncomingWeight[i]->_pbWeight->_pDevData;
                    NNFloat* pC                     = _pbUnit->_pDevData;
                    int m                           = batch;
                    int n                           = _localStride;
                    int k                           = _vIncomingLayer[i]->_stride;
                    int lda                         = _vIncomingWeight[i]->_bTransposed ? k : n;
                    int ldb                         = k;
                    int ldc                         = n;

                    cstatus                         =
                                                    cublasSgemm(getGpu()._cuBLASHandle, 
                                                    _vIncomingWeight[i]->_bTransposed ? CUBLAS_OP_T : CUBLAS_OP_N,
                                                    CUBLAS_OP_N,
                                                    n,
                                                    m,
                                                    k,
                                                    &sgemm_alpha,
                                                    pB,
                                                    lda,
                                                    pA,
                                                    ldb,
                                                    &sgemm_beta,
                                                    pC,
                                                    ldc);  

                    // Make sure matrix multiply succeeded
                    if (cstatus != CUBLAS_STATUS_SUCCESS)
                    {
                        if (getGpu()._id == 0)
                            printf("NNLayer::ForwardPropagate: SGEMM failure, aborting, status %d.\n", cstatus);
                        getGpu().Shutdown();
                        exit(-1);
                    }
                }
            }

            // Copy data from incoming skip layers
            for (auto l : _vIncomingSkip)
            {
                kAddBuffers(_pbUnit->_pDevData, l->_pbUnit->_pDevData, batch * _stride);
            }
           
            // Calculate activation
            CalculateActivation(batch);
            
            // Apply dropout if active
            if (bTraining && (_pDropout > (NNFloat)0.0))
                CalculateDropout(batch);             
       
#if 0
        string fname = "activation_" + _name;
        Dump(fname, _pbUnit->_pDevData);
#endif              
        }       
    }
    else
    {
        if (_kind != Input)
        {              
            // Calculate activations from incoming larger layers locally, then reduce the result
            // to the appropriate process, picking up contribution from each process
            if (_vIncomingLargerLayer.size() > 0)
            {
                NNFloat sgemm_beta                  = (NNFloat)0.0;
                for (uint32_t i = 0; i < _vIncomingLargerLayer.size(); i++)
                {
                    NNLayer* pInputLayer            = _vIncomingLargerLayer[i];
                    NNFloat* pWeight                = _vIncomingLargerWeight[i]->_bShared ? 
                                                      _vIncomingLargerWeight[i]->_pSharedWeight->_pbWeight->_pDevData : 
                                                      _vIncomingLargerWeight[i]->_pbWeight->_pDevData;                                           

                    if (pInputLayer->_bFastSparse)
                    {
                        if (bTraining && pInputLayer->_bDenoising)
                            pInputLayer->_pDataSet->CalculateSparseDenoisedZ(position, batch, _stride, pWeight, getGpu()._pNetwork->GetP2PSendBuffer(), sgemm_beta);  
                        else
                            pInputLayer->_pDataSet->CalculateSparseZ(position, batch, _stride, pWeight, getGpu()._pNetwork->GetP2PSendBuffer(), sgemm_beta);  
                    }
                    else
                    {
                
                        // Calculate local SGEMM
                        const NNFloat sgemm_alpha   = (NNFloat)1.0;

                        NNFloat* pA                 = pWeight;
                        NNFloat* pB                 = _vIncomingLargerLayer[i]->_pbUnit->_pDevData;
                        NNFloat* pC                 = getGpu()._pNetwork->GetP2PSendBuffer();
                        int m                       = _stride;
                        int n                       = batch;
                        int k                       = pInputLayer->_localStride;
                        int lda                     = _stride;
                        int ldb                     = pInputLayer->_localStride;
                        int ldc                     = _stride;

                        cublasStatus_t cstatus      =
                                                    cublasSgemm(getGpu()._cuBLASHandle, 
                                                    CUBLAS_OP_N,
                                                    CUBLAS_OP_N,
                                                    m,
                                                    n,
                                                    k,
                                                    &sgemm_alpha,
                                                    pA,
                                                    lda,
                                                    pB,
                                                    ldb,
                                                    &sgemm_beta,
                                                    pC,
                                                    ldc);  

                        // Make sure matrix multiply succeeded
                        if (cstatus != CUBLAS_STATUS_SUCCESS)
                        {
                            if (getGpu()._id == 0)
                                printf("NNLayer::ForwardPropagate: SGEMM failure, aborting, status %d.\n", cstatus);
                            getGpu().Shutdown();
                            exit(-1);
                        }                                     
                    }
                    
                    // Accumulate subsequent calculations if active
                    sgemm_beta                      = (NNFloat)1.0;
                }
                //printf("FP %s IL UC %d\n", _name.c_str(), _unitUpdateCount);

                // Reduce output
                Reduce(batch, _stride, _pbUnit->_pDevData, _localStride, _unitUpdateCount);
                _unitUpdateCount++;
            }
            
            // Copy data from incoming skip layers
            for (auto l : _vIncomingSkip)
            {
                kAddBuffers(_pbUnit->_pDevData, l->_pbUnit->_pDevData, batch * _localStride);
            }            
                   
            // Add biases and calculate activations
            switch (_vIncomingLayer.size())
            {
                case 0: // Only skip layers
                    break;
                
                case 1:
                    kAddBias(_pbUnit->_pDevData, _vIncomingWeight[0]->_pbBias->_pDevData, _localStride, batch);
                    break; 
                        
                case 2:
                    kAddDualBias(_pbUnit->_pDevData, _vIncomingWeight[0]->_pbBias->_pDevData, 
                                                     _vIncomingWeight[1]->_pbBias->_pDevData, _localStride, batch);
                    break;                   
                        
                case 3:
                    kAddTripleBias(_pbUnit->_pDevData, _vIncomingWeight[0]->_pbBias->_pDevData, 
                                                       _vIncomingWeight[1]->_pbBias->_pDevData, 
                                                       _vIncomingWeight[2]->_pbBias->_pDevData, _localStride, batch);
                    break;      

                case 4:
                    kAddQuadBias(_pbUnit->_pDevData, _vIncomingWeight[0]->_pbBias->_pDevData, 
                                                     _vIncomingWeight[1]->_pbBias->_pDevData, 
                                                     _vIncomingWeight[2]->_pbBias->_pDevData, 
                                                     _vIncomingWeight[3]->_pbBias->_pDevData, _localStride, batch);
                    break;                  
                        
                default:
                    if (getGpu()._id == 0)
                        printf("NNLayer::ForwardPropagate: Too many input layers for network layer %s\n", _name.c_str());
                    getGpu().Shutdown();
                    exit(-1);
                    break; 
            }    
                                      
            // Calculate activation
            CalculateActivation(batch);   
            
            // Apply dropout if active
            if (bTraining && (_pDropout > (NNFloat)0.0))
                CalculateDropout(batch);  
        }
        

#if 0
        string fname = "activation_" + _name;
        Dump(fname, _pbUnit->_pDevData);
#endif                                      
        // Circulate activations to outgoing larger layers
        if (_vOutgoingLargerLayer.size() > 0)
        {  
        
            if (_bFastSparse)
            {
                for (uint32_t i = 0; i < _vOutgoingLargerLayer.size(); i++)
                {
                    NNLayer* pOutputLayer       = _vOutgoingLargerLayer[i];
                    NNFloat* pWeight            = _vOutgoingLargerWeight[i]->_bShared ? 
                                                  _vOutgoingLargerWeight[i]->_pSharedWeight->_pbWeight->_pDevData : 
                                                  _vOutgoingLargerWeight[i]->_pbWeight->_pDevData;
                    const NNFloat sgemm_beta    = (pOutputLayer->_unitUpdateCount == 0) ? (NNFloat)0.0 : (NNFloat)1.0;
                    
                    if (bTraining && _bDenoising)
                        _pDataSet->CalculateSparseDenoisedZ(position, batch, pOutputLayer->_localStride, pWeight, pOutputLayer->_pbUnit->_pDevData, sgemm_beta);  
                    else
                        _pDataSet->CalculateSparseZ(position, batch, pOutputLayer->_localStride, pWeight, pOutputLayer->_pbUnit->_pDevData, sgemm_beta);
                }
            }
            else
            {
        
                // Gather inputs to this layer
                Gather(batch, _stride, _pbUnit->_pDevData, _localStride);

                // Calculate contributions to all outgoing X(L)
                for (uint32_t i = 0; i < _vOutgoingLargerLayer.size(); i++)
                {
                    NNLayer* pOutputLayer       = _vOutgoingLargerLayer[i];
                    NNWeight* pWeight           = _vOutgoingLargerWeight[i];     
                    NNWeight* pSrcWeight        = pWeight->_bShared ? pWeight->_pSharedWeight : pWeight;
                    NNFloat* pA                 = pSrcWeight->_pbWeight->_pDevData;
                    NNFloat* pB                 = getGpu()._pNetwork->GetP2PSendBuffer();
                    NNFloat* pC                 = pOutputLayer->_pbUnit->_pDevData;
                    
                    int m                       = pOutputLayer->_localStride;
                    int n                       = batch;
                    int k                       = _stride;
                    int lda                     = pOutputLayer->_localStride;
                    int ldb                     = _stride;
                    int ldc                     = pOutputLayer->_localStride;
                    const NNFloat sgemm_alpha   = 1.0;
                    const NNFloat sgemm_beta    = (pOutputLayer->_unitUpdateCount == 0) ? (NNFloat)0.0 : (NNFloat)1.0;
            
                    cublasStatus_t cstatus      = cublasSgemm(getGpu()._cuBLASHandle, 
                                                CUBLAS_OP_N,
                                                CUBLAS_OP_N,
                                                m,
                                                n,
                                                k,
                                                &sgemm_alpha,
                                                pA,
                                                lda,
                                                pB,
                                                ldb,
                                                &sgemm_beta,
                                                pC,
                                                ldc);

                    if (cstatus != CUBLAS_STATUS_SUCCESS)
                    {
                        if (getGpu()._id == 0)
                            printf("NNLayer::ForwardPropagate: SGEMM failure, aborting.\n");
                        getGpu().Shutdown();
                        exit(-1);
                    }
                        
                    // Increment unit update count
                   // printf("FP %s OL UC %d\n", _name.c_str(), pOutputLayer->_unitUpdateCount);
                    pOutputLayer->_unitUpdateCount++;
                }
            }
        }
    }
    
#if 0
    // REMOVE
    _pbUnit->Download(_vUnit.data());
    MPI_Barrier(MPI_COMM_WORLD);
    if (getGpu()._id == 0)
        cout << _name << " ";
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < getGpu()._numprocs; i++)
    {
        if (i == getGpu()._id)
        {
            for (auto f : _vUnit)
                printf("%8.4f ", f);
            printf("\n");
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    cout << endl;
    exit(-1);
#endif    
}


void NNLayer::ForwardPropagateConvolutional(uint32_t position, uint32_t batch, bool bTraining)
{ 
    if (_kind != NNLayer::Kind::Input)
    {
        // Single GPU is the simplest case
        if (getGpu()._numprocs == 1)
        {
            NNFloat alpha                   = (NNFloat)1.0;
            NNFloat beta                    = (NNFloat)0.0;            
            for (uint32_t i = 0; i < _vIncomingLayer.size(); i++)
            {
                NNLayer* pLayer             = _vIncomingLayer[i];
                NNWeight* pWeight           = _vIncomingWeight[i]->_bShared ? 
                                              _vIncomingWeight[i]->_pSharedWeight : 
                                              _vIncomingWeight[i];

                cudnnStatus_t cudnnStatus   = cudnnConvolutionForward(getGpu()._cuDNNHandle,
                                                                      &alpha,
                                                                      pLayer->getTensorDescriptor(batch),
                                                                      pLayer->_pbUnit->_pDevData,
                                                                      pWeight->_convFilterDesc,
                                                                      pWeight->_pbWeight->_pDevData,
                                                                      pWeight->_convDesc,
                                                                      pWeight->_convFWAlgo,
                                                                      getGpu()._pNetwork->_pbCUDNNWorkspace->_pDevData,
                                                                      getGpu()._pNetwork->_CUDNNWorkspaceSize,
                                                                      &beta,
                                                                      getTensorDescriptor(batch),
                                                                      _pbUnit->_pDevData);
                CUDNNERROR(cudnnStatus, "NNLayer::ForwardPropagateConvolutional: cudnnConvolutionForward Failed");
                                                                                 
                // All weights have their own biases, so don't used those from shared weight if so
                cudnnStatus                 = cudnnAddTensor(getGpu()._cuDNNHandle,
                                                             &alpha,
                                                             _vIncomingWeight[i]->_convBiasTensor,
                                                             _vIncomingWeight[i]->_pbBias->_pDevData,
                                                             &alpha,
                                                             getTensorDescriptor(batch),
                                                             _pbUnit->_pDevData);
                CUDNNERROR(cudnnStatus, "NNLayer::ForwardPropagateConvolutional: cudnnAddTensor Failed");
                beta                        = 1.0f;            
            }
            
            // Copy data from incoming skip layers
            for (auto l : _vIncomingSkip)
            {
                kAddBuffers(_pbUnit->_pDevData, l->_pbUnit->_pDevData, batch * _stride);
            }
           
            // Calculate activation
            CalculateActivation(batch);
            
            // Apply dropout if active
            if (bTraining && (_pDropout > (NNFloat)0.0))
                CalculateDropout(batch);             
        }       
    }
}

void NNLayer::ForwardPropagatePooling(uint32_t position, uint32_t batch, bool bTraining)
{ 
    if (_kind != NNLayer::Kind::Input)
    {
        NNFloat alpha                           = (NNFloat)1.0;
        NNFloat beta                            = (NNFloat)0.0;
        for (int i = 0; i < _vIncomingLayer.size(); i++)
        {
            NNLayer* pLayer                     = _vIncomingLayer[i];
            cudnnStatus_t cudnnStatus;
            switch (_poolingFunction)
            {
                case PoolingFunction::Max:
                case PoolingFunction::Average:             
                    cudnnStatus                 = cudnnPoolingForward(getGpu()._cuDNNHandle,
                                                                  _poolingDescriptor,
                                                                  &alpha,
                                                                  pLayer->getTensorDescriptor(batch),
                                                                  pLayer->_pbUnit->_pDevData,
                                                                  &beta,
                                                                  getTensorDescriptor(batch),
                                                                  _pbUnit->_pDevData);
                    CUDNNERROR(cudnnStatus, "NNLayer::ForwardPropagatePooling: cudnnPoolingForward Failed");
                    break;

                case PoolingFunction::LRN:
                    cudnnStatus                 = cudnnLRNCrossChannelForward(getGpu()._cuDNNHandle,
                                                                          _LRNDescriptor,
                                                                          CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                                                          &alpha,
                                                                          pLayer->getTensorDescriptor(batch),
                                                                          pLayer->_pbUnit->_pDevData,
                                                                          &beta,
                                                                          getTensorDescriptor(batch),
                                                                          _pbUnit->_pDevData);
                    CUDNNERROR(cudnnStatus, "NNLayer::ForwardPropagatePooling: cudnnLRNCrossChannelForward Failed");                                                                              
                    break;
                    
                    
                case PoolingFunction::Maxout:
                    // Will special case 4 or fewer sources into one pass, this will be for remainder of >4 sources
                    if (beta != (NNFloat)0.0)
                    {
                        kCalculateMaxout(pLayer->_pbUnit->_pDevData, batch * _localStride, _pbUnit->_pDevData);
                    }
                    else
                    {
                        cudaError_t status     = cudaMemcpy(_pbUnit->_pDevData, pLayer->_pbUnit->_pDevData, batch * _localStride * sizeof(NNFloat), cudaMemcpyDefault);
                        RTERROR(status, "NNLayer::ForwardPropagate: Error calling cudaMemcpy for maxout pooling.");
                    }
                    break;
                    
            }
            beta                            = (NNFloat)1.0;
        }

        // Copy data from incoming skip layers
        for (auto l : _vIncomingSkip)
        {
            kAddBuffers(_pbUnit->_pDevData, l->_pbUnit->_pDevData, batch * _stride);
        }        
    }
}

void NNLayer::CalculateActivation(uint32_t batch)
{
    uint64_t size                   = (uint64_t)batch * (uint64_t)_localStride;
    switch (_activation)
    {
        case Sigmoid:
            kCalculateSigmoidActivation(_pbUnit->_pDevData, size);
            break;

        case Tanh:
            kCalculateTanhActivation(_pbUnit->_pDevData, size);
            break;

        case RectifiedLinear:
            kCalculateReluActivation(_pbUnit->_pDevData, size);
            break;
        
        case SoftMax:
            kCalculateSoftMaxActivation(_pbUnit->_pDevData, batch, _localStride);
            break;

        // Stub for no activation needed
        case Linear:
            break;
    }
}

void NNLayer::CalculateDropout(uint32_t batch)
{
    kCalculateDropout(_pbUnit->_pDevData, _pbDropout->_pDevData, batch, _localStride, _pDropout);
}

NNFloat NNLayer::CalculateError(uint32_t position, uint32_t batch, ErrorFunction ef)
{
    if (_kind != Output)
    {
        if (getGpu()._id == 0)
            printf("NNLayer::CalculateError: Attempt to calculate error on non-output layer %s.\n", _name.c_str());
        getGpu().Shutdown();
        exit(-1);
    }

    switch (ef)
    {
        case L1:
            return _pDataSet->CalculateL1Error(position, batch, _localStride, _pbUnit->_pDevData);

        case L2:
            return _pDataSet->CalculateL2Error(position, batch, _localStride, _pbUnit->_pDevData);  

        case CrossEntropy:
            if (_activation == SoftMax)
                return _pDataSet->CalculateMultinomialCrossEntropyError(position, batch, _localStride, _pbUnit->_pDevData);
            else
                return _pDataSet->CalculateCrossEntropyError(position, batch, _localStride, _pbUnit->_pDevData);

        case ScaledMarginalCrossEntropy:
            if (_activation == SoftMax)
                return _pDataSet->CalculateMultinomialScaledMarginalCrossEntropyError(position, batch, _localStride, _pbUnit->_pDevData);
            else        
                return _pDataSet->CalculateScaledMarginalCrossEntropyError(position, batch, _localStride, _pbUnit->_pDevData);

        case DataScaledMarginalCrossEntropy:
            if (_activation == SoftMax)
            {
                cout << "unsupported combination of activation with cost function" << endl;
                getGpu().Shutdown();
                exit(-1);
            }
            else
            {
                return _pDataSet->CalculateDataScaledMarginalCrossEntropyError(position, batch, _localStride, _pbUnit->_pDevData);
            }
    }
    
    return (NNFloat)0.0;
}

void NNLayer::CalculateOutputDelta(uint32_t position, uint32_t batch, ErrorFunction ef)
{
    if (_kind != Output)
    {
        if (getGpu()._id == 0)
            printf("NNLayer::CalculateOutputDelta: Attempt to calculate output delta on non-output layer %s.\n", _name.c_str());
        getGpu().Shutdown();
        exit(-1);
    }

    switch (ef)
    {
        case L1:
            _pDataSet->CalculateL1OutputDelta(_activation, position, batch, _localStride, _pbUnit->_pDevData, _pbDelta->_pDevData);
            break;

        case CrossEntropy:
            _pDataSet->CalculateCrossEntropyOutputDelta(_activation, position, batch, _localStride, _pbUnit->_pDevData, _pbDelta->_pDevData);
            break;

        case ScaledMarginalCrossEntropy:
            _pDataSet->CalculateScaledMarginalCrossEntropyOutputDelta(_activation, position, batch, _localStride, _pbUnit->_pDevData, _pbDelta->_pDevData);
            break;

        case L2:
            _pDataSet->CalculateOutputDelta(_activation, position, batch, _localStride, _pbUnit->_pDevData, _pbDelta->_pDevData);
            break;

        case DataScaledMarginalCrossEntropy:
            _pDataSet->CalculateDataScaledMarginalCrossEntropyOutputDelta(_activation, position, batch, _localStride, _pbUnit->_pDevData, _pbDelta->_pDevData);
            break;

        default:
            cout << "Unsupported cost function" << endl;
            exit(2);
    }
    
    
    // Normalize deltas if desired
    if (_deltaNorm > (NNFloat)0.0)
    {
        if (getGpu()._numprocs == 1)
            kNormalizeDeltas(_deltaNorm, batch, _localStride, _pbDelta->_pDevData);
        else
        {
            NNFloat* pMagnitude                 = getGpu()._pNetwork->GetScratchBuffer(batch);
            kCalculateDeltaMagnitudes(batch, _localStride, _pbDelta->_pDevData, pMagnitude);
            getGpu()._pNetwork->P2P_Allreduce(pMagnitude, batch);
            kNormalizeDeltaMagnitudes(_deltaNorm, batch, _localStride, _pbDelta->_pDevData, pMagnitude);            
        }
    }
}


void NNLayer::BackPropagate(uint32_t position, uint32_t batch, NNFloat alpha)
{
    
    // Will switch to class-based decision shortly once working
    switch (_type)
    {
        case FullyConnected:
            BackPropagateFullyConnected(position, batch, alpha);
            break;
            
        case Convolutional:
            BackPropagateConvolutional(position, batch, alpha);
            break;
            
        case Pooling:
            BackPropagatePooling(position, batch, alpha);
            break;                        
        
    }
}

void NNLayer::BackPropagateConvolutional(uint32_t position, uint32_t batch, NNFloat alpha)
{
    // Special case single GPU
    if (getGpu()._numprocs == 1)
    {
        // Calculate average sparseness and add sparse penalty on hidden layers if active (100% local calculation)
        if (_kind == Hidden)
        {
            if (_bSparse && getGpu()._data._bSparsenessPenalty)
            {
                NNFloat p       = (_sparsenessPenalty_p > (NNFloat)0.0)   ? _sparsenessPenalty_p     : getGpu()._pNetwork->_sparsenessPenalty_p;
                NNFloat beta    = (_sparsenessPenalty_beta > (NNFloat)0.0) ? _sparsenessPenalty_beta : getGpu()._pNetwork->_sparsenessPenalty_beta;
                kCalculateSparsenessPenalty(batch, _localStride, _pbUnit->_pDevData, _pbDelta->_pDevData, p, beta);
            }   

            // Account for dropout when calculating deltas (100% local calculation)
            NNFloat scale                           = (NNFloat)1.0 / ((NNFloat)1.0 - _pDropout);
            kCalculateHadamardProduct(_activation, batch * _localStride, scale, _pbUnit->_pDevData, _pbDelta->_pDevData);
            
            // Normalize deltas if desired
            if (_deltaNorm > (NNFloat)0.0)
            {            
                kNormalizeDeltas(_deltaNorm, batch, _localStride, _pbDelta->_pDevData);
            }
        }


        // Cycle through incoming layers and process gradient and delta contributions
        for (uint32_t i = 0; i < _vIncomingLayer.size(); i++)
        {
            // Calculate gradients on weights between this and the current incoming layer
            NNLayer* pInputLayer                = _vIncomingLayer[i];

            NNWeight* pWeight                   = _vIncomingWeight[i];     
            NNWeight* pSrcWeight                = pWeight->_bShared ? pWeight->_pSharedWeight : pWeight;
            NNFloat gradient_alpha              = -(NNFloat)1.0 / (pSrcWeight->_sharingCount * (NNFloat)batch);            

            // Skip update if weights are locked
            cudnnStatus_t cudnnStatus;
            if (!pWeight->_bLocked)
            {
                NNFloat beta                    = (pSrcWeight->_updateCount == 0) ? (NNFloat)0.0 : (NNFloat)1.0;
                cudnnStatus                     = cudnnConvolutionBackwardFilter(getGpu()._cuDNNHandle,
                                                                                 &gradient_alpha,
                                                                                 pInputLayer->getTensorDescriptor(batch),
                                                                                 pInputLayer->_pbUnit->_pDevData,
                                                                                 getTensorDescriptor(batch),
                                                                                 _pbDelta->_pDevData,
                                                                                 pSrcWeight->_convDesc,
                                                                                 pSrcWeight->_convBWWeightAlgo,
                                                                                 getGpu()._pNetwork->_pbCUDNNWorkspace->_pDevData,
                                                                                 getGpu()._pNetwork->_CUDNNWorkspaceSize,
                                                                                 &beta,
                                                                                 pSrcWeight->_convFilterDesc,
                                                                                 pSrcWeight->_pbWeightGradient->_pDevData);
                CUDNNERROR(cudnnStatus, "NNLayer::BackPropagateConvolutional: cudnnConvolutionBackwardFilter Failed"); 
                
                // Biases are unshared, so overwrite any old gradient data
                beta                            = (NNFloat)0.0;
                cudnnStatus                     = cudnnConvolutionBackwardBias(getGpu()._cuDNNHandle,
                                                                           &gradient_alpha,
                                                                           getTensorDescriptor(batch),
                                                                           _pbDelta->_pDevData,
                                                                           &beta,
                                                                           pWeight->_convBiasTensor,
                                                                           pWeight->_pbBiasGradient->_pDevData);                
                

                // Increment update count
                pSrcWeight->_updateCount++;
            }
     
            // Calculate delta contributions for incoming non-input layers
            if (pInputLayer->_kind != Input)
            {
                NNFloat delta_alpha             = (NNFloat)1.0;                
                NNFloat beta                    = (pInputLayer->_deltaUpdateCount == 0) ? (NNFloat)0.0 : (NNFloat)1.0;
                cudnnStatus                     = cudnnConvolutionBackwardData(getGpu()._cuDNNHandle,
                                                                               &delta_alpha,
                                                                               pSrcWeight->_convFilterDesc,
                                                                               pSrcWeight->_pbWeight->_pDevData,
                                                                               getTensorDescriptor(batch),
                                                                               _pbDelta->_pDevData,
                                                                               pSrcWeight->_convDesc, 
                                                                               pSrcWeight->_convBWDeltaAlgo,
                                                                               getGpu()._pNetwork->_pbCUDNNWorkspace->_pDevData,
                                                                               getGpu()._pNetwork->_CUDNNWorkspaceSize,
                                                                               &beta,
                                                                               pInputLayer->getTensorDescriptor(batch),
                                                                               pInputLayer->_pbDelta->_pDevData);
                CUDNNERROR(cudnnStatus, "NNLayer::BackPropagateConvolutional: cudnnConvolutionBackwardData Failed");

                // Increment update count
                pInputLayer->_deltaUpdateCount++; 
            }
        }    
        
        // Copy deltas to incoming skip layers
        for (auto l : _vIncomingSkip)
        {
            if (l->_deltaUpdateCount > 0)
            {
                kAddBuffers(l->_pbDelta->_pDevData, _pbDelta->_pDevData, batch * _localStride);
            }
            else
            {
                cudaMemcpy(l->_pbDelta->_pDevData, _pbDelta->_pDevData, batch * _localStride * sizeof(NNFloat), cudaMemcpyDefault);
            }
         
            l->_deltaUpdateCount++;
        }
    }
}

void NNLayer::BackPropagatePooling(uint32_t position, uint32_t batch, NNFloat alpha)
{
    // Special case single GPU
    if (getGpu()._numprocs == 1)
    {
        // Cycle through incoming layers and process gradient and delta contributions
        NNFloat pooling_alpha                   = (NNFloat)1.0;
        for (uint32_t i = 0; i < _vIncomingLayer.size(); i++)
        {
            // Calculate gradients on weights between this and the current incoming layer
            NNLayer* pInputLayer                = _vIncomingLayer[i];

            // Calculate delta contributions for incoming non-input layers
            if (pInputLayer->_kind != Input)
            {
                cudnnStatus_t cudnnStatus;
                NNFloat beta                    = (pInputLayer->_deltaUpdateCount == 0) ? (NNFloat)0.0 : (NNFloat)1.0;
                switch (_poolingFunction)
                {
                    case Max:
                    case Average:
                        cudnnStatus             = cudnnPoolingBackward(getGpu()._cuDNNHandle,
                                                                       _poolingDescriptor,
                                                                       &pooling_alpha,
                                                                       getTensorDescriptor(batch),
                                                                       _pbUnit->_pDevData,
                                                                       getTensorDescriptor(batch),
                                                                       _pbDelta->_pDevData,
                                                                       pInputLayer->getTensorDescriptor(batch),
                                                                       pInputLayer->_pbUnit->_pDevData,
                                                                       &beta,
                                                                       pInputLayer->getTensorDescriptor(batch),
                                                                       pInputLayer->_pbDelta->_pDevData);                                                                         
                        CUDNNERROR(cudnnStatus, "NNLayer::BackPropagatePooling: cudnnPoolingBackward Failed");
                        break;

                    case LRN:
                        cudnnStatus             = cudnnLRNCrossChannelBackward(getGpu()._cuDNNHandle,
                                                                                _LRNDescriptor,
                                                                                CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                                                                &pooling_alpha,
                                                                                getTensorDescriptor(batch),
                                                                                _pbUnit->_pDevData,
                                                                                getTensorDescriptor(batch),
                                                                                _pbDelta->_pDevData,
                                                                                pInputLayer->getTensorDescriptor(batch),
                                                                                pInputLayer->_pbUnit->_pDevData,
                                                                                &beta,
                                                                                pInputLayer->getTensorDescriptor(batch),
                                                                                pInputLayer->_pbDelta->_pDevData);                      
                        CUDNNERROR(cudnnStatus, "NNLayer::BackPropagatePooling: cudnnLRNCrossChannelBackward Failed");
                        break;
                        
                    case Maxout:
                        kCalculateMaxoutDelta(_pbUnit->_pDevData, _pbDelta->_pDevData, batch * _localStride, beta, pInputLayer->_pbUnit->_pDevData, pInputLayer->_pbDelta->_pDevData);
                        break;

                }
                
                // Increment update count
                pInputLayer->_deltaUpdateCount++; 
            }
        }    
        
        // Copy deltas to incoming layers
        for (auto l : _vIncomingSkip)
        {
            if (l->_deltaUpdateCount > 0)
            {
                kAddBuffers(l->_pbDelta->_pDevData, _pbDelta->_pDevData, batch * _localStride);
            }
            else
            {
                cudaMemcpy(l->_pbDelta->_pDevData, _pbDelta->_pDevData, batch * _localStride * sizeof(NNFloat), cudaMemcpyDefault);
            }
         
            l->_deltaUpdateCount++;
        }
    }
}

// Calculates all contributions to Delta(t-1) or (Delta(t) * W(t-1->t)^T) which is the product of a [batch][stride] and [stride][outgoing stride] matrix
// And for efficiency purposes, the local contribution to dW(t-1->t), which is x(t-1)^T * Delta(t)
void NNLayer::BackPropagateFullyConnected(uint32_t position, uint32_t batch, NNFloat alpha)
{    
    // Special case single GPU
    if (getGpu()._numprocs == 1)
    {
        // Calculate average sparseness and add sparse penalty on hidden layers if active (100% local calculation)
        if (_kind == Hidden)
        {
            if (_bSparse && getGpu()._data._bSparsenessPenalty)
            {
                NNFloat p       = (_sparsenessPenalty_p > (NNFloat)0.0)   ? _sparsenessPenalty_p     : getGpu()._pNetwork->_sparsenessPenalty_p;
                NNFloat beta    = (_sparsenessPenalty_beta > (NNFloat)0.0) ? _sparsenessPenalty_beta : getGpu()._pNetwork->_sparsenessPenalty_beta;
                kCalculateSparsenessPenalty(batch, _localStride, _pbUnit->_pDevData, _pbDelta->_pDevData, p, beta);
            }   

            // Account for dropout when calculating deltas (100% local calculation)
            NNFloat scale                           = (NNFloat)1.0 / ((NNFloat)1.0 - _pDropout);
            kCalculateHadamardProduct(_activation, batch * _localStride, scale, _pbUnit->_pDevData, _pbDelta->_pDevData);
            
            // Normalize deltas if desired (Norms must be reduced across all GPUs)
            if (_deltaNorm > (NNFloat)0.0)
            {            
                kNormalizeDeltas(_deltaNorm, batch, _localStride, _pbDelta->_pDevData);
            }
        }

#if 0
        if (_kind == Hidden)
        {
            string fname = "delta_" + _name;
            Dump(fname, _pbDelta->_pDevData);
        }
#endif 
        
        // Cycle through incoming layers and process gradient and delta contributions
        for (uint32_t i = 0; i < _vIncomingLayer.size(); i++)
        {
            // Calculate gradients on weights between this and the current incoming layer
            NNLayer* pInputLayer                = _vIncomingLayer[i];
            cublasStatus_t cstatus;
            NNWeight* pWeight                   = _vIncomingWeight[i];     
            NNWeight* pSrcWeight                = pWeight->_bShared ? pWeight->_pSharedWeight : pWeight;

            // Skip update if weights are locked
            if (!pWeight->_bLocked)
            {
                // Calculate weight gradients
                NNFloat* pDelta                 = _pbDelta->_pDevData;
                NNFloat* pUnit                  = pInputLayer->_pbUnit ? pInputLayer->_pbUnit->_pDevData : NULL;
                NNFloat* pA                     = pWeight->_bTransposed ? pDelta                    : pUnit;
                NNFloat* pB                     = pWeight->_bTransposed ? pUnit                     : pDelta;
                int m                           = pWeight->_bTransposed ? pInputLayer->_localStride : _localStride;
                int n                           = pWeight->_bTransposed ? _localStride              : pInputLayer->_localStride;
                int k                           = batch;
                int lda                         = pWeight->_bTransposed ? pInputLayer->_localStride : _localStride;
                int ldb                         = pWeight->_bTransposed ? _localStride              : pInputLayer->_localStride;
                int ldc                         = pWeight->_bTransposed ? pInputLayer->_localStride : _localStride;

                // Update weights
                NNFloat sgemm_alpha             = -(NNFloat)1.0 / (pSrcWeight->_sharingCount * (NNFloat)batch);
                NNFloat sgemm_beta              = (pSrcWeight->_updateCount == 0) ? (NNFloat)0.0 : (NNFloat)1.0;
                NNFloat* pC                     = pSrcWeight->_pbWeightGradient->_pDevData;
                
                if ((pInputLayer->_kind == NNLayer::Kind::Input) && pInputLayer->_bFastSparse && !pWeight->_bTransposed)
                {
                    pInputLayer->_pDataSet->CalculateSparseTransposedWeightGradient(sgemm_alpha, sgemm_beta, n, m, pB, pC);
                }
                else
                {
                    cstatus                 = cublasSgemm(getGpu()._cuBLASHandle, 
                                              CUBLAS_OP_N,
                                              CUBLAS_OP_T,
                                              m,
                                              n,
                                              k,
                                              &sgemm_alpha,
                                              pB,
                                              lda,
                                              pA,
                                              ldb,
                                              &sgemm_beta,
                                              pC,
                                              ldc);

                    if (cstatus != CUBLAS_STATUS_SUCCESS)
                    {
                        if (getGpu()._id == 0)
                            printf("NNLayer::BackPropagate: SGEMM failure, aborting.\n");
                        getGpu().Shutdown();
                        exit(-1);
                    }
                }
                
                // Increment update count
                pSrcWeight->_updateCount++;
            }
     
            // Calculate delta contributions for incoming non-input layers
            if (pInputLayer->_kind != Input)
            {
                NNFloat sgemm_alpha         = (NNFloat)1.0;
                NNFloat sgemm_beta          = (pInputLayer->_deltaUpdateCount == 0) ? (NNFloat)0.0 : (NNFloat)1.0;
                int m                       = pInputLayer->_localStride;
                int n                       = batch;  
                
                
                NNFloat* pA                 = _pbDelta->_pDevData;
                NNFloat* pB                 = pWeight->_bShared ? 
                                              pSrcWeight->_pbWeight->_pDevData :
                                              pWeight->_pbWeight->_pDevData;

                NNFloat* pC                 = pInputLayer->_pbDelta->_pDevData;
                int k                       = _localStride;
                int lda                     = pWeight->_bTransposed ? pInputLayer->_localStride : k;
                int ldb                     = k;
                int ldc                     = pInputLayer->_localStride;
                
                //printf("Delta between %s and %s %16.8f %16.8f\n", pInputLayer->_name.c_str(), _name.c_str(), sgemm_alpha, sgemm_beta);             
                
                //printf("%s: %d %d %d | %d %d %d\n", _name.c_str(), m, n, k, lda, ldb, ldc);
                cstatus                     = cublasSgemm(getGpu()._cuBLASHandle, 
                                            pWeight->_bTransposed ? CUBLAS_OP_N : CUBLAS_OP_T,
                                            CUBLAS_OP_N,
                                            m,
                                            n,
                                            k,
                                            &sgemm_alpha,
                                            pB,
                                            lda,
                                            pA,
                                            ldb,
                                            &sgemm_beta,
                                            pC,
                                            ldc);   

                if (cstatus != CUBLAS_STATUS_SUCCESS)
                {
                    if (getGpu()._id == 0)
                        printf("NNLayer::BackPropagate: SGEMM failure, aborting.\n");
                    getGpu().Shutdown();
                    exit(-1);
                }
                
                pInputLayer->_deltaUpdateCount++; 
            }
        }    
        
        // Copy deltas to incoming layers
        for (auto l : _vIncomingSkip)
        {
            if (l->_deltaUpdateCount > 0)
            {
                kAddBuffers(l->_pbDelta->_pDevData, _pbDelta->_pDevData, batch * _localStride);
            }
            else
            {
                cudaMemcpy(l->_pbDelta->_pDevData, _pbDelta->_pDevData, batch * _localStride * sizeof(NNFloat), cudaMemcpyDefault);
            }
         
            l->_deltaUpdateCount++;
        }
    }
    else    // Multi-GPU case
    {
        // Process outgoing larger layers by gathering additional contributions to delta(L) and scattering X(L) to contribute
        // to dW(L->L+1)
        if (_vOutgoingLargerLayer.size() > 0)
        {
            // Gather X(L) on all GPUs to calculate contributions to all dW(L->L+1)
            Gather(batch, _stride, _pbUnit->_pDevData, _localStride);

            // Calculate contribution to weight gradients of each outgoing layer
            for (int i = 0; i < _vOutgoingLargerLayer.size(); i++)
            {
                NNLayer* pOutputLayer           = _vOutgoingLargerLayer[i];
                NNWeight* pWeight               = _vOutgoingLargerWeight[i];     
                NNWeight* pSrcWeight            = pWeight->_bShared ? pWeight->_pSharedWeight : pWeight;
                
                // Calculate weight gradient contribution
                NNFloat* pA                     = pOutputLayer->_pbDelta->_pDevData;
                NNFloat* pB                     = getGpu()._pNetwork->GetP2PSendBuffer();
                NNFloat* pC                     = pSrcWeight->_pbWeightGradient->_pDevData;
                int m                           = pOutputLayer->_localStride;
                int n                           = _stride;
                int k                           = batch;
                int lda                         = pOutputLayer->_localStride;
                int ldb                         = _stride;
                int ldc                         = pOutputLayer->_localStride;

                // Update weight gradients
                NNFloat sgemm_alpha             = -(NNFloat)1.0 / (pSrcWeight->_sharingCount * (NNFloat)batch);
                NNFloat sgemm_beta              = (pSrcWeight->_updateCount == 0) ? (NNFloat)0.0 : (NNFloat)1.0;               
                
                cublasStatus_t cstatus          = cublasSgemm(getGpu()._cuBLASHandle, 
                                                CUBLAS_OP_N,
                                                CUBLAS_OP_T,
                                                m,
                                                n,
                                                k,
                                                &sgemm_alpha,
                                                pA,
                                                lda,
                                                pB,
                                                ldb,
                                                &sgemm_beta,
                                                pC,
                                                ldc);

                if (cstatus != CUBLAS_STATUS_SUCCESS)
                {
                    if (getGpu()._id == 0)
                        printf("NNLayer::BackPropagate: SGEMM failure, aborting.\n");
                    getGpu().Shutdown();
                    exit(-1);
                }
                //printf("BP %s OL UW %d\n", _name.c_str(), pSrcWeight->_updateCount);

                // Increment update count
                pSrcWeight->_updateCount++;
            }  

            // Calculate contributions to Delta(L)            
            NNFloat sgemm_beta                  = (NNFloat)0.0;              
            for (uint32_t i = 0; i < _vOutgoingLargerLayer.size(); i++)
            {
                NNLayer* pOutputLayer           = _vOutgoingLargerLayer[i];
                const NNFloat sgemm_alpha       = (NNFloat)1.0;
                NNFloat* pA                     = _vOutgoingLargerWeight[i]->_bShared ? 
                                                  _vOutgoingLargerWeight[i]->_pSharedWeight->_pbWeight->_pDevData : 
                                                  _vOutgoingLargerWeight[i]->_pbWeight->_pDevData;
                NNFloat* pB                     = pOutputLayer->_pbDelta->_pDevData;
                NNFloat* pC                     = getGpu()._pNetwork->GetP2PSendBuffer();
                int m                           = _stride;
                int n                           = batch;
                int k                           = pOutputLayer->_localStride;
                int lda                         = pOutputLayer->_localStride;
                int ldb                         = pOutputLayer->_localStride;
                int ldc                         = _stride;

                cublasStatus_t cstatus          =
                                                cublasSgemm(getGpu()._cuBLASHandle, 
                                                CUBLAS_OP_T,
                                                CUBLAS_OP_N,
                                                m,
                                                n,
                                                k,
                                                &sgemm_alpha,
                                                pA,
                                                lda,
                                                pB,
                                                ldb,
                                                &sgemm_beta,
                                                pC,
                                                ldc);  

                // Make sure matrix multiply succeeded
                if (cstatus != CUBLAS_STATUS_SUCCESS)
                {
                    if (getGpu()._id == 0)
                        printf("NNLayer::BackPropagate: SGEMM failure, aborting, status %d.\n", cstatus);
                    getGpu().Shutdown();
                    exit(-1);
                }
#if 0
                NNFloat* pD = pOutputLayer->_vDelta.data();
                NNFloat* pW = _vOutgoingWeight[i]->_vWeight.data();
                
                pOutputLayer->_pbDelta->Download(pD);
                _vOutgoingLargerWeight[i]->_pbWeight->Download(pW);
                pW += pOutputLayer->_localStride;
                NNFloat sum = 0.0f;
                for (int j = 0; j < pOutputLayer->_localStride; j++)
                {
                    sum += (*pD) * (*pW);
                    pD++;
                    pW++;
                }
                MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                if (getGpu()._id == 0)
                    printf("ZAG %16.12f\n", sum);
                MPI_Barrier(MPI_COMM_WORLD);  
#endif
                        
                // Add subsequent layers
                sgemm_beta                      = (NNFloat)1.0;
            }

            //printf("BP %s OL UD %d\n", _name.c_str(), _deltaUpdateCount);

            // Reduce Delta(L)
            Reduce(batch, _stride, _pbDelta->_pDevData, _localStride, _deltaUpdateCount);
            _deltaUpdateCount++;
        }


        
        // Calculate average sparseness and add sparse penalty on hidden layers if active (100% local calculation)
        if (_kind == Hidden)
        {
            if (_bSparse && getGpu()._data._bSparsenessPenalty)
            {
                NNFloat p       = (_sparsenessPenalty_p > (NNFloat)0.0)   ? _sparsenessPenalty_p     : getGpu()._pNetwork->_sparsenessPenalty_p;
                NNFloat beta    = (_sparsenessPenalty_beta > (NNFloat)0.0) ? _sparsenessPenalty_beta : getGpu()._pNetwork->_sparsenessPenalty_beta;
                kCalculateSparsenessPenalty(batch, _localStride, _pbUnit->_pDevData, _pbDelta->_pDevData, p, beta);                
            }   

            // Account for dropout when calculating deltas (100% local calculation)
            NNFloat scale                           = (NNFloat)1.0 / ((NNFloat)1.0 - _pDropout);
            kCalculateHadamardProduct(_activation, batch * _localStride, scale, _pbUnit->_pDevData, _pbDelta->_pDevData);
            
            // Normalize deltas if desired (Norms must be reduced across all GPUs)
            if (_deltaNorm > (NNFloat)0.0)
            {            
                NNFloat* pMagnitude             = getGpu()._pNetwork->GetScratchBuffer(batch);
                kCalculateDeltaMagnitudes(batch, _localStride, _pbDelta->_pDevData, pMagnitude);
                getGpu()._pNetwork->P2P_Allreduce(pMagnitude, batch);
                kNormalizeDeltaMagnitudes(_deltaNorm, batch, _localStride, _pbDelta->_pDevData, pMagnitude);
            }
        }

        // Copy deltas to incoming layers that skip into this layer
        for (auto l : _vIncomingSkip)
        {
            if (l->_deltaUpdateCount > 0)
            {
                kAddBuffers(l->_pbDelta->_pDevData, _pbDelta->_pDevData, batch * _localStride);
            }
            else
            {
                cudaMemcpy(l->_pbDelta->_pDevData, _pbDelta->_pDevData, batch * _localStride * sizeof(NNFloat), cudaMemcpyDefault);
            }
         
            l->_deltaUpdateCount++;
        }          

        // Gather delta(L) to contribute to delta and dW of incoming larger layers
        if (_vIncomingLargerLayer.size() > 0)
        {
            Gather(batch, _stride, _pbDelta->_pDevData, _localStride);   
                 
            for (int i = 0; i < _vIncomingLargerLayer.size(); i++)
            {
                NNLayer* pInputLayer            = _vIncomingLargerLayer[i];
                NNWeight* pWeight               = _vIncomingLargerWeight[i];     
                NNWeight* pSrcWeight            = pWeight->_bShared ? pWeight->_pSharedWeight : pWeight;
                
                // Calculate weight gradient contribution
                NNFloat* pA                     = getGpu()._pNetwork->GetP2PSendBuffer();
                NNFloat* pC                     = pSrcWeight->_pbWeightGradient->_pDevData;
                int m                           = _stride;
                int n                           = pInputLayer->_localStride;
                int k                           = batch;
                int lda                         = _stride;
                int ldb                         = pInputLayer->_localStride;
                int ldc                         = _stride;

                // Update weight gradients
                NNFloat sgemm_alpha             = -(NNFloat)1.0 / (pSrcWeight->_sharingCount * (NNFloat)batch);
                NNFloat sgemm_beta              = (pSrcWeight->_updateCount == 0) ? (NNFloat)0.0 : (NNFloat)1.0;
                
                // Use sparse kernels if possible
                if ((pInputLayer->_kind == NNLayer::Kind::Input) && pInputLayer->_bFastSparse)
                {
                    pInputLayer->_pDataSet->CalculateSparseTransposedWeightGradient(sgemm_alpha, sgemm_beta, n, m, pA, pC);
                }
                else
                { 
                    NNFloat* pB                 = pInputLayer->_pbUnit->_pDevData;          
                    cublasStatus_t cstatus      = cublasSgemm(getGpu()._cuBLASHandle, 
                                                CUBLAS_OP_N,
                                                CUBLAS_OP_T,
                                                m,
                                                n,
                                                k,
                                                &sgemm_alpha,
                                                pA,
                                                lda,
                                                pB,
                                                ldb,
                                                &sgemm_beta,
                                                pC,
                                                ldc);

                    if (cstatus != CUBLAS_STATUS_SUCCESS)
                    {
                        if (getGpu()._id == 0)
                            printf("NNLayer::BackPropagate: SGEMM failure, aborting.\n");
                        getGpu().Shutdown();
                        exit(-1);
                    }
                }

                //printf("BP %s IL UW %d\n", _name.c_str(), pSrcWeight->_updateCount);
                
                // Increment update count
                pSrcWeight->_updateCount++;
               
                // Calculate delta contribution if not input layer
                if (pInputLayer->_kind != Input)
                {
                    sgemm_alpha                 = (NNFloat)1.0;
                    sgemm_beta                  = (pInputLayer->_deltaUpdateCount == 0) ? (NNFloat)0.0 : (NNFloat)1.0;
                    pA                          = pSrcWeight->_pbWeight->_pDevData;
                    NNFloat* pB                 = getGpu()._pNetwork->GetP2PSendBuffer();
                    pC                          = pInputLayer->_pbDelta->_pDevData;
                    m                           = pInputLayer->_localStride;
                    n                           = batch;
                    k                           = _stride;                           
                    lda                         = _stride;
                    ldb                         = _stride;
                    ldc                         = pInputLayer->_localStride;
                    cublasStatus_t cstatus      = cublasSgemm(getGpu()._cuBLASHandle, 
                                                CUBLAS_OP_T,
                                                CUBLAS_OP_N,
                                                m,
                                                n,
                                                k,
                                                &sgemm_alpha,
                                                pA,
                                                lda,
                                                pB,
                                                ldb,
                                                &sgemm_beta,
                                                pC,
                                                ldc);

                    if (cstatus != CUBLAS_STATUS_SUCCESS)
                    {
                        if (getGpu()._id == 0)
                            printf("NNLayer::BackPropagate: SGEMM failure, aborting.\n");
                        getGpu().Shutdown();
                        exit(-1);
                    }

                    //printf("BP %s IL UD %d\n", _name.c_str(), pInputLayer->_deltaUpdateCount);
                    pInputLayer->_deltaUpdateCount++;
                }
            }
        }
    }
    
    
#if 0
    // Dump weight gradient
    NNWeight* pWeight                       = _vIncomingWeight[0];
    vector<NNFloat> vLocalWeightGradient(pWeight->_size);
    pWeight->_pbWeightGradient->Download(vLocalWeightGradient.data());
    for (int i = 0; i < getGpu()._numprocs; i++)
    {
        if (i == getGpu()._id)
        {
            uint32_t count = 0;
            while (count < pWeight->_size)
            {
                for (int j = 0; j < pWeight->_outputLayer._stride; j++)
                {
                    printf("%8.4f ", vLocalWeightGradient[count++]);
                }
                printf("\n");
           }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }   
    if (getGpu()._id == 0)
        cout << endl;
    //getGpu().Shutdown();
    //exit(-1);
#endif   
}

// Reduces contributions from all GPUs to local component of X(L) or Delta(L)
void NNLayer::Reduce(uint32_t batch, uint32_t stride, NNFloat* pBuffer, uint32_t localStride, uint32_t updateCount)
{

    // Only valid for multi-GPU execution
    if (getGpu()._numprocs > 1)
    {
        uint32_t stages                             = getGpu()._numprocs - 1;
        uint64_t pos                                = (getGpu()._id + 1) % getGpu()._numprocs; 
        uint32_t minX                               = (stride * pos) / getGpu()._numprocs;
        uint32_t maxX                               = (stride * (pos + 1)) / getGpu()._numprocs;
        uint32_t span                               = maxX - minX;
        NNFloat* pSendBuffer                        = getGpu()._pNetwork->GetP2PSendBuffer();

        if (getGpu()._bP2P)
        {
            NNFloat* pReceiveBuffer                 = getGpu()._pNetwork->GetP2PReceiveBuffer();
            NNFloat* pPeerBuffer                    = getGpu()._pNetwork->GetPeerBuffer();

            // Send segments around the adding local contributions from each process
            for (uint32_t i = 0; i < stages; i++)
            {
                kCopy2D(pPeerBuffer + minX, stride, pSendBuffer + minX, stride, span, batch);
                cudaDeviceSynchronize();       
                MPI_Barrier(MPI_COMM_WORLD);
        
                // Move to next position and add just arrived contribution
                pos                                 = (pos + 1) % getGpu()._numprocs;
                minX                                = (stride * pos) / getGpu()._numprocs;
                maxX                                = (stride * (pos + 1)) / getGpu()._numprocs;
                span                                = maxX - minX;
                kAddBuffers2D(pSendBuffer + minX, stride, pReceiveBuffer + minX, stride, span, batch);
            }
        }
        else
        {
            // Download to system memory and use MPI to perform reduction
            NNFloat* pCPUBuffer                     = getGpu()._pNetwork->GetP2PCPUBuffer();
            cudaError_t status                      = cudaMemcpy(pCPUBuffer, pSendBuffer, batch * stride * sizeof(NNFloat), cudaMemcpyDefault);
            RTERROR(status, "NNLayer::Reduce1: cudaMemcpy download failed " + getGpu()._id );
            MPI_Allreduce(MPI_IN_PLACE, pCPUBuffer, batch * stride, MPI_NNFLOAT, MPI_SUM, MPI_COMM_WORLD);

            // Upload back to GPU memory
            status = cudaMemcpy(pSendBuffer, pCPUBuffer, batch * stride * sizeof(NNFloat), cudaMemcpyDefault);
            RTERROR(status, "NNLayer::Reduce: cudaMemcpy upload failed" + getGpu()._id );
            minX                                    = (stride * getGpu()._id) / getGpu()._numprocs;
            maxX                                    = (stride * (getGpu()._id + 1)) / getGpu()._numprocs;
            span                                    = maxX - minX;            
        }

        // Copy data out to pBuffer
        if (updateCount > 0) 
        {
            kAddBuffers2D(pBuffer, localStride, pSendBuffer + minX, stride, span, batch);
        }
        else 
        {
            kCopy2D(pBuffer, localStride, pSendBuffer + minX, stride, span, batch);
        }

#if 0             
        MPI_Barrier(MPI_COMM_WORLD
       
        vector<NNFloat> vOut(16 * 16);
        cudaMemcpy(vOut.data(), pBuffer, batch * localStride * sizeof(NNFloat), cudaMemcpyDefault);
        for (int n = 0; n < getGpu()._numprocs; n++)
        {
            if (getGpu()._id == n)
            {
                for (int i = 0; i < batch; i++)
                {
                    printf("%2d ", i);
                    for (int j = 0; j < localStride; j++)
                    {
                        printf("%8.6f ", vOut[i * localStride + j]);
                    }
                    printf("\n");
                }
                printf("\n");
                fflush(stdout);
            }
            
            MPI_Barrier(MPI_COMM_WORLD);
        } 
        exit(-1);  
#endif
    }
}

// Copies all local components of X(L) or Delta(L) to all other GPUs
void NNLayer::Gather(uint32_t batch, uint32_t stride, NNFloat* pBuffer, uint32_t localStride)
{
    // Only valid for multi-GPU execution
    if (getGpu()._numprocs > 1)
    {
        uint32_t stages                                 = getGpu()._numprocs - 1;
        uint64_t pos                                    = getGpu()._id;
        NNFloat* pSendBuffer                            = getGpu()._pNetwork->GetP2PSendBuffer();
        uint32_t minX                                   = (stride * pos) / getGpu()._numprocs;
        uint32_t maxX                                   = (stride * (pos + 1)) / getGpu()._numprocs;
        uint32_t span                                   = maxX - minX;

        if (getGpu()._bP2P)
        {
            NNFloat* pPeerBuffer                        = getGpu()._pNetwork->GetPeerBackBuffer();

            // Copy local segment to send buffer
            kCopy2D(pSendBuffer + minX, stride, pBuffer, localStride, span, batch); 


            // Send segments around the adding local contributions from each process
            for (uint32_t i = 0; i < stages; i++)
            {                    
                kCopy2D(pPeerBuffer + minX, stride, pSendBuffer + minX, stride, span, batch);
                cudaDeviceSynchronize();  
                MPI_Barrier(MPI_COMM_WORLD);
                pos                                     = (pos + 1) % getGpu()._numprocs;
                minX                                    = (stride * pos) / getGpu()._numprocs;
                maxX                                    = (stride * (pos + 1)) / getGpu()._numprocs;
                span                                    = maxX - minX;
            }
        }
        else
        {
            NNFloat* pCPUBuffer                        = getGpu()._pNetwork->GetP2PCPUBuffer();

            // Download local segment to system memory
            cudaError_t status                         = cudaMemcpy2D(pCPUBuffer + minX, stride * sizeof(NNFloat), pBuffer, localStride * sizeof(NNFloat), localStride * sizeof(NNFloat), batch, cudaMemcpyDefault);
            RTERROR(status, "NNLayer::Gather: cudaMemcpy download failed");


            // use MPI_Bcast to scatter to all other processes
            for (uint32_t i = 0; i < getGpu()._numprocs; i++)
            {
                uint32_t minX                          = (stride * i) / getGpu()._numprocs;
                uint32_t maxX                          = (stride * (i + 1)) / getGpu()._numprocs;
                uint32_t span                          = maxX - minX;
                MPI_Datatype spanType;
                MPI_Type_vector(batch, span, stride, MPI_NNFLOAT, &spanType);
                MPI_Type_commit(&spanType);
                MPI_Bcast(pCPUBuffer + minX, 1, spanType, i, MPI_COMM_WORLD);
                MPI_Type_free(&spanType);
            }
 
            // Upload gathered buffer back to GPU memory
            status                                     = cudaMemcpy(pSendBuffer, pCPUBuffer, batch * stride * sizeof(NNFloat), cudaMemcpyDefault);
            RTERROR(status, "NNLayer::Gather: cudaMemcpy upload failed");
        }
#if 0
        if (getGpu()._id == 0)
        {
            vector<NNFloat> vOut(16 * 16);
            cudaMemcpy(vOut.data(), pSendBuffer, 16 * 16 * sizeof(NNFloat), cudaMemcpyDefault);
            for (int i = 0; i < 16; i++)
            {
                printf("%2d ", i);
                for (int j = 0; j < 16; j++)
                {
                    printf("%8.6f ", vOut[i * 16 + j]);
                }
                printf("\n");
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        exit(-1);
#endif
    }
}

// Dumps unit or delta data to file
void NNLayer::Dump(string fname, NNFloat* pBuffer)
{   
    vector<NNFloat> vData(_batch * _stride);
    if (getGpu()._numprocs == 1) 
    {
        cudaMemcpy(vData.data(), pBuffer, _batch * _stride * sizeof(NNFloat), cudaMemcpyDefault);
    } 
    else 
    {
        if (getGpu()._id == 0)
        {
            NNFloat* pData              = vData.data();       
            cudaMemcpy2D(pData, _stride * sizeof(NNFloat), pBuffer, _localStride * sizeof(NNFloat), _localStride * sizeof(NNFloat), _batch, cudaMemcpyDefault);
            pData                      += _localStride;
            for (uint32_t i = 1; i < getGpu()._numprocs; i++)
            {                        
                uint64_t size;
                MPI_Status status;                
                MPI_Recv(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &status);
                vector<NNFloat> vTemp(size);
                MPI_Recv(vTemp.data(), size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
                uint64_t lstride    = size / _batch;
                NNFloat* pSrc = vTemp.data();
                NNFloat* pDst = pData;
                for (uint32_t j = 0; j < _batch; j++)
                {
                    memcpy(pDst, pSrc, lstride * sizeof(NNFloat));
                    pSrc               += lstride;
                    pDst               += _stride;
                }                          
                pData                  += lstride;
            }
        }
        else
        {
            uint64_t size               = _batch * _localStride;
            vector<NNFloat> vLocalData(size);
            cudaMemcpy(vLocalData.data(), pBuffer, size * sizeof(NNFloat), cudaMemcpyDefault);
            MPI_Send(&size, 1, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD);
            MPI_Send(vLocalData.data(), size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);                  
        }
    }

    // Dump data to file from process 0
    if (getGpu()._id == 0)
    {
        FILE* fp                    = fopen(fname.c_str(), "w");
        NNFloat* pData              = vData.data();
        for (int i = 0; i < _batch; i++)
        {
            fprintf(fp, "%4d ", i);
            for (int j = 0; j < _stride; j++)
            {
                fprintf(fp, "%12.9f ", *pData);
                pData++;
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
    }
}


std::pair<NNLayer::Kind, string> NNLayer::_sKindPair[] =
{
    std::pair<NNLayer::Kind, string>(NNLayer::Kind::Input,      "Input"),
    std::pair<NNLayer::Kind, string>(NNLayer::Kind::Hidden,     "Hidden"),
    std::pair<NNLayer::Kind, string>(NNLayer::Kind::Output,     "Output"),
    std::pair<NNLayer::Kind, string>(NNLayer::Kind::Target,     "Target"),    
};

std::map<NNLayer::Kind, string> NNLayer::_sKindMap =
std::map<NNLayer::Kind, string>(_sKindPair, _sKindPair + sizeof(_sKindPair) / sizeof(_sKindPair[0]));


std::pair<NNLayer::Type, string> NNLayer::_sTypePair[] =
{
    std::pair<NNLayer::Type, string>(NNLayer::Type::FullyConnected, "FullyConnected"),
    std::pair<NNLayer::Type, string>(NNLayer::Type::Convolutional,  "Convolutional"),
    std::pair<NNLayer::Type, string>(NNLayer::Type::Pooling,        "Pooling"),    
};

std::map<NNLayer::Type, string> NNLayer::_sTypeMap =
std::map<NNLayer::Type, string>(_sTypePair, _sTypePair + sizeof(_sTypePair) / sizeof(_sTypePair[0]));

std::pair<NNLayer::Attributes, string> NNLayer::_sAttributesPair[] =
{
    std::pair<NNLayer::Attributes, string>(NNLayer::Attributes::None,       "None"),
    std::pair<NNLayer::Attributes, string>(NNLayer::Attributes::Sparse,     "Sparse"),
    std::pair<NNLayer::Attributes, string>(NNLayer::Attributes::Denoising,  "Denoising"),
};

std::map<NNLayer::Attributes, string> NNLayer::_sAttributesMap =
std::map<NNLayer::Attributes, string>(_sAttributesPair, _sAttributesPair + sizeof(_sAttributesPair) / sizeof(_sAttributesPair[0]));

std::pair<NNLayer::Parallelization, string> NNLayer::_sParallelizationPair[] =
{
    
    std::pair<NNLayer::Parallelization, string>(NNLayer::Parallelization::Data,     "Data"),
    std::pair<NNLayer::Parallelization, string>(NNLayer::Parallelization::Model,    "Model"),
    std::pair<NNLayer::Parallelization, string>(NNLayer::Parallelization::Serial,   "Serial"),
};

std::map<NNLayer::Parallelization, string> NNLayer::_sParallelizationMap =
std::map<NNLayer::Parallelization, string>(_sParallelizationPair, _sParallelizationPair + sizeof(_sParallelizationPair) / sizeof(_sParallelizationPair[0]));


ostream& operator<< (ostream& out, NNLayer::Kind& k)
{
    out << NNLayer::_sKindMap[k];
    return out;
}
ostream& operator<< (ostream& out, NNLayer::Type& t)
{
    out << NNLayer::_sTypeMap[t];
    return out;
}

ostream& operator<< (ostream& out, NNLayer::Parallelization& p)
{
    out << NNLayer::_sParallelizationMap[p];
    return out;
}

ostream& operator<< (ostream& out, NNLayer::Attributes& a)
{
    out << NNLayer::_sAttributesMap[a];
    return out;
}


bool NNLayer::WriteNetCDF(NcFile& nc, uint32_t index)
{
    bool bResult                        = true;
    if (getGpu()._id == 0)
    {
        string lstring                  = "layer" + std::to_string(index) + "_";
        nc.putAtt(lstring + "name", _name);
        nc.putAtt(lstring + "kind", ncUint, _kind);
        nc.putAtt(lstring + "type", ncUint, _type);
        nc.putAtt(lstring + "poolingfunction", ncUint, _poolingFunction);
        nc.putAtt(lstring + "dataSet", _dataSet);
        nc.putAtt(lstring + "Nx", ncUint, _Nx);
        nc.putAtt(lstring + "Ny", ncUint, _Ny);
        nc.putAtt(lstring + "Nz", ncUint, _Nz);
        nc.putAtt(lstring + "Nw", ncUint, _Nw);
        nc.putAtt(lstring + "dimensions", ncUint, _dimensions);
        nc.putAtt(lstring + "kernelX", ncUint, _kernelX);
        nc.putAtt(lstring + "kernelY", ncUint, _kernelY);
        nc.putAtt(lstring + "kernelZ", ncUint, _kernelZ);
        nc.putAtt(lstring + "kernelDimensions", ncUint, _kernelDimensions);
        nc.putAtt(lstring + "kernelStrideX", ncUint, _kernelStrideX);
        nc.putAtt(lstring + "kernelStrideY", ncUint, _kernelStrideY);
        nc.putAtt(lstring + "kernelStrideZ", ncUint, _kernelStrideZ);
        nc.putAtt(lstring + "kernelPaddingX", ncUint, _kernelPaddingX);
        nc.putAtt(lstring + "kernelPaddingY", ncUint, _kernelPaddingY);
        nc.putAtt(lstring + "kernelPaddingZ", ncUint, _kernelPaddingZ);
        nc.putAtt(lstring + "pDropout", ncFloat, _pDropout);
        nc.putAtt(lstring + "weightInit", ncUint, _weightInit);
        nc.putAtt(lstring + "weightInitScale", ncFloat, _weightInitScale);
        nc.putAtt(lstring + "biasInit", ncFloat, _biasInit);
        nc.putAtt(lstring + "weightNorm", ncFloat, _weightNorm);
        nc.putAtt(lstring + "deltaNorm", ncFloat, _deltaNorm);
        nc.putAtt(lstring + "activation", ncUint, _activation);
        nc.putAtt(lstring + "sparsenessPenalty_p", ncFloat, _sparsenessPenalty_p);
        nc.putAtt(lstring + "sparsenessPenalty_beta", ncFloat, _sparsenessPenalty_beta);

        uint32_t attributes             = 0;
        if (_bSparse)
            attributes                 |= NNLayer::Attributes::Sparse;
        if (_bDenoising)
            attributes                 |= NNLayer::Attributes::Denoising;
        nc.putAtt(lstring + "attributes", ncUint, attributes);
        nc.putAtt(lstring + "sources", ncUint, (uint32_t)_vSource.size());
        for (size_t i = 0; i < _vSource.size(); i++)
        {
            string nstring             = std::to_string(i);
            nc.putAtt(lstring + "source" + nstring, _vSource[i]);
        }
        nc.putAtt(lstring + "skips", ncUint, (uint32_t)_vSkip.size());        
        for (size_t i = 0; i < _vSkip.size(); i++)
        {
            string nstring             = std::to_string(i);
            nc.putAtt(lstring + "skip" + nstring, _vSkip[i]);
        }
    }
    else
        bResult                     = false;

    return bResult;
}
