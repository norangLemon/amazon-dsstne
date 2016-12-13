/*


   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#include "NNWeight.h"

#include <cuda.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>

#include "GpuTypes.h"
#include "NNLayer.h"
#include "NNNetwork.h"
#include "kernels.h"

using namespace netCDF;
using namespace netCDF::exceptions;

// An X vector is X[B][S] where B is that batch size and S is the stride of the layer
// X(L+1) is therefore X[OS] where OS is the outgoing stride from applying the weights
// to B discrete X vectors of size S
//
// Therefore, a weight matrix satisfies X(L+1) = X * W(L) will be W[S][OS]
NNWeight::NNWeight(NNLayer& inputLayer, NNLayer& outputLayer, bool bShared, bool bTransposed, bool bLocked, NNFloat norm) :
_inputLayer(inputLayer),
_outputLayer(outputLayer),
_width(1),
_height(1),
_length(1),
_depth(1),
_breadth(1),
_sharingCount(1),
_updateCount(0),
_bShared(bShared),
_bTransposed(bTransposed),
_bLocked(bLocked),
_norm(norm),
_pSharedWeight(NULL),
_pbWeight(NULL),
_pbBias(NULL),
_pbWeightGradient(NULL),
_pbBiasGradient(NULL),
_pbWeightVelocity(NULL),
_pbBiasVelocity(NULL),
_pbWeightGradientVelocity(NULL),
_pbBiasGradientVelocity(NULL)
{
    // Add to input and output layer lists
    inputLayer._vOutgoingLayer.push_back(&outputLayer);
    outputLayer._vIncomingLayer.push_back(&inputLayer);
    inputLayer._vOutgoingWeight.push_back(this);
    outputLayer._vIncomingWeight.push_back(this);
    
    if (_outputLayer._type == NNLayer::Type::Convolutional)
    {
        // Set transform type
        _transform                  = Convolution;
        
        // Allocate convolution data structures
        cudnnStatus_t cudnnStatus   = cudnnCreateTensorDescriptor(&_convBiasTensor);
        CUDNNERROR(cudnnStatus, "NNWeight::NNWeight: Unable to create tensor descriptor");
        cudnnStatus                 = cudnnCreateFilterDescriptor(&_convFilterDesc);
        CUDNNERROR(cudnnStatus, "NNWeight::NNWeight: Unable to create filter descriptor");
        cudnnStatus                 = cudnnCreateConvolutionDescriptor(&_convDesc);
        CUDNNERROR(cudnnStatus, "NNWeight::NNWeight: Unable to create convolution descriptor");


        // Set filter dimensions        
        vector<int> vFilterDim(5, 1);
        switch (_outputLayer._dimensions)
        {
            case 2:
                vFilterDim[0]       = _outputLayer._Ny;
                vFilterDim[1]       = _inputLayer._Ny;
                vFilterDim[2]       = _inputLayer._kernelX;
                break;
                
            case 3:
                vFilterDim[0]       = _outputLayer._Nz;
                vFilterDim[1]       = _inputLayer._Nz;
                vFilterDim[2]       = _outputLayer._kernelY;
                vFilterDim[3]       = _outputLayer._kernelX;
                break;   
                         
            case 4:
                vFilterDim[0]       = _outputLayer._Nw;
                vFilterDim[1]       = _inputLayer._Nw;
                vFilterDim[2]       = _outputLayer._kernelZ;
                vFilterDim[3]       = _outputLayer._kernelY;
                vFilterDim[4]       = _outputLayer._kernelX;
                break;  
        }
        cudnnStatus = cudnnSetFilterNdDescriptor(_convFilterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, _outputLayer._dimensions + 1, vFilterDim.data());
        CUDNNERROR(cudnnStatus, "NNWeight::NNWeight: Unable to set filter descriptor");
        
        // Copy dimensions
        _width                      = vFilterDim[0];
        _height                     = vFilterDim[1];
        _length                     = vFilterDim[2];
        _depth                      = vFilterDim[3];
        _breadth                    = vFilterDim[4];


        // Set convolution parameters
        vector<int> vConvPad(3, 0);
        vector<int> vConvStride(3, 1);
        vector<int> vConvUpscale(3, 1);
        switch (_outputLayer._dimensions)
        {
            case 2:
                vConvPad[0]         = _outputLayer._kernelPaddingX; 
                vConvStride[0]      = _outputLayer._kernelStrideX;
                break;
            
            case 3:
                vConvPad[0]         = _outputLayer._kernelPaddingY; 
                vConvStride[0]      = _outputLayer._kernelStrideY;            
                vConvPad[1]         = _outputLayer._kernelPaddingX; 
                vConvStride[1]      = _outputLayer._kernelStrideX;
                break;
                
            case 4:
                vConvPad[0]         = _outputLayer._kernelPaddingZ; 
                vConvStride[0]      = _outputLayer._kernelStrideZ;
                vConvPad[1]         = _outputLayer._kernelPaddingY; 
                vConvStride[1]      = _outputLayer._kernelStrideY;
                vConvPad[2]         = _outputLayer._kernelPaddingX; 
                vConvStride[2]      = _outputLayer._kernelStrideX;
                break;
        }
        cudnnStatus                 = cudnnSetConvolutionNdDescriptor(_convDesc, _outputLayer._kernelDimensions, vConvPad.data(), vConvStride.data(), vConvUpscale.data(), CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
        CUDNNERROR(cudnnStatus, "NNWeight::NNWeight: cudnnSetConvolutionNdDescriptor failed.");
        
        // Create bias tensor
        vector<int> vBiasDim(5, 1);
        vector<int> vBiasStride(5, 1);
        vBiasDim[1]                 = vFilterDim[0];
        cudnnStatus                 = cudnnSetTensorNdDescriptor(_convBiasTensor, CUDNN_DATA_FLOAT, outputLayer._dimensions + 1, vBiasDim.data(), vBiasStride.data());
        CUDNNERROR(cudnnStatus, "NNWeight::NNWeight: Unable to set bias tensor descriptor");
        
        
        // Calculate weight dimensions
        _size                       = vFilterDim[0] * vFilterDim[1] * _outputLayer._kernelX * _outputLayer._kernelY * _outputLayer._kernelZ;
        _biasSize                   = vFilterDim[0];
        
        
        
        if (getGpu()._id == 0)
        {
            printf("NNWeight::NNWeight: Allocating %" PRIu64 " bytes (%d x %d x %u", _size * sizeof(NNFloat), vFilterDim[0], vFilterDim[1], _outputLayer._kernelX);
            if (_outputLayer._dimensions >= 3)
                printf(" x %u", _outputLayer._kernelY);
            if (_outputLayer._dimensions >= 4)
                printf(" x %u", _outputLayer._kernelZ);
            printf(") for convolutional weights between layers %s and %s\n", inputLayer._name.c_str(), outputLayer._name.c_str());
        }
        
    }
    else
    {
        // Set transform type
        _transform                  = Linear;
    
        // Add to Incoming Large or Outgoing larger
        uint32_t outgoingSize       = outputLayer._stride * 3;
        uint32_t incomingSize       = inputLayer._stride * 2;

        // BUG? Multi-GPU potentially broken here (10/14/16) revisit before checking in
        if (outgoingSize > incomingSize)
        {
            inputLayer._vOutgoingLargerLayer.push_back(&outputLayer);
            inputLayer._vOutgoingLargerWeight.push_back(this);
            _width                  = outputLayer._localStride;    
            _height                 = inputLayer._stride;
        }
        else
        {
            outputLayer._vIncomingLargerLayer.push_back(&inputLayer);
            outputLayer._vIncomingLargerWeight.push_back(this);
            _width                  = outputLayer._stride;
            _height                 = inputLayer._localStride;
        }
        _size                       = _width * _height * _length * _depth * _breadth;
        _biasSize                   = outputLayer._localStride;
        if (getGpu()._id == 0)
            printf("NNWeight::NNWeight: Allocating %" PRIu64 " bytes (%" PRIu64 ", %" PRIu64 ") for fully connected weights between layers %s and %s\n", _size * sizeof(float), _width, _height, inputLayer._name.c_str(), outputLayer._name.c_str());
    }
        
    if (!_bShared)
    {
        _vWeight.resize(_size);
        _pbWeight           = new GpuBuffer<NNFloat>(_size);
        _pbWeightGradient   = new GpuBuffer<NNFloat>(_size);        
    }

    _vBias.resize(_biasSize);
    _pbBias                 = new GpuBuffer<NNFloat>(_biasSize);

    // Add bias graident to convolutions
    if (_transform == Convolution)
    {
        _pbBiasGradient     = new GpuBuffer<NNFloat>(_biasSize);
    }
}

NNWeight::~NNWeight()
{
    if (!_bShared)
    {
        delete _pbWeight;
        delete _pbWeightVelocity;
        delete _pbWeightGradient;
        delete _pbWeightGradientVelocity;
    }
    delete _pbBias;
    delete _pbBiasVelocity;    
    delete _pbBiasGradient;
    delete _pbBiasGradientVelocity;
}

void NNWeight::ClearVelocity()
{
    cudaMemset(_pbWeightVelocity->_pDevData, 0, _size * sizeof(NNFloat));
    cudaMemset(_pbBiasVelocity->_pDevData, 0, _biasSize * sizeof(NNFloat));
    if (_pbWeightGradientVelocity != NULL)
        cudaMemset(_pbWeightGradientVelocity->_pDevData, 0, _size * sizeof(NNFloat));
    if (_pbBiasGradientVelocity != NULL)
        cudaMemset(_pbBiasGradientVelocity->_pDevData, 0, _biasSize * sizeof(NNFloat));
}

void NNWeight::ClearGradient()
{
    cudaMemset(_pbWeightGradient->_pDevData, 0, _size * sizeof(NNFloat));
}

void NNWeight::Randomize()
{
    if (!_bShared)
    {
        NNFloat scale, bias;        
        switch (_outputLayer._weightInit)
        {
        case CaffeXavier:
            // Initialize weights to range from _weightInitScale * (-sqrt(3 / n_output) to sqrt(3 / n_output))
            // ala the adaptation of Gloriot and Bengio in Caffe
            curandGenerateUniform(getGpu()._RNG, _pbWeight->_pDevData, _size);
            scale               = _outputLayer._weightInitScale * 2.0f * sqrtf(3.0f / _outputLayer._stride);
            bias                = 0.5f * scale;                 
            kScaleAndBias(_pbWeight->_pDevData, _size, scale, bias);
            break;
            
        case Xavier:
            // Initialize weights to range from _weightInitScale * (-sqrt(6 / (n_output+n_input)) and sqrt(6 / (n_output+n_input)))
            // ala Gloriot and Bengio
            curandGenerateUniform(getGpu()._RNG, _pbWeight->_pDevData, _size);
            scale               = _outputLayer._weightInitScale * sqrtf(6.0f / (_outputLayer._stride + _inputLayer._stride));
            bias                = 0.5f * scale;
            kScaleAndBias(_pbWeight->_pDevData, _size, scale, bias);
            break;
     
        case Uniform:
            // Initialize weights uniformly from -_weightInitScale to +_weightInitScale
            curandGenerateUniform(getGpu()._RNG, _pbWeight->_pDevData, _size);
            scale               = 2.0f * _outputLayer._weightInitScale;
            bias                = 0.5f * scale;                 
            kScaleAndBias(_pbWeight->_pDevData, _size, scale, bias);  
            break;
            
        case Gaussian:
            // Initialize weights to N(0, _weightInitScale)
            curandGenerateNormal(getGpu()._RNG, _pbWeight->_pDevData, _size, 0.0f, _outputLayer._weightInitScale);
            break;        
            
        case UnitBall:      
            // Initialize weights uniformly from 0 to _weightInitScale  
            curandGenerateUniform(getGpu()._RNG, _pbWeight->_pDevData, _size);
            scale               = _outputLayer._weightInitScale;              
            kScaleAndBias(_pbWeight->_pDevData, _size, scale, 0.0f);     
            break;
          
        case Constant:
            // Initialize all weights to _weightInitScale
            cudaMemset(_pbWeight->_pDevData, 0, _size * sizeof(NNFloat));
            kScaleAndBias(_pbWeight->_pDevData, _size, (NNFloat)0.0, _outputLayer._weightInitScale); 
            break;
        };
    }
        
    // Initialize Biases
    cudaMemset(_pbBias->_pDevData, 0, _biasSize * sizeof(NNFloat));
    kScaleAndBias(_pbBias->_pDevData, _biasSize, (NNFloat)0.0, -_outputLayer._biasInit); 
}

void NNWeight::Lock()
{
    _bLocked                = true;
}

void NNWeight::Unlock()
{
    _bLocked                = false;
}

void NNWeight::RefreshState(NNNetwork* pNetwork, TrainingMode mode)
{
    if (mode != TrainingMode::SGD)
    {
        if (!_pbWeightVelocity)
            _pbWeightVelocity               = new GpuBuffer<NNFloat>(_size);
        if (!_pbBiasVelocity)
            _pbBiasVelocity                 = new GpuBuffer<NNFloat>(_biasSize);
            
        // Add additional buffers for AdaDelta and Adam
        if (mode == TrainingMode::AdaDelta)
        {
            if (!_pbWeightGradientVelocity)
                _pbWeightGradientVelocity   = new GpuBuffer<NNFloat>(_size);
            if (!_pbBiasGradientVelocity)
                _pbBiasGradientVelocity     = new GpuBuffer<NNFloat>(_biasSize);            
        }
        else
        {
            delete _pbWeightGradientVelocity;
            delete _pbBiasGradientVelocity;        
            _pbWeightGradientVelocity       = NULL;
            _pbBiasGradientVelocity         = NULL;
        }
    }
    else
    {
        delete _pbWeightVelocity;
        delete _pbBiasVelocity;
        delete _pbWeightGradientVelocity;
        delete _pbBiasGradientVelocity;
        _pbWeightVelocity                   = NULL;
        _pbBiasVelocity                     = NULL;
        _pbWeightGradientVelocity           = NULL;
        _pbBiasGradientVelocity             = NULL;
    }
    
    // If convolution layer, recalculate Convolution settings
    if (_outputLayer._type == NNLayer::Type::Convolutional)
    {
        printf("Getting algorithm between %s and %s\n", _inputLayer._name.c_str(), _outputLayer._name.c_str());
        size_t workspaceSize;
        cudnnStatus_t cudnnStatus           = cudnnGetConvolutionForwardAlgorithm(getGpu()._cuDNNHandle,
                                              _inputLayer._tensorDescriptor,
                                              _convFilterDesc,
                                              _convDesc,
                                              _outputLayer._tensorDescriptor,
                                              CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                              1,
                                              &_convFWAlgo);
        CUDNNERROR(cudnnStatus, "NNWeight::Refresh: cudnnGetConvolutionForwardAlgorithm failed.");                                              
        
        cudnnStatus                         = cudnnGetConvolutionForwardWorkspaceSize(getGpu()._cuDNNHandle,
                                              _inputLayer._tensorDescriptor,
                                              _convFilterDesc,
                                              _convDesc,
                                              _outputLayer._tensorDescriptor,
                                              _convFWAlgo,
                                              &workspaceSize); 
        CUDNNERROR(cudnnStatus, "NNWeight::Refresh: cudnnGetConvolutionForwardWorkspaceSize failed.");
        pNetwork->SetCUDNNWorkspace(workspaceSize);
         
        cudnnStatus                         = cudnnGetConvolutionBackwardFilterAlgorithm(getGpu()._cuDNNHandle,
                                             _inputLayer._tensorDescriptor,
                                             _outputLayer._tensorDescriptor,
                                             _convDesc,
                                             _convFilterDesc,
                                             CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
                                             0,
                                             &_convBWWeightAlgo);
        CUDNNERROR(cudnnStatus, "NNWeight::Refresh: cudnnGetConvolutionBackwardFilterAlgorithm failed.");                                             
    
        // cudnn BUG hacking in deterministic backprop
        //_convBWWeightAlgo                   = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

        cudnnStatus                         = cudnnGetConvolutionBackwardFilterWorkspaceSize(getGpu()._cuDNNHandle,
                                            _inputLayer._tensorDescriptor,
                                            _outputLayer._tensorDescriptor,
                                            _convDesc,
                                            _convFilterDesc,
                                            _convBWWeightAlgo,
                                            &workspaceSize);
        CUDNNERROR(cudnnStatus, "NNWeight::Refresh: cudnnGetConvolutionBackwardFilterWorkspaceSize failed.");
        pNetwork->SetCUDNNWorkspace(workspaceSize); 
                
        cudnnStatus                         = cudnnGetConvolutionBackwardDataAlgorithm(getGpu()._cuDNNHandle,
                                            _convFilterDesc,
                                            _outputLayer._tensorDescriptor,
                                            _convDesc,
                                            _inputLayer._tensorDescriptor,
                                            CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                                            0,
                                            &_convBWDeltaAlgo);
        CUDNNERROR(cudnnStatus, "NNWeight::Refresh: cudnnGetConvolutionBackwardDataAlgorithm failed.");      
        // cudnn BUG hacking in deterministic backprop
        //_convBWDeltaAlgo                    = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;                                         
                                        
        cudnnStatus                         = cudnnGetConvolutionBackwardDataWorkspaceSize(getGpu()._cuDNNHandle,
                                              _convFilterDesc,
                                              _outputLayer._tensorDescriptor,
                                              _convDesc,
                                              _inputLayer._tensorDescriptor,
                                              _convBWDeltaAlgo,
                                              &workspaceSize);
        CUDNNERROR(cudnnStatus, "NNWeight::Refresh: cudnnGetConvolutionBackwardDataWorkspaceSize failed.");
        pNetwork->SetCUDNNWorkspace(workspaceSize);
        
        // Validate output layer size
        vector<int> vOutputDim(8, 1);
        cudnnStatus                         = cudnnGetConvolutionNdForwardOutputDim(_convDesc,
                                                                                    _inputLayer._tensorDescriptor,
                                                                                    _convFilterDesc,
                                                                                    _outputLayer._dimensions + 1,
                                                                                    vOutputDim.data());
        CUDNNERROR(cudnnStatus, "NNWeight::Refresh: cudnnGetConvolutionNdForwardOutputDim failed.");                                                                                    
        size_t dim = 1;
        for (size_t i = 0; i < _outputLayer._dimensions + 1; i++)
            dim *= vOutputDim[i];
        if (dim != _outputLayer._maxLocalStride * _outputLayer._localBatch)
        {
            if (getGpu()._id == 0)
                printf("Output layer %s has incorrectly calculated dimensions for cuDNN.\n", _outputLayer._name.c_str());
            getGpu().Shutdown();
        }
    }
}

float NNWeight::CalculateRegularizationError(NNFloat lambda)
{
    // Error on a shared set of weights is only calculated from its original source
    if (_bShared)
        return 0;
    else
        return kCalculateRegularizationError(lambda, _pbWeight->_pDevData, _size);
}

// Calculates Unit(l)^T * Delta(l + 1), the product of a [stride][batch] and [batch][outgoing stride] matrix
// and then updates weight values utilizing the current training mode
void NNWeight::UpdateWeights(TrainingMode trainingMode, uint32_t batch, NNFloat alpha, NNFloat lambda, NNFloat mu)
{
    cublasStatus_t cstatus;

    // Skip update if weights are locked
    if (_bLocked)
        return; 

    // Update weights if the original holder or unshared in general
    if (!_bShared)
    {
        switch (trainingMode)
        {
            case SGD:
                kSGDUpdateWeights(alpha, lambda, _size, _pbWeightGradient->_pDevData, _pbWeight->_pDevData);
                break;
                
            case Momentum:
                kMomentumUpdateWeights(alpha, lambda, mu, _size, _pbWeightVelocity->_pDevData, _pbWeightGradient->_pDevData, _pbWeight->_pDevData);
                break;
                        
            case AdaGrad:
                kAdaGradUpdateWeights(alpha, lambda, _size, _pbWeightVelocity->_pDevData, _pbWeightGradient->_pDevData, _pbWeight->_pDevData);
                break;
                        
            case Nesterov:
                kNesterovUpdateWeights(alpha, lambda, mu, _size, _pbWeightVelocity->_pDevData, _pbWeightGradient->_pDevData, _pbWeight->_pDevData);
                break;
                        
            case RMSProp:
                kRMSPropUpdateWeights(alpha, lambda, mu, _size, _pbWeightVelocity->_pDevData, _pbWeightGradient->_pDevData, _pbWeight->_pDevData);
                break;

            case AdaDelta:
                kAdaDeltaUpdateWeights(lambda, mu, _size, _pbWeightVelocity->_pDevData, _pbWeightGradient->_pDevData, _pbWeightGradientVelocity->_pDevData, _pbWeight->_pDevData);
                break;     
        }
    }

    // Biases are unshared so always update them
    if (_transform == Linear)
    {
        switch (trainingMode)
        {
            case SGD:
                kSGDUpdateBiases(alpha, batch, _biasSize, _outputLayer._pbDelta->_pDevData, _pbBias->_pDevData);
                break;

            case Momentum:
                kMomentumUpdateBiases(alpha, mu, batch, _biasSize, _outputLayer._pbDelta->_pDevData, _pbBiasVelocity->_pDevData, _pbBias->_pDevData);
                break;
                    
            case AdaGrad:
                kAdaGradUpdateBiases(alpha, batch, _outputLayer._localStride, _outputLayer._pbDelta->_pDevData, _pbBiasVelocity->_pDevData, _pbBias->_pDevData);
                break;
                    
            case Nesterov:
                kNesterovUpdateBiases(alpha, mu, batch, _outputLayer._localStride, _outputLayer._pbDelta->_pDevData, _pbBiasVelocity->_pDevData, _pbBias->_pDevData);
                break;
                    
            case RMSProp:
                kRMSPropUpdateBiases(alpha, mu, batch, _outputLayer._localStride, _outputLayer._pbDelta->_pDevData, _pbBiasVelocity->_pDevData, _pbBias->_pDevData);
                break;
                
            case AdaDelta:
                kAdaDeltaUpdateBiases(mu, batch, _outputLayer._localStride, _outputLayer._pbDelta->_pDevData, _pbBiasVelocity->_pDevData, _pbBiasGradientVelocity->_pDevData, _pbBias->_pDevData);
                break;                         
        }
    }
    else
    {
        // Because cuDNN is stupid, resort to hijacking weight update routines to compensate for the stupid within cuDNN
        switch (trainingMode)
        {
            case SGD:
                kSGDUpdateWeights(alpha, (NNFloat)0.0, _biasSize, _pbBiasGradient->_pDevData, _pbBias->_pDevData);
                break;

            case Momentum:
                kMomentumUpdateWeights(alpha, (NNFloat)0.0, mu, _biasSize, _pbBiasVelocity->_pDevData, _pbBiasGradient->_pDevData, _pbBias->_pDevData);
                break;
                    
            case AdaGrad:
                kAdaGradUpdateWeights(alpha, (NNFloat)0.0, _biasSize, _pbBiasVelocity->_pDevData, _pbBiasGradient->_pDevData, _pbBias->_pDevData);
                break;
                        
            case Nesterov:
                kNesterovUpdateWeights(alpha, (NNFloat)0.0, mu, _biasSize, _pbBiasVelocity->_pDevData, _pbBiasGradient->_pDevData, _pbBias->_pDevData);
                break;
                        
            case RMSProp:
                kRMSPropUpdateWeights(alpha, (NNFloat)0.0, mu, _biasSize, _pbBiasVelocity->_pDevData, _pbBiasGradient->_pDevData, _pbBias->_pDevData);
                break;

            case AdaDelta:
                kAdaDeltaUpdateWeights((NNFloat)0.0, mu, _biasSize, _pbBiasVelocity->_pDevData, _pbBiasGradient->_pDevData, _pbBiasGradientVelocity->_pDevData, _pbBias->_pDevData);
                break;                 
        }       
    }
#if 0
        if (_width < 1024)
        {
            _pbBias->Download(&_vBias[0]);
            for (int i = 0; i < _width; i++)
                printf("%3d %16.8f\n", i, _vBias[i]);
        }
#endif
          
    // and only do so after all updates have been applied    
    if ((_norm > (NNFloat)0.0) && (!_bShared))
    {
        if (getGpu()._numprocs == 1)  // TODO Detect data-parallel here
            kNormalizeWeights(_norm, _outputLayer._stride, _inputLayer._localStride, _pbWeight->_pDevData);
        else
        {
            NNFloat* pMagnitude                 = getGpu()._pNetwork->GetScratchBuffer(_outputLayer._stride);
            kCalculateWeightMagnitudes(_outputLayer._stride, _inputLayer._localStride, _pbWeight->_pDevData, pMagnitude);
            getGpu()._pNetwork->P2P_Allreduce(pMagnitude, _outputLayer._stride);
            kNormalizeWeightMagnitudes(_norm, _outputLayer._stride, _inputLayer._localStride, _pbWeight->_pDevData, pMagnitude);       
        }
    }
}

bool NNWeight::WriteNetCDF(netCDF::NcFile& nc, uint32_t index, NNFloat* pWeight, NNFloat* pBias)
{
    bool bResult                = true;
    if (getGpu()._id == 0)
    {
        string wstring          = "weight" + std::to_string(index) + "_";
        nc.putAtt(wstring + "inputLayer", _inputLayer._name);
        nc.putAtt(wstring + "outputLayer", _outputLayer._name);

        nc.putAtt(wstring + "width", ncUint64, (unsigned long long int)_width);  
        nc.putAtt(wstring + "height", ncUint64, (unsigned long long int)_height);
        nc.putAtt(wstring + "length", ncUint64, (unsigned long long int)_length);
        nc.putAtt(wstring + "depth", ncUint64, (unsigned long long int)_depth);
        nc.putAtt(wstring + "breadth", ncUint64, (unsigned long long int)_breadth);  

        nc.putAtt(wstring + "bShared", ncUint, (uint32_t)_bShared);
        nc.putAtt(wstring + "bLocked", ncUint, (uint32_t)_bLocked);
        nc.putAtt(wstring + "norm", ncFloat, _norm);
        
        NcDim biasDim           = nc.addDim(wstring + "biasDim", _biasSize);
        NcVar biasVar           = nc.addVar(wstring + "bias", ncFloat, biasDim);
        if (pBias == NULL)
            pBias               = _vBias.data();
        biasVar.putVar(pBias);  
        if (_bShared)
        {
            nc.putAtt(wstring + "bTransposed", ncUint, (uint32_t)_bTransposed);
            nc.putAtt(wstring + "sourceInputLayer", _pSharedWeight->_inputLayer._name);
            nc.putAtt(wstring + "sourceOutputLayer", _pSharedWeight->_outputLayer._name);
        }
        else
        {

#if 0
        printf("Weights %d %lu %lu\n", index, _vWeight.size(), _vBias.size());
        for (int i = 0; i < 20; i++)
            printf("%3d %16.8f %16.8f\n", i, _vWeight[i], _vBias[i]);
#endif
            NcDim weightDim     = nc.addDim(wstring + "weightDim", _size);            
            NcVar weightVar     = nc.addVar(wstring + "weights", ncFloat, weightDim);            
            if (!pWeight)
                pWeight         = _vWeight.data();
            weightVar.putVar(pWeight);
        }
    }

    return bResult;
}

NNFloat* NNWeight::GetWeightBuffer()
{
    return _pbWeight ? _pbWeight->_pDevData : NULL;
}

NNFloat* NNWeight::GetWeightGradientBuffer()
{
    return _pbWeightGradient ? _pbWeightGradient->_pDevData : NULL;
}

uint64_t NNWeight::GetBufferSize()
{
    return _size;
}

bool NNWeight::CopyWeights(NNWeight* pWeight)
{
    bool bValid                 = true;

    // Check for valid weight pointer
    if (!pWeight)
    {
        if (getGpu()._id == 0)
            printf("NNWeight::CopyWeights: Invalid weight pointer.\n");
        bValid                  = false;
    }
    else if ((pWeight->_width != _width) || (pWeight->_height != _height) || (pWeight->_length != _length))
    {
        if (getGpu()._id == 0)
        {
            printf("NNWeight::CopyWeights: Mismatched weight dimensions (%" PRIu64 " x %" PRIu64 " x %" PRIu64") versus (%" PRIu64 " x %" PRIu64 " x %" PRIu64 ").\n", _width, _height, _length,
            pWeight->_width, pWeight->_height, pWeight->_length);
        }
        bValid                  = false;        
    }
    else
    {
        _vWeight                = pWeight->_vWeight;
        _vBias                  = pWeight->_vBias;
        _pbWeight->Upload(&_vWeight[0]);
        _pbBias->Upload(&_vBias[0]);
    }
    return bValid;
}

void NNWeight::Dump(string fname, NNFloat* pBuffer)
{
    // Create vector to hold entire weight matrix and resize it as such on process 0
    vector<NNFloat> vWeight;

    // Special case single GPU TODO data-parallel weights
    if (getGpu()._numprocs == 1)
    {
        vWeight.resize(_size);
        cudaMemcpy(vWeight.data(), pBuffer, _size * sizeof(NNFloat), cudaMemcpyDefault);
    }
    else
    {
        // Cannibalize system weight vector to hold buffer data
        if (getGpu()._id == 0)
            vWeight.resize(_outputLayer._stride * _inputLayer._stride);        
        uint32_t outgoingSize       = _outputLayer._stride * 3;               
        uint32_t incomingSize       = _inputLayer._stride * 2;     
        cudaMemcpy(_vWeight.data(), pBuffer, _size * sizeof(NNFloat), cudaMemcpyDefault);

        // Reduce weight data into GPU 0
        if (getGpu()._id == 0)
        {
            NNFloat* pWeight            = vWeight.data();                    
            if (outgoingSize > incomingSize)
            {
                cudaMemcpy2D(pWeight, _outputLayer._stride * sizeof(NNFloat), _vWeight.data(), _outputLayer._localStride * sizeof(NNFloat), _outputLayer._localStride * sizeof(NNFloat), _inputLayer._stride, cudaMemcpyDefault);
                pWeight                += _outputLayer._localStride;
                for (uint32_t i = 1; i < getGpu()._numprocs; i++)
                {                        
                    uint64_t size;
                    MPI_Status status;                
                    MPI_Recv(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &status);
                    vector<NNFloat> vTemp(size);
                    MPI_Recv(vTemp.data(), size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
                    uint64_t lstride    = size / _inputLayer._stride;
                    NNFloat* pSrcWeight = vTemp.data();
                    NNFloat* pDstWeight = pWeight;
                    for (uint32_t j = 0; j < _inputLayer._stride; j++)
                    {
                        memcpy(pDstWeight, pSrcWeight, lstride * sizeof(NNFloat));
                        pSrcWeight     += lstride;
                        pDstWeight     += _outputLayer._stride;
                    }                          
                    pWeight            += lstride;
                }
            }
            else
            {
                cudaMemcpy(pWeight, _vWeight.data(), _outputLayer._stride * _inputLayer._localStride * sizeof(NNFloat), cudaMemcpyDefault);
                pWeight                += _outputLayer._stride * _inputLayer._localStride;
                for (uint32_t i = 1; i < getGpu()._numprocs; i++)
                {
                    uint64_t size;
                    MPI_Status status;                
                    MPI_Recv(&size, 1, MPI_UINT64_T, i, 0, MPI_COMM_WORLD, &status);
                    MPI_Recv(pWeight, size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &status);
                    pWeight            += size;
                }                        
            }
        }              
        else
        {
            uint64_t size               = _vWeight.size();
            MPI_Send(&size, 1, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD);
            MPI_Send(_vWeight.data(), size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);                  
        }

    }

    // Write file
    if (getGpu()._id == 0)
    {
        FILE* fp                        = fopen(fname.c_str(), "w");
        NNFloat* pData                  = vWeight.data();
        for (int i = 0; i < _inputLayer._stride; i++)
        {
            for (int j = 0; j < _outputLayer._stride; j++)
            {
                fprintf(fp, "%12.9f ", *pData);
                pData++;
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
    }
}
