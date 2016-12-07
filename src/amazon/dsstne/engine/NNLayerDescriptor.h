/*
   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#ifndef NNLAYER_DESCRIPTOR_H
#define NNLAYER_DESCRIPTOR_H
#ifndef __NVCC__

#include <string>
#include <vector>

#include "GpuTypes.h"
#include "NNLayer.h"

using std::vector;
using std::string;

struct NNLayerDescriptor
{
    string                  _name;                      // Name of layer
    NNLayer::Kind           _kind;                      // Input, Hidden, Pooling, or Output
    NNLayer::Type           _type;                      // FullyConnected, Convolutional, or Pooling
    PoolingFunction         _poolingFunction;           // Pooling function for pooling layers
    string                  _dataSet;                   // Name of dataset for input and output layers
    vector<string>          _vSource;                   // Source layers/data sets
    vector<string>          _vSkip;                     // Skip layer sources
    uint32_t                _Nx;                        // Unit X size (or image or voxel width)
    uint32_t                _Ny;                        // Image or voxel height (or 1)
    uint32_t                _Nz;                        // Number of image neurons or voxel depth (or 1)
    uint32_t                _Nw;                        // Number of voxel neurons (or 1)
    uint32_t                _dimensions;                // Convolution unit or input data dimensions
    bool                    _bDimensionsProvided;       // Have all dimensions been determined?
    WeightInitialization    _weightInit;                // Weight initialization scheme
    NNFloat                 _weightInitScale;           // Weight Initialization scaling factor
    NNFloat                 _biasInit;                  // Bias initialization value
    uint32_t                _kernelX;                   // kernel X size
    uint32_t                _kernelY;                   // kernel Y size
    uint32_t                _kernelZ;                   // kernel Z size
    uint32_t                _kernelStrideX;             // kernel X stride
    uint32_t                _kernelStrideY;             // kernel Y stride
    uint32_t                _kernelStrideZ;             // kernel Z stride
    uint32_t                _kernelPaddingX;            // kernel X padding
    uint32_t                _kernelPaddingY;            // kernel Y padding
    uint32_t                _kernelPaddingZ;            // kernel Z padding
    uint32_t                _kernelDimensions;          // Number of components to kernel and kernel stride
    NNFloat                 _weightNorm;                // Maximum weight vector length
    NNFloat                 _deltaNorm;                 // Maximum delta vector length
    NNFloat                 _pDropout;                  // Dropout probability
    Activation              _activation;                // Activation function
    NNFloat                 _sparsenessPenalty_p;       // Layer-specific sparseness target
    NNFloat                 _sparsenessPenalty_beta;    // Layer-specific sparseness penalty weight
    uint32_t                _attributes;                // Specific layer properties
    NNLayerDescriptor();
};

namespace netCDF {
    class NcFile;
}

bool LoadNNLayerDescriptorNetCDF(const string& fname, netCDF::NcFile& nc, uint32_t index, NNLayerDescriptor& ld);
ostream& operator<< (ostream& out, NNLayerDescriptor& d);
uint32_t MPI_Bcast_NNLayerDescriptor(NNLayerDescriptor& d);

#endif // __NVCC__
#endif // NNLAYER_DESCRIPTOR_H
