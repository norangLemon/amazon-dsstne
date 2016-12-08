/*
   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#ifndef NNWEIGHT_DESCRIPTOR_H
#define NNWEIGHT_DESCRIPTOR_H
#ifndef __NVCC__

#include <iostream>
#include <string>
#include <vector>

#include "GpuTypes.h"

using std::ostream;
using std::string;
using std::vector;

struct NNWeightDescriptor
{
    string                  _inputLayer;
    string                  _outputLayer;
    uint64_t                _width;
    uint64_t                _height;
    uint64_t                _length;
    uint64_t                _depth;
    uint64_t                _breadth;
    vector<NNFloat>         _vWeight;
    vector<NNFloat>         _vBias;
    bool                    _bShared;
    bool                    _bTransposed;
    bool                    _bLocked;
    NNFloat                 _norm;
    string                  _sourceInputLayer;     // _sourceInputLayer and _sourceOutputLayer collectively
    string                  _sourceOutputLayer;    // specify which weight matrix will be shared here

    NNWeightDescriptor();
};

namespace netCDF {
    class NcFile;
}

bool LoadNNWeightDescriptorNetCDF(const string& fname, netCDF::NcFile& nc, uint32_t index, NNWeightDescriptor& wd);
ostream& operator<< (ostream& out, NNWeightDescriptor& d);
uint32_t MPI_Bcast_NNWeightDescriptor(NNWeightDescriptor& d);

#endif // __NVCC__
#endif // NNWEIGHT_DESCRIPTOR_H
