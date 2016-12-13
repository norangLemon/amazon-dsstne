/*
   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#ifndef NNDATA_SET_BASE_H
#define NNDATA_SET_BASE_H
#ifndef __NVCC__

#include <memory>
#include <netcdf>
#include <string>
#include <vector>

#include "GpuTypes.h"
#include "NNEnum.h"
#include "NNTypes.h"

using std::string;
using std::unique_ptr;
using std::vector;

class NNLayer;

struct NNDataSetBase {
    string                      _name;                          // Dataset name
    NNDataSetEnums::DataType    _dataType;                      // Dataset type (see above enum)
    uint32_t                    _attributes;                    // Dataset characteristics (see NNDataSetEnum::Attributes in NNEnum.h)
    uint32_t                    _examples;                      // Number of examples
    uint32_t                    _localExamples;                 // Number of local examples when data sharded
    uint32_t                    _dimensions;                    // Dimensionality of data set
    uint32_t                    _width;                         // Dataset x dimension
    uint32_t                    _height;                        // Dataset y dimension
    uint32_t                    _length;                        // Dataset z dimension
    uint32_t                    _stride;                        // Stride between examples
    NNDataSetEnums::Sharding    _sharding;                      // Sharding of dataset for parallel execution
    uint32_t                    _minX;                          // Beginning of local X sharding for model parallel execution
    uint32_t                    _maxX;                          // End of local X sharding for model parallel execution
    uint64_t                    _sparseDataSize;                // Total sparse datapoints
    uint32_t                    _maxSparseDatapoints;           // Maximum observed sparse datapoints per example
    NNFloat                     _sparseDensity;                 // Overall sparse density (0.0 - 1.0)
    vector<uint64_t>            _vSparseStart;                  // Vector of sparse datapoint starts per example
    unique_ptr<GpuBuffer<uint64_t>> _pbSparseStart;             // GPU copy of _vSparseStart
    vector<uint64_t>            _vSparseEnd;                    // Vector of sparse datapoint ends per example
    unique_ptr<GpuBuffer<uint64_t>> _pbSparseEnd;               // GPU copy of _vSparseEnd
    vector<uint32_t>            _vSparseIndex;                  // Vector of sparse indices
    unique_ptr<GpuBuffer<uint32_t>> _pbSparseIndex;             // GPU copy of _vSparseIndex
    unique_ptr<GpuBuffer<NNFloat>> _pbDenoisingRandom;          // Denoising randoms

    // Transposed sparse lookup for sparse backpropagation
    vector<uint64_t>            _vSparseDatapointCount;
    vector<uint32_t>            _vSparseTransposedStart;
    uint32_t                    _sparseTransposedIndices;
    unique_ptr<GpuBuffer<uint32_t>> _pbSparseTransposedStart;
    unique_ptr<GpuBuffer<uint32_t>> _pbSparseTransposedEnd;
    unique_ptr<GpuBuffer<uint32_t>> _pbSparseTransposedIndex;

    // States
    bool                        _bDenoising;
    bool                        _bDirty;
    uint32_t                    _batch;




    NNDataSetBase();
    NNDataSetDimensions GetDimensions();
    uint32_t GetExamples();

    virtual bool SaveNetCDF(const string& fname) = 0;
    virtual bool WriteNetCDF(netCDF::NcFile& nfc, const string& fname, const uint32_t n) = 0;
    virtual ~NNDataSetBase() = 0;
    virtual void RefreshState(uint32_t batch) = 0;
    virtual bool Shard(NNDataSetEnums::Sharding sharding) = 0;
    virtual bool UnShard() = 0;
    virtual vector<tuple<uint64_t, uint64_t> > getMemoryUsage() = 0;
    virtual bool CalculateSparseDatapointCounts() = 0;
    virtual bool GenerateSparseTransposedMatrix(uint32_t batch, NNLayer* pLayer) = 0;
    virtual bool CalculateSparseTransposedMatrix(uint32_t position, uint32_t batch, NNLayer* pLayer) = 0;
    virtual bool CalculateSparseTransposedDenoisedMatrix(uint32_t position, uint32_t batch, NNLayer* pLayer) = 0;
    virtual bool CalculateSparseTransposedWeightGradient(NNFloat alpha, NNFloat beta, uint32_t m, uint32_t n, NNFloat* pDelta, NNFloat* pWeightGradient) = 0;
    virtual bool SetDenoising(bool flag) = 0;
    virtual void GenerateDenoisingData() = 0;
    virtual bool LoadInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit) = 0;
    virtual bool LoadSparseInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit) = 0;
    virtual bool LoadSparseDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit) = 0;
    virtual bool CalculateSparseZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, NNFloat* pUnit, NNFloat beta = (NNFloat)0.0) = 0;
    virtual bool CalculateSparseDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, NNFloat* pUnit, NNFloat beta = (NNFloat)0.0) = 0;
    virtual void CalculateL1Error(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit) = 0;
    virtual void CalculateL2Error(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit) = 0;
    virtual void CalculateCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit) = 0;
    virtual void CalculateScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit) = 0;
    virtual void CalculateMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit) = 0;
    virtual void CalculateMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit) = 0;
    virtual bool CalculateL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta) = 0;
    virtual bool CalculateCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta) = 0;
    virtual bool CalculateScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta) = 0;
    virtual bool CalculateOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta) = 0;
    virtual void CalculateDataScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit) = 0;
    virtual bool CalculateDataScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta) = 0;
};

vector<NNDataSetBase*> LoadNetCDF(const string& fname);
bool SaveNetCDF(const string& fname, vector<NNDataSetBase*> vDataset);
vector<NNDataSetBase*> LoadImageData(const string& fname);
vector<NNDataSetBase*> LoadCSVData(const string& fname);
vector<NNDataSetBase*> LoadJSONData(const string& fname);
vector<NNDataSetBase*> LoadAudioData(const string& name);

#endif // __NVCC__
#endif // NNDATA_SET_BASE_H
