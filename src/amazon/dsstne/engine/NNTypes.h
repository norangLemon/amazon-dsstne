/*


   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#ifndef NNTYPES_H
#define NNTYPES_H
#include <vector>
#include <set>

#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <netcdf>
#ifndef __NVCC__
#include <tuple>
#include <json/json.h>
#endif
#include <sys/time.h>
#include <cmath>

class NNDataSetBase;
class NNLayer;
class NNNetwork;
class NNWeight;

// Activates step by step CPU validation
#define VALIDATION
#ifdef VALIDATION
extern "C"
{
    #include <cblas.h>
}
#endif


static const float NN_VERSION       = 0.85f;
static const float MIN_ERROR        = 1.0e-12f;
static const float MIN_ACTIVATION   = 0.000001f;
static const float MAX_ACTIVATION   = 0.999999f;
static const float MAX_VALUE        = 999999999999999.0f;

template <typename T> struct GpuBuffer;

enum 
{
    DefaultBatch    = 512
};

enum Mode {
    Prediction = 0,
    Training = 1,
    Validation = 2,
    Unspecified = 3
};

enum TrainingMode 
{
    SGD = 0,
    Momentum = 1,
    AdaGrad = 2,
    Nesterov = 3,
    RMSProp = 4,
    AdaDelta = 5,
};

ostream& operator<< (ostream& out, const TrainingMode& e);

enum ErrorFunction 
{
    L1,
    L2,
    CrossEntropy,
    ScaledMarginalCrossEntropy,
    DataScaledMarginalCrossEntropy,
};

ostream& operator<< (ostream& out, const ErrorFunction& e);

enum Activation {
    Sigmoid,
    Tanh,
    RectifiedLinear,
    Linear,
    ParametricRectifiedLinear,
    SoftPlus,
    SoftSign,
    SoftMax,
    ReluMax,
    LinearMax,
    ExponentialLinear,
};

ostream& operator<< (ostream& out, const Activation& a);

enum WeightInitialization
{
    Xavier,
    CaffeXavier,
    Gaussian,
    Uniform,
    UnitBall,
    Constant 
};
    
ostream& operator<< (ostream& out, const WeightInitialization& w);
    
enum PoolingFunction {
    None,
    Max,
    Average,
    LRN,
    Maxout,
    Stochastic,
    LCN,
    GlobalTemporal,
};

ostream& operator<< (ostream& out, const PoolingFunction& p);

#include "kernels.h"
#include "GpuSort.h"
#include "NNEnum.h"
#include "NNWeight.h"
#include "NNLayer.h"
#include "NNNetwork.h"


int MPI_Bcast_string(string& s);

struct NNDataSetDimensions
{
    uint32_t _dimensions;
    uint32_t _width;
    uint32_t _height;
    uint32_t _length;
};

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
    GpuBuffer<uint64_t>*        _pbSparseStart;                 // GPU copy of _vSparseStart
    vector<uint64_t>            _vSparseEnd;                    // Vector of sparse datapoint ends per example
    GpuBuffer<uint64_t>*        _pbSparseEnd;                   // GPU copy of _vSparseEnd
    vector<uint32_t>            _vSparseIndex;                  // Vector of sparse indices
    GpuBuffer<uint32_t>*        _pbSparseIndex;                 // GPU copy of _vSparseIndex
    GpuBuffer<NNFloat>*         _pbDenoisingRandom;             // Denoising randoms 
    
    // Transposed sparse lookup for sparse backpropagation
    vector<uint64_t>            _vSparseDatapointCount;
    vector<uint32_t>            _vSparseTransposedStart;
    uint32_t                    _sparseTransposedIndices;
    GpuBuffer<uint32_t>*        _pbSparseTransposedStart;
    GpuBuffer<uint32_t>*        _pbSparseTransposedEnd;
    GpuBuffer<uint32_t>*        _pbSparseTransposedIndex;

    // States
    bool                        _bDenoising;
    bool                        _bDirty;
    uint32_t                    _batch;
    
      


    NNDataSetBase();
    NNDataSetDimensions GetDimensions();
    uint32_t GetExamples() { return _examples; };

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
    virtual bool GenerateDenoisingData() = 0;
    virtual bool LoadInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit) = 0;
    virtual bool LoadSparseInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit) = 0;
    virtual bool LoadSparseDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit) = 0;
    virtual bool CalculateSparseZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, NNFloat* pUnit, NNFloat beta = (NNFloat)0.0) = 0;
    virtual bool CalculateSparseDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, NNFloat* pUnit, NNFloat beta = (NNFloat)0.0) = 0;
    virtual float CalculateL1Error(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit) = 0;
    virtual float CalculateL2Error(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit) = 0;
    virtual float CalculateCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit) = 0;
    virtual float CalculateScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit) = 0;
    virtual float CalculateMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit) = 0;
    virtual float CalculateMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit) = 0;
    virtual bool CalculateL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta) = 0;
    virtual bool CalculateCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta) = 0;   
    virtual bool CalculateScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta) = 0;   
    virtual bool CalculateOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta) = 0;  
    virtual float CalculateDataScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit) = 0;
    virtual bool CalculateDataScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta) = 0;
};

ostream& operator<< (ostream& out, NNDataSetEnums::Attributes& a);
ostream& operator<< (ostream& out, NNDataSetEnums::Kind& k);
ostream& operator<< (ostream& out, NNDataSetEnums::DataType& t);
ostream& operator<< (ostream& out, NNDataSetEnums::Sharding& s);



template<typename T> class NNDataSet : public NNDataSetBase {
public:
    friend class NNetwork;
    friend class NNLayer;
    friend vector<NNDataSetBase*> LoadNetCDF(const string& fname);
    friend bool SaveNetCDF(const string& fname, vector<NNDataSetBase*> vDataSet);

private:

    vector<T>               _vData;
    GpuBuffer<T>*           _pbData;
    vector<T>               _vSparseData;
    GpuBuffer<T>*           _pbSparseData;
    GpuBuffer<T>*           _pbSparseTransposedData;


    // Force constructor private
    NNDataSet(const string& fname, uint32_t n);
    bool Rename(const string& name);
    bool SaveNetCDF(const string& fname);
    bool WriteNetCDF(netCDF::NcFile& nfc, const string& fname, const uint32_t n);
    void RefreshState(uint32_t batch) {}    
    bool Shard(NNDataSetEnums::Sharding sharding);
    bool UnShard();
    vector<tuple<uint64_t, uint64_t> > getMemoryUsage();
    bool CalculateSparseDatapointCounts();
    bool GenerateSparseTransposedMatrix(uint32_t batch, NNLayer* pLayer);
    bool CalculateSparseTransposedMatrix(uint32_t position, uint32_t batch, NNLayer* pLayer);
    bool CalculateSparseTransposedDenoisedMatrix(uint32_t position, uint32_t batch, NNLayer* pLayer);
    bool CalculateSparseTransposedWeightGradient(NNFloat alpha, NNFloat beta, uint32_t m, uint32_t n, NNFloat* pDelta, NNFloat* pWeightGradient);     
    bool SetDenoising(bool flag);
    bool GenerateDenoisingData();
    bool LoadInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit);
    bool LoadSparseInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit);
    bool LoadSparseDenoisedInputUnit(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit);
    bool CalculateSparseZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, NNFloat* pUnit, NNFloat beta);
    bool CalculateSparseDenoisedZ(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pWeight, NNFloat* pUnit, NNFloat beta);
    float CalculateL1Error(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit);
    float CalculateL2Error(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit);
    float CalculateCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit);
    float CalculateScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit);
    float CalculateMultinomialCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit);
    float CalculateMultinomialScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit);
    bool CalculateL1OutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta);
    bool CalculateCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta);
    bool CalculateScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta);    
    bool CalculateOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta);
    float CalculateDataScaledMarginalCrossEntropyError(uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit);
    bool CalculateDataScaledMarginalCrossEntropyOutputDelta(Activation activation, uint32_t position, uint32_t batch, uint32_t stride, NNFloat* pUnit, NNFloat* pDelta);

public:

    ~NNDataSet();
    void Shuffle();
    T GetDataPoint(uint32_t n, uint32_t x, uint32_t y = 0, uint32_t z = 0);
    bool SetDataPoint(T v, uint32_t n, uint32_t x, uint32_t y = 0, uint32_t z = 0);
    uint32_t GetSparseDataPoints(uint32_t n);
    uint32_t GetSparseIndex(uint32_t n, uint32_t i);
    bool SetSparseIndex(uint32_t n, uint32_t i, uint32_t v);
    T GetSparseDataPoint(uint32_t n, uint32_t i);
    bool SetSparseDataPoint(uint32_t n, uint32_t i, T v);
};

vector<NNDataSetBase*> LoadNetCDF(const string& fname);
bool SaveNetCDF(const string& fname, vector<NNDataSetBase*> vDataset);
vector<NNDataSetBase*> LoadImageData(const string& fname);
vector<NNDataSetBase*> LoadCSVData(const string& fname);
vector<NNDataSetBase*> LoadJSONData(const string& fname);
vector<NNDataSetBase*> LoadAudioData(const string& name);

#endif
