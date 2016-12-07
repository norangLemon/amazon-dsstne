/*


   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#include "GpuTypes.h"
#include "NNTypes.h"
#include "NNDataSet.h"
#include "kernels.h"

using namespace std;
using namespace netCDF;
using namespace netCDF::exceptions;


static std::pair<TrainingMode, string> sTrainingModePair[] =
{
    std::pair<TrainingMode, string>(TrainingMode::SGD,      "SGD"),
    std::pair<TrainingMode, string>(TrainingMode::Momentum, "Momentum"),
    std::pair<TrainingMode, string>(TrainingMode::AdaGrad,  "AdaGrad"),
    std::pair<TrainingMode, string>(TrainingMode::Nesterov, "Nesterov"),
    std::pair<TrainingMode, string>(TrainingMode::RMSProp,  "RMSProp"),
    std::pair<TrainingMode, string>(TrainingMode::AdaDelta, "AdaDelta"),  
};

static std::map<TrainingMode, string> sTrainingModeMap =
std::map<TrainingMode, string>(sTrainingModePair, sTrainingModePair + sizeof(sTrainingModePair) / sizeof(sTrainingModePair[0]));

ostream& operator<< (ostream& out, const TrainingMode& e)
{
    out << sTrainingModeMap[e];
    return out;
}

static std::pair<ErrorFunction, string> sErrorFunctionPair[] =
{
    std::pair<ErrorFunction, string>(ErrorFunction::L1,                             "L1"),
    std::pair<ErrorFunction, string>(ErrorFunction::L2,                             "L2"),
    std::pair<ErrorFunction, string>(ErrorFunction::CrossEntropy,                   "CrossEntropy"),
    std::pair<ErrorFunction, string>(ErrorFunction::ScaledMarginalCrossEntropy,     "ScaledMarginalCrossEntropy")
};

static std::map<ErrorFunction, string> sErrorFunctionMap =
std::map<ErrorFunction, string>(sErrorFunctionPair, sErrorFunctionPair + sizeof(sErrorFunctionPair) / sizeof(sErrorFunctionPair[0]));

ostream& operator<< (ostream& out, const ErrorFunction& e)
{
    out << sErrorFunctionMap[e];
    return out;
}



static std::pair<Activation, string> sActivationPair[] =
{
    std::pair<Activation, string>(Activation::Sigmoid,                              "Sigmoid"),
    std::pair<Activation, string>(Activation::Tanh,                                 "Tanh"),
    std::pair<Activation, string>(Activation::RectifiedLinear,                      "RectifiedLinear"),
    std::pair<Activation, string>(Activation::Linear,                               "Linear"),
    std::pair<Activation, string>(Activation::ParametricRectifiedLinear,            "ParametricRectifiedLinear"),
    std::pair<Activation, string>(Activation::SoftSign,                             "SoftSign"),
    std::pair<Activation, string>(Activation::SoftPlus,                             "SoftPlus"),
    std::pair<Activation, string>(Activation::SoftMax,                              "SoftMax"),
    std::pair<Activation, string>(Activation::ReluMax,                              "ReluMax"),
    std::pair<Activation, string>(Activation::LinearMax,                            "LinearMax"),
};

static std::map<Activation, string> sActivationMap =
std::map<Activation, string>(sActivationPair, sActivationPair + sizeof(sActivationPair) / sizeof(sActivationPair[0]));


ostream& operator<< (ostream& out, const Activation& a)
{
    out << sActivationMap[a];
    return out;
}

static std::pair<WeightInitialization, string> sWeightInitializationPair[] =
{
    std::pair<WeightInitialization, string>(WeightInitialization::Xavier,           "Xavier"),
    std::pair<WeightInitialization, string>(WeightInitialization::CaffeXavier,      "CaffeXavier"),
    std::pair<WeightInitialization, string>(WeightInitialization::Gaussian,         "Gaussian"),
    std::pair<WeightInitialization, string>(WeightInitialization::Uniform,          "Uniform"),
    std::pair<WeightInitialization, string>(WeightInitialization::UnitBall,         "UnitBall"),
    std::pair<WeightInitialization, string>(WeightInitialization::Constant,         "Constant"),
};
static std::map<WeightInitialization, string> sWeightInitializationMap =
std::map<WeightInitialization, string>(sWeightInitializationPair, sWeightInitializationPair + sizeof(sWeightInitializationPair) / 
sizeof(sWeightInitializationPair[0]));

ostream& operator<< (ostream& out, const WeightInitialization& w)
{
    out << sWeightInitializationMap[w];
    return out;
}

static std::pair<PoolingFunction, string> sPoolingFunctionPair[] =
{
    std::pair<PoolingFunction, string>(PoolingFunction::None,                       "None"),
    std::pair<PoolingFunction, string>(PoolingFunction::Max,                        "Max"),
    std::pair<PoolingFunction, string>(PoolingFunction::Average,                    "Average"),
    std::pair<PoolingFunction, string>(PoolingFunction::Maxout,                     "Maxout"),
    std::pair<PoolingFunction, string>(PoolingFunction::Stochastic,                 "Stochastic"),
    std::pair<PoolingFunction, string>(PoolingFunction::LCN,                        "LocalContrastNormalization"),
    std::pair<PoolingFunction, string>(PoolingFunction::LRN,                        "LocalResponseNormalization"),
    std::pair<PoolingFunction, string>(PoolingFunction::GlobalTemporal,             "GlobalTemporal"),
};

static std::map<PoolingFunction, string> sPoolingFunctionMap =
std::map<PoolingFunction, string>(sPoolingFunctionPair, sPoolingFunctionPair + sizeof(sPoolingFunctionPair) / sizeof(sPoolingFunctionPair[0]));


ostream& operator<< (ostream& out, const PoolingFunction& a)
{
    out << sPoolingFunctionMap[a];
    return out;
}

ostream& operator<< (ostream& out, PoolingFunction& p);


static std::pair<NNDataSetEnums::Kind, string> sKindPair[] =
{
    std::pair<NNDataSetEnums::Kind, string>(NNDataSetEnums::Numeric, "Numeric"),
    std::pair<NNDataSetEnums::Kind, string>(NNDataSetEnums::Image,   "Image"),
    std::pair<NNDataSetEnums::Kind, string>(NNDataSetEnums::Audio,   "Audio")
};

static std::map<NNDataSetEnums::Kind, string> sKindMap =
std::map<NNDataSetEnums::Kind, string>(sKindPair, sKindPair + sizeof(sKindPair) / sizeof(sKindPair[0]));

ostream& operator<< (ostream& out, NNDataSetEnums::Kind& k)
{
    out << sKindMap[k];
    return out;
}



static std::pair<NNDataSetEnums::Attributes, string> sAttributesPair[] =
{
    std::pair<NNDataSetEnums::Attributes, string>(NNDataSetEnums::Sparse,                        "Sparse"),
    std::pair<NNDataSetEnums::Attributes, string>(NNDataSetEnums::Boolean,                       "Boolean"),
    std::pair<NNDataSetEnums::Attributes, string>(NNDataSetEnums::Compressed,                    "Compressed"),
    std::pair<NNDataSetEnums::Attributes, string>(NNDataSetEnums::Recurrent,                     "Recurrent"),
    std::pair<NNDataSetEnums::Attributes, string>(NNDataSetEnums::Mutable,                       "Mutable"),
    std::pair<NNDataSetEnums::Attributes, string>(NNDataSetEnums::Attributes::SparseIgnoreZero,   "SparseIgnoreZero"),
    std::pair<NNDataSetEnums::Attributes, string>(NNDataSetEnums::Attributes::Streaming,          "Streaming"),        

};

static std::map<NNDataSetEnums::Attributes, string> sAttributesMap =
std::map<NNDataSetEnums::Attributes, string>(sAttributesPair, sAttributesPair + sizeof(sAttributesPair) / sizeof(sAttributesPair[0]));


ostream& operator<< (ostream& out, NNDataSetEnums::Attributes& a)
{
    out << sAttributesMap[a];
    return out;
}


static std::pair<NNDataSetEnums::Sharding, string> sShardingPair[] =
{
    std::pair<NNDataSetEnums::Sharding, string>(NNDataSetEnums::None,  "None"),
    std::pair<NNDataSetEnums::Sharding, string>(NNDataSetEnums::Model, "Model"),
    std::pair<NNDataSetEnums::Sharding, string>(NNDataSetEnums::Data,  "Data")
};

static std::map<NNDataSetEnums::Sharding, string> sShardingMap =
std::map<NNDataSetEnums::Sharding, string>(sShardingPair, sShardingPair + sizeof(sShardingPair) / sizeof(sShardingPair[0]));

ostream& operator<< (ostream& out, NNDataSetEnums::Sharding& s)
{
    out << sShardingMap[s];
    return out;
}

static std::pair<NNDataSetEnums::DataType, string> sDataTypePair[] =
{
    std::pair<NNDataSetEnums::DataType, string>(NNDataSetEnums::UInt,   "UInt"),
    std::pair<NNDataSetEnums::DataType, string>(NNDataSetEnums::Int,    "Int"),
    std::pair<NNDataSetEnums::DataType, string>(NNDataSetEnums::LLInt,  "LLInt"),
    std::pair<NNDataSetEnums::DataType, string>(NNDataSetEnums::ULLInt, "ULLInt"),
    std::pair<NNDataSetEnums::DataType, string>(NNDataSetEnums::Float,  "Float"),
    std::pair<NNDataSetEnums::DataType, string>(NNDataSetEnums::Double, "Double"),
    std::pair<NNDataSetEnums::DataType, string>(NNDataSetEnums::RGB8,  "RGB8"),
    std::pair<NNDataSetEnums::DataType, string>(NNDataSetEnums::RGB16, "RGB16"),
    std::pair<NNDataSetEnums::DataType, string>(NNDataSetEnums::UChar,  "UChar"),
    std::pair<NNDataSetEnums::DataType, string>(NNDataSetEnums::Char,   "Char"),
};

static std::map<NNDataSetEnums::DataType, string> sDataTypeMap =
std::map<NNDataSetEnums::DataType, string>(sDataTypePair, sDataTypePair + sizeof(sDataTypePair) / sizeof(sDataTypePair[0]));


ostream& operator<< (ostream& out, NNDataSetEnums::DataType& t)
{
    out << sDataTypeMap[t];
    return out;
}

static inline bool has_suffix(const std::string &str, const std::string &suffix)
{
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

int MPI_Bcast_string(string& s)
{
    int length                          = s.size();
    MPI_Bcast(&length, 1, MPI_INT, 0, MPI_COMM_WORLD); 
    char buff[length + 1];
    strcpy(buff, s.c_str());
    int result                          = MPI_Bcast(&buff, length, MPI_CHAR, 0, MPI_COMM_WORLD); 
    buff[length]                        = '\0';  
    s                                   = buff;    
    return result;
}

NNDataSetBase::NNDataSetBase() :
_name(""),
_attributes(0),
_examples(0),
_dimensions(0),
_width(0),
_height(0),
_length(0),
_stride(0),
_sharding(NNDataSetEnums::None),
_minX(0),
_maxX(0),
_sparseDataSize(0),
_sparseTransposedIndices(0),
_maxSparseDatapoints(0),
_sparseDensity(0),
_bDenoising(false),
_pbSparseStart(NULL),
_pbSparseEnd(NULL),
_pbSparseIndex(NULL),
_pbSparseTransposedStart(NULL),
_pbSparseTransposedEnd(NULL),
_pbSparseTransposedIndex(NULL),
_batch(0),
_pbDenoisingRandom(NULL),
_bDirty(true)
{


}

NNDataSetBase::~NNDataSetBase() {}

NNDataSetDimensions NNDataSetBase::GetDimensions()
{
    NNDataSetDimensions dim;
    dim._dimensions                             = _dimensions;
    dim._width                                  = _width;
    dim._height                                 = _height;
    dim._length                                 = _length;
    return dim;
}

bool SaveNetCDF(const string& fname, vector<NNDataSetBase*> vDataSet)
{
    bool bResult                            = true;

    // Unshard data back to process 0 if necessary
    vector<NNDataSetEnums::Sharding> vSharding(vDataSet.size());
    for (uint32_t i = 0; i < vDataSet.size(); i++)
    {
        vSharding[i]                        = vDataSet[i]->_sharding;
        vDataSet[i]->UnShard();
    }

    // Now save data entirely from process 0
    if (getGpu()._id == 0)
    {
        bool bOpened                        = false;
        try
        {
            NcFile nfc(fname, NcFile::replace);
            bOpened                         = true;


            NcGroupAtt datasetsAtt          = nfc.putAtt("datasets", ncUint, (unsigned int)vDataSet.size());
            if (datasetsAtt.isNull())
            {
                throw NcException("NcException", "SaveNetCDF: Unable to write datasets attribute to NetCDF file " + fname, __FILE__, __LINE__);
            }
            for (uint32_t i = 0; i < vDataSet.size(); i++)
            {
                bool bResult                = vDataSet[i]->WriteNetCDF(nfc, fname, i);
                if (!bResult)
                    throw NcException("NcException", "SaveNetCDF: Unable to write dataset to NetCDF file " + fname, __FILE__, __LINE__);
            }
        }
        catch (NcException& e)
        {
            if (!bOpened)
            {
                cout << "SaveNetCDF: Unable to create NetCDF output file " << fname << endl;
            }
            else
            {
                cout << e.what() << endl;
            }
            bResult                         = false;
        }
    }

    // Gather and test on result
    MPI_Bcast(&bResult, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    if (!bResult)
    {
        getGpu().Shutdown();
        exit(-1);
    }

    // Restore original sharding
    for (uint32_t i = 0; i < vDataSet.size(); i++)
    {
        vDataSet[i]->Shard(vSharding[i]);
    }

    return bResult;
}

vector<NNDataSetBase*> LoadNetCDF(const string& fname) 
{
    vector<NNDataSetBase*> vDataSet;
    vector<NNDataSetEnums::DataType> vDataType;
    bool bResult                                = true;

    if (getGpu()._id == 0)
    {
        bool bOpened                            = false;
        try
        {
            NcFile rnc(fname.c_str(), NcFile::read);
            bOpened                             = true;

            // Determine # of data sets
            NcGroupAtt dataSetsAtt              = rnc.getAtt("datasets");
            if (dataSetsAtt.isNull())
            {
                throw NcException("NcException", "LoadNetCDF: No datasets count supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            uint32_t datasets;
            dataSetsAtt.getValues(&datasets);
            
            for (uint32_t i = 0; i < datasets; i++)
            {
                string nstring                  = std::to_string(i);
                string vname                    = "dataType" + nstring;
                NcGroupAtt dataTypeAtt          = rnc.getAtt(vname);
                if (dataTypeAtt.isNull())
                {
                      throw NcException("NcException", "LoadNetCDF: No " + vname + " attribute located in NetCDF input file " + fname, __FILE__, __LINE__); 
                }
                uint32_t dataType;
                dataTypeAtt.getValues(&dataType);   
                switch (dataType)
                {
                    case NNDataSetEnums::UInt:
                    case NNDataSetEnums::Int:
                    case NNDataSetEnums::LLInt:
                    case NNDataSetEnums::ULLInt:
                    case NNDataSetEnums::Float:
                    case NNDataSetEnums::Double:
                    case NNDataSetEnums::RGB8:
                    case NNDataSetEnums::RGB16:
                    case NNDataSetEnums::UChar:
                    case NNDataSetEnums::Char:
                        vDataType.push_back((NNDataSetEnums::DataType)dataType);
                        break;
                        
                    default:
                        printf("LoadNetCDF: Invalid data type in binary input file %s.\n", fname.c_str());                       
                }     
            }        
        }
        catch (NcException& e)
        {
            if (!bOpened)
            {
                cout << "NcException: LoadNetCDF: Error opening NetCDF input file " << fname << endl;
            }
            else
            {
                cout << "Exception: " << e.what() << endl;
            }
            bResult                         = false;
        }
    }

    // Gather and test on result
    MPI_Bcast(&bResult, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    if (!bResult)
    {
        getGpu().Shutdown();
        exit(-1);
    }

    uint32_t size                           = vDataType.size();
    MPI_Bcast(&size, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    vDataType.resize(size);
    MPI_Bcast(vDataType.data(), size, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    
    // Read data sets into vDataSet
    for (int i = 0; i < vDataType.size(); i++)
    {

        NNDataSetBase* pDataSet             = NULL;
        if (getGpu()._id == 0)
            cout << "LoadNetCDF: Loading " << vDataType[i] << " data set" << endl; 
        switch (vDataType[i])
        {
            case NNDataSetEnums::UInt:
                pDataSet                    = new NNDataSet<uint32_t>(fname, i);
                break;

            case NNDataSetEnums::Int:
                pDataSet                    = new NNDataSet<long>(fname, i);
                break;

            case NNDataSetEnums::Float:
                pDataSet                    = new NNDataSet<float>(fname, i);
                break;

            case NNDataSetEnums::Double:
                pDataSet                    = new NNDataSet<double>(fname, i);
                break;

            case NNDataSetEnums::Char:
                pDataSet                    = new NNDataSet<char>(fname, i);
                break;

            case NNDataSetEnums::UChar:
            case NNDataSetEnums::RGB8:
                pDataSet                    = new NNDataSet<uint8_t>(fname, i);
                break;

            default:
                printf("LoadNetCDF: invalid dataset type in binary input file %s.\n", fname.c_str());
                getGpu().Shutdown();
                exit(-1);
        }
        vDataSet.push_back(pDataSet);
    }

    return vDataSet;
}
vector<NNDataSetBase*> LoadImageData(const string& fname) {}
vector<NNDataSetBase*> LoadCSVData(const string& fname) {}
vector<NNDataSetBase*> LoadJSONData(const string& fname) {}
vector<NNDataSetBase*> LoadAudioData(const string& name) {}
