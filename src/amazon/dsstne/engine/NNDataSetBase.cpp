/*
   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#include "NNDataSetBase.h"

#include "GpuContext.h"
#include "NNDataSet.h"

using namespace netCDF;
using namespace netCDF::exceptions;

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

uint32_t NNDataSetBase::GetExamples()
{
    return _examples;
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
