/*
   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#include "NNWeightDescriptor.h"

#include <netcdf>

#include "NNTypes.h"

using namespace netCDF;
using namespace netCDF::exceptions;

NNWeightDescriptor::NNWeightDescriptor() :
_width(1),
_height(1),
_length(1),
_breadth(1),
_depth(1),
_bShared(false),
_bTransposed(false),
_bLocked(false),
_norm((NNFloat)0.0)
{
}

bool LoadNNWeightDescriptorNetCDF(const string& fname, netCDF::NcFile& nc, uint32_t index, NNWeightDescriptor& wd)
{
    bool bResult                                = true;

    if (getGpu()._id == 0)
    {
        string wstring                          = "weight" + std::to_string(index) + "_";
        try {
            NcGroupAtt inputLayerAtt            = nc.getAtt(wstring + "inputLayer");
            if (inputLayerAtt.isNull())
            {
                throw NcException("NcException", "No input layer supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            inputLayerAtt.getValues(wd._inputLayer);

            NcGroupAtt outputLayerAtt           = nc.getAtt(wstring + "outputLayer");
            if (outputLayerAtt.isNull())
            {
                throw NcException("NcException", "No output layer supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            outputLayerAtt.getValues(wd._outputLayer);

            NcGroupAtt normAtt                  = nc.getAtt(wstring + "norm");
            if (normAtt.isNull())
            {
                //throw NcException("NcException", "No norm supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                wd._norm                        = (NNFloat)0.0;
            }
            else
                normAtt.getValues(&wd._norm);

            NcGroupAtt bSharedAtt               = nc.getAtt(wstring + "bShared");
            if (bSharedAtt.isNull())
            {
                throw NcException("NcException", "No bShared supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            uint32_t bShared;
            bSharedAtt.getValues(&bShared);
            wd._bShared                         = (bShared != 0);

            // Read shared weight attributes if _bShared is true
            if (wd._bShared)
            {
                NcGroupAtt sourceInputLayerAtt  = nc.getAtt(wstring + "sourceInputLayer");
                if (sourceInputLayerAtt.isNull())
                {
                    throw NcException("NcException", "No sourceInputLayer for shared weights supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                sourceInputLayerAtt.getValues(wd._sourceInputLayer);
                NcGroupAtt sourceOutputLayerAtt = nc.getAtt(wstring + "sourceOutputLayer");
                if (sourceInputLayerAtt.isNull())
                {
                    throw NcException("NcException", "No sourceOutputLayer for shared weights supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                sourceOutputLayerAtt.getValues(wd._sourceOutputLayer);
                NcGroupAtt bTransposedAtt       = nc.getAtt(wstring + "bTransposed");
                if (bTransposedAtt.isNull())
                {
                    throw NcException("NcException", "No bTransposed for shared weights supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                uint32_t bTransposed;
                bTransposedAtt.getValues(&bTransposed);
                wd._bTransposed                 = (bTransposed != 0);
            }

            NcGroupAtt bLockedAtt               = nc.getAtt(wstring + "bLocked");
            if (bLockedAtt.isNull())
            {
                wd._bLocked                     = false;
            }
            else
            {
                uint32_t bLocked;
                bLockedAtt.getValues(&bLocked);
                wd._bLocked                     = (bLocked != 0);
            }

            NcGroupAtt widthAtt                 = nc.getAtt(wstring + "width");
            if (widthAtt.isNull())
            {
                throw NcException("NcException", "No weight width supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            widthAtt.getValues(&wd._width);

            NcGroupAtt heightAtt                = nc.getAtt(wstring + "height");
            if (heightAtt.isNull())
            {
                throw NcException("NcException", "No weight height supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            heightAtt.getValues(&wd._height);

            NcGroupAtt lengthAtt                = nc.getAtt(wstring + "length");
            if (lengthAtt.isNull())
            {
                throw NcException("NcException", "No weight length supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            lengthAtt.getValues(&wd._length);

            NcGroupAtt depthAtt                 = nc.getAtt(wstring + "depth");
            if (depthAtt.isNull())
            {
                throw NcException("NcException", "No weight depth supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            depthAtt.getValues(&wd._depth);

            NcGroupAtt breadthAtt               = nc.getAtt(wstring + "breadth");
            if (breadthAtt.isNull())
            {
                throw NcException("NcException", "No weight breadth supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            breadthAtt.getValues(&wd._breadth);

            // Read biases
            NcDim biasDim                       = nc.getDim(wstring + "biasDim");
            NcVar biasVar                       = nc.getVar(wstring + "bias");
            wd._vBias.resize(biasDim.getSize());
            biasVar.getVar(wd._vBias.data());

            if (!wd._bShared)
            {
                NcDim weightDim                 = nc.getDim(wstring + "weightDim");
                NcVar weightVar                 = nc.getVar(wstring + "weights");
                wd._vWeight.resize(weightDim.getSize());
                weightVar.getVar(wd._vWeight.data());
            }
#if 0
            printf("Weights %d %lu %lu\n", index, _vWeight.size(), _vBias.size());
            for (int i = 0; i < 20; i++)
                printf("%3d %16.8f %16.8f\n", i, _vWeight[i], _vBias[i]);
#endif
        }
        catch (NcException& e)
        {
            cout << "NNWeightDescriptor::NNWeightDescriptor: Exception: " << e.what() << endl;
            bResult                             = false;
        }

    }

    return bResult;
}

uint32_t MPI_Bcast_NNWeightDescriptor(NNWeightDescriptor& d)
{
    MPI_Bcast_string(d._inputLayer);
    MPI_Bcast_string(d._outputLayer);
    MPI_Bcast(&d._bShared, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._bTransposed, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._bLocked, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._norm, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast_string(d._sourceInputLayer);
    MPI_Bcast_string(d._sourceOutputLayer);
    MPI_Bcast(&d._width, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._height, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._length, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._depth, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._breadth, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    uint64_t weights                        = d._vWeight.size();
    MPI_Bcast(&weights, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    d._vWeight.resize(weights);
    MPI_Bcast(d._vWeight.data(), weights, MPI_FLOAT, 0, MPI_COMM_WORLD);
    uint64_t biases                         = d._vBias.size();
    MPI_Bcast(&biases, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    d._vBias.resize(biases);
    MPI_Bcast(d._vBias.data(), biases, MPI_FLOAT, 0, MPI_COMM_WORLD);
    return 0;
}

ostream& operator<< (ostream& out, NNWeightDescriptor& d)
{
    if (getGpu()._id == 0)
    {
        out << "Input Layer:        " << d._inputLayer << endl;
        out << "Output Layer:       " << d._outputLayer << endl;
        out << "Width               " << d._width << endl;
        out << "Height              " << d._height << endl;
        out << "Length              " << d._length << endl;
        out << "Depth               " << d._depth << endl;
        out << "Breadth             " << d._breadth << endl;
        out << "bShared:            " << std::boolalpha << d._bShared << endl;
        out << "bTransposed:        " << std::boolalpha << d._bTransposed << endl;
        if (d._bShared)
        {
            out << "sourceInputLayer:   " << d._sourceInputLayer << endl;
            out << "sourceOutputLayer:  " << d._sourceOutputLayer << endl;
        }
        out << "bLocked:            " << std::boolalpha << d._bLocked << endl;
        out << "norm:               " << d._norm << endl;
    }
    return out;
}
