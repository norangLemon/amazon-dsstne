/*
   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#include "NNLayerDescriptor.h"

#include <netcdf>

using namespace netCDF;
using namespace netCDF::exceptions;

NNLayerDescriptor::NNLayerDescriptor() :
_kind(NNLayer::Kind::Hidden),
_type(NNLayer::Type::FullyConnected),
_poolingFunction(None),
_Nx(1),
_Ny(1),
_Nz(1),
_Nw(1),
_dimensions(1),
_bDimensionsProvided(true),
_weightInit(Xavier),
_weightInitScale((NNFloat)1.0),
_biasInit((NNFloat)0.0),
_kernelX(1),
_kernelY(1),
_kernelZ(1),
_kernelStrideX(1),
_kernelStrideY(1),
_kernelStrideZ(1),
_kernelPaddingX(0),
_kernelPaddingY(0),
_kernelPaddingZ(0),
_kernelDimensions(1),
_weightNorm((NNFloat)0.0),
_deltaNorm((NNFloat)0.0),
_pDropout((NNFloat)0.0),
_activation(Activation::Sigmoid),
_sparsenessPenalty_p((NNFloat)0.0),
_sparsenessPenalty_beta((NNFloat)0.0),
_attributes(NNLayer::Attributes::None)
{
}

bool LoadNNLayerDescriptorNetCDF(const string& fname, netCDF::NcFile& nc, uint32_t index, NNLayerDescriptor& ld)
{
    bool bResult                                = true;

    if (getGpu()._id == 0)
    {
        try {
            string lstring                      = "layer" + std::to_string(index) + "_";
            NcGroupAtt nameAtt                  = nc.getAtt(lstring + "name");
            if (nameAtt.isNull())
            {
                throw NcException("NcException", "NNLayer::NNLayer: No name supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            nameAtt.getValues(ld._name);

            NcGroupAtt kindAtt                  = nc.getAtt(lstring + "kind");
            if (kindAtt.isNull())
            {
                throw NcException("NcException", "NNLayer::NNLayer: No kind supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            kindAtt.getValues(&ld._kind);

            NcGroupAtt typeAtt                  = nc.getAtt(lstring + "type");
            if (typeAtt.isNull())
            {
                throw NcException("NcException", "NNLayer::NNLayer: No type supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            typeAtt.getValues(&ld._type);

            NcGroupAtt poolingFunctionAtt       = nc.getAtt(lstring + "poolingfunction");
            if (poolingFunctionAtt.isNull())
            {
                if (ld._type == NNLayer::Type::Pooling)
                    throw NcException("NcException", "NNLayer::NNLayer: No pooling function supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                ld._poolingFunction             = None;
            }
            else
                poolingFunctionAtt.getValues(&ld._poolingFunction);

            NcGroupAtt dataSetAtt               = nc.getAtt(lstring + "dataSet");
            if (dataSetAtt.isNull())
            {
                throw NcException("NcException", "NNLayer::NNLayer: No dataSet supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            dataSetAtt.getValues(ld._dataSet);

            NcGroupAtt NxAtt                    = nc.getAtt(lstring + "Nx");
            if (NxAtt.isNull())
            {
                throw NcException("NcException", "NNLayer::NNLayer: No Nx supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            NxAtt.getValues(&ld._Nx);

            NcGroupAtt NyAtt                    = nc.getAtt(lstring + "Ny");
            if (NyAtt.isNull())
            {
                throw NcException("NcException", "NNLayer::NNLayer: No Ny supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            NyAtt.getValues(&ld._Ny);

            NcGroupAtt NzAtt                    = nc.getAtt(lstring + "Nz");
            if (NzAtt.isNull())
            {
                throw NcException("NcException", "NNLayer::NNLayer: No Nz supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            NzAtt.getValues(&ld._Nz);

            NcGroupAtt NwAtt                    = nc.getAtt(lstring + "Nw");
            if (NwAtt.isNull())
            {
                throw NcException("NcException", "NNLayer::NNLayer: No Nw supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            NwAtt.getValues(&ld._Nw);

            NcGroupAtt dimensionsAtt            = nc.getAtt(lstring + "dimensions");
            if (dimensionsAtt.isNull())
            {
                throw NcException("NcException", "NNLayer::NNLayer: No dimensions supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            dimensionsAtt.getValues(&ld._dimensions);

            NcGroupAtt kernelXAtt               = nc.getAtt(lstring + "kernelX");
            if (kernelXAtt.isNull())
            {
                throw NcException("NcException", "NNLayer::NNLayer: No kernelX supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            kernelXAtt.getValues(&ld._kernelX);

            NcGroupAtt kernelYAtt               = nc.getAtt(lstring + "kernelY");
            if (kernelYAtt.isNull())
            {
                throw NcException("NcException", "NNLayer::NNLayer: No kernelY supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            kernelYAtt.getValues(&ld._kernelY);

            NcGroupAtt kernelZAtt               = nc.getAtt(lstring + "kernelZ");
            if (kernelZAtt.isNull())
            {
                throw NcException("NcException", "NNLayer::NNLayer: No kernelZ supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            kernelZAtt.getValues(&ld._kernelZ);

            NcGroupAtt kernelStrideXAtt         = nc.getAtt(lstring + "kernelStrideX");
            if (kernelStrideXAtt.isNull())
            {
                throw NcException("NcException", "NNLayer::NNLayer: No kernelStrideX supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            kernelStrideXAtt.getValues(&ld._kernelStrideX);

            NcGroupAtt kernelStrideYAtt         = nc.getAtt(lstring + "kernelStrideY");
            if (kernelStrideYAtt.isNull())
            {
                throw NcException("NcException", "NNLayer::NNLayer: No kernelStrideY supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            kernelStrideYAtt.getValues(&ld._kernelStrideY);

            NcGroupAtt kernelStrideZAtt         = nc.getAtt(lstring + "kernelStrideZ");
            if (kernelStrideZAtt.isNull())
            {
                throw NcException("NcException", "NNLayer::NNLayer: No kernelStrideZ supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            kernelStrideZAtt.getValues(&ld._kernelStrideZ);


            NcGroupAtt kernelPaddingXAtt        = nc.getAtt(lstring + "kernelPaddingX");
            if (kernelPaddingXAtt.isNull())
            {
                throw NcException("NcException", "NNLayer::NNLayer: No kernelPaddingX supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            kernelPaddingXAtt.getValues(&ld._kernelPaddingX);

            NcGroupAtt kernelPaddingYAtt        = nc.getAtt(lstring + "kernelPaddingY");
            if (kernelPaddingYAtt.isNull())
            {
                throw NcException("NcException", "NNLayer::NNLayer: No kernelPaddingY supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            kernelPaddingYAtt.getValues(&ld._kernelPaddingY);

            NcGroupAtt kernelPaddingZAtt        = nc.getAtt(lstring + "kernelPaddingZ");
            if (kernelPaddingZAtt.isNull())
            {
                throw NcException("NcException", "NNLayer::NNLayer: No kernelPaddingZ supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            kernelPaddingZAtt.getValues(&ld._kernelPaddingZ);

            NcGroupAtt kernelDimensionsAtt      = nc.getAtt(lstring + "kernelDimensions");
            if (kernelDimensionsAtt.isNull())
            {
                throw NcException("NcException", "NNLayer::NNLayer: No kernelDimensions supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            kernelDimensionsAtt.getValues(&ld._kernelDimensions);

            NcGroupAtt weightInitAtt            = nc.getAtt(lstring + "weightInit");
            if (weightInitAtt.isNull())
            {
                throw NcException("NcException", "NNLayer::NNLayer: No weightInit supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                ld._weightInit                  = Xavier;
            }
            else
                weightInitAtt.getValues(&ld._weightInit);

            NcGroupAtt weightInitScaleAtt       = nc.getAtt(lstring + "weightInitScale");
            if (weightInitScaleAtt.isNull())
            {
                throw NcException("NcException", "NNLayer::NNLayer: No weightInitScale supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                ld._weightInitScale             = (NNFloat)1.0;
            }
            else
                weightInitScaleAtt.getValues(&ld._weightInitScale);

            NcGroupAtt biasInitAtt              = nc.getAtt(lstring + "biasInit");
            if (biasInitAtt.isNull())
            {
                throw NcException("NcException", "NNLayer::NNLayer: No biasInit supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                ld._biasInit                    = (NNFloat)0.0;
            }
            else
                biasInitAtt.getValues(&ld._biasInit);

            NcGroupAtt weightNormAtt            = nc.getAtt(lstring + "weightNorm");
            if (weightNormAtt.isNull())
            {
                throw NcException("NcException", "NNLayer::NNLayer: No weightNorm supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                ld._weightNorm                  = (NNFloat)0.0;
            }
            else
                weightNormAtt.getValues(&ld._weightNorm);

            NcGroupAtt deltaNormAtt             = nc.getAtt(lstring + "deltaNorm");
            if (deltaNormAtt.isNull())
            {
                throw NcException("NcException", "NNLayer::NNLayer: No deltaNorm supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                ld._deltaNorm                   = (NNFloat)0.0;
            }
            else
                deltaNormAtt.getValues(&ld._deltaNorm);

            NcGroupAtt pDropoutAtt              = nc.getAtt(lstring + "pDropout");
            if (pDropoutAtt.isNull())
            {
                throw NcException("NcException", "NNLayer::NNLayer: No pDropout supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                ld._pDropout                    = (NNFloat)0.0;
            }
            else
                pDropoutAtt.getValues(&ld._pDropout);

            NcGroupAtt activationAtt            = nc.getAtt(lstring + "activation");
            if (activationAtt.isNull())
            {
                throw NcException("NcException", "NNLayer::NNLayer: No activation supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            activationAtt.getValues(&ld._activation);

            // Added in version 0.81, supply default values here if not present, eventually throw exception
            NcGroupAtt sparsenessPenalty_pAtt   = nc.getAtt("sparsenessPenalty_p");
            if (sparsenessPenalty_pAtt.isNull())
            {
                throw NcException("NcException", "NNLayer::NNLayer: No sparsenessPenalty_p supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                ld._sparsenessPenalty_p = (NNFloat)0.0;
            }
            else
            {
                sparsenessPenalty_pAtt.getValues(&(ld._sparsenessPenalty_p));
            }

            // Added in version 0.81, supply default values here if not present, eventually throw exception
            NcGroupAtt sparsenessPenalty_betaAtt= nc.getAtt("sparsenessPenalty_beta");
            if (sparsenessPenalty_betaAtt.isNull())
            {
                throw NcException("NcException", "NNLayer::NNLayer: No sparsenessPenalty_beta supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                ld._sparsenessPenalty_p = (NNFloat)0.0;
            }
            else
            {
                sparsenessPenalty_betaAtt.getValues(&(ld._sparsenessPenalty_beta));
            }

            NcGroupAtt attributesAtt            = nc.getAtt(lstring + "attributes");
            if (attributesAtt.isNull())
            {
                throw NcException("NcException", "NNLayer::NNLayer: No attributes supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            attributesAtt.getValues(&ld._attributes);

            // Read sources
            uint32_t sources                    = 0;
            NcGroupAtt sourcesAtt               = nc.getAtt(lstring + "sources");
            if (sourcesAtt.isNull())
            {
                throw NcException("NcException", "NNLayer::NNLayer: No sources supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            sourcesAtt.getValues(&sources);

            for (uint32_t i = 0; i < sources; i++)
            {
                string nstring                  = std::to_string(i);
                NcGroupAtt sourceAtt            = nc.getAtt(lstring + "source" + nstring);
                if (sourcesAtt.isNull())
                {
                    throw NcException("NcException", "NNLayer::NNLayer: No source attributes supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                string source;
                sourceAtt.getValues(source);
                ld._vSource.push_back(source);
            }

            // Read skips
            uint32_t skips                      = 0;
            NcGroupAtt skipsAtt                 = nc.getAtt(lstring + "skips");
            if (skipsAtt.isNull())
            {
                throw NcException("NcException", "NNLayer::NNLayer: No skips supplied in NetCDF input file " + fname, __FILE__, __LINE__);
            }
            skipsAtt.getValues(&skips);

            for (uint32_t i = 0; i < skips; i++)
            {
                string nstring                  = std::to_string(i);
                NcGroupAtt skipAtt              = nc.getAtt(lstring + "skip" + nstring);
                if (skipAtt.isNull())
                {
                    throw NcException("NcException", "NNLayer::NNLayer: No skip attributes supplied in NetCDF input file " + fname, __FILE__, __LINE__);
                }
                string skip;
                skipAtt.getValues(skip);
                ld._vSkip.push_back(skip);
            }
        }
        catch (NcException& e)
        {
            cout << "Exception: " << e.what() << endl;
            bResult                             = false;
        }
    }

    return bResult;
}

ostream& operator<< (ostream& out, NNLayerDescriptor& d)
{
    out << "Name:                  " << d._name << endl;
    out << "Kind:                  " << d._kind << endl;
    out << "Type:                  " << d._type << endl;
    if (d._type != NNLayer::Type::Pooling)
        out << "Pooling Function:      " << d._poolingFunction << endl;
    out << "Nx:                    " << d._Nx << endl;
    out << "Ny:                    " << d._Ny << endl;
    out << "Nz:                    " << d._Nz << endl;
    out << "Nw:                    " << d._Nw << endl;
    if (d._type != NNLayer::Type::FullyConnected)
    {
        out << "kernelX:               " << d._kernelX << endl;
        out << "kernelY:               " << d._kernelY << endl;
        out << "kernelZ:               " << d._kernelZ << endl;
        out << "kernelStrideX:         " << d._kernelStrideX << endl;
        out << "kernelStrideY:         " << d._kernelStrideY << endl;
        out << "kernelStrideZ:         " << d._kernelStrideZ << endl;
        out << "kernelPaddingX:        " << d._kernelPaddingX << endl;
        out << "kernelPaddingY:        " << d._kernelPaddingY << endl;
        out << "kernelPaddingZ:        " << d._kernelPaddingZ << endl;
        out << "kernelDimensions:      " << d._kernelDimensions << endl;
    }
    if (d._type != NNLayer::Type::Pooling)
    {
        out << "pDropout:              " << d._pDropout << endl;
        out << "weightInit:            " << d._weightInit << endl;
        out << "weightInitScale:       " << d._weightInitScale << endl;
        out << "biasInit:              " << d._biasInit << endl;
        out << "weightNorm:            " << d._weightNorm << endl;
        out << "deltaNorm:             " << d._deltaNorm << endl;
        out << "activation:            " << d._activation << endl;
        out << "Sparse:                " << ((d._attributes & NNLayer::Attributes::Sparse) != 0) << endl;
        if (d._type == NNLayer::Type::FullyConnected)
        {
            if (d._sparsenessPenalty_p > (NNFloat)0.0)
                out << "sparsenessPenalty_p    " << d._sparsenessPenalty_p << endl;
            if (d._sparsenessPenalty_beta > (NNFloat)0.0)
                out << "sparsenessPenalty_beta " << d._sparsenessPenalty_beta << endl;
        }
        if (d._kind != NNLayer::Kind::Hidden)
            out << "DataSet:               " << d._dataSet << endl;
    }
    for (size_t i = 0 ; i < d._vSource.size(); i++)
    {
        out << "source " << setw(3) << i << ":            " << d._vSource[i] << endl;
    }
    for (size_t i = 0 ; i < d._vSkip.size(); i++)
    {
        out << "skip " << setw(3) << i << ":            " << d._vSkip[i] << endl;
    }
    return out;
}

uint32_t MPI_Bcast_NNLayerDescriptor(NNLayerDescriptor& d)
{
    MPI_Bcast_string(d._name);
    MPI_Bcast(&d._kind, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._type, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._poolingFunction, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._Nx, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._Ny, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._Nz, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._Nw, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._dimensions, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._bDimensionsProvided, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._kernelX, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._kernelY, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._kernelZ, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._kernelStrideX, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._kernelStrideY, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._kernelStrideZ, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._kernelPaddingX, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._kernelPaddingY, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._kernelPaddingZ, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._pDropout, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._weightInit, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._weightInitScale, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._biasInit, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._weightNorm, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._deltaNorm, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._activation, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._sparsenessPenalty_p, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._sparsenessPenalty_beta, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d._attributes, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    MPI_Bcast_string(d._dataSet);
    size_t size                         = d._vSource.size();
    MPI_Bcast(&size, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    d._vSource.resize(size);
    for (size_t i = 0; i < size; i++)
        MPI_Bcast_string(d._vSource[i]);
    size                                = d._vSkip.size();
    MPI_Bcast(&size, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    d._vSkip.resize(size);
    for (size_t i = 0; i < size; i++)
        MPI_Bcast_string(d._vSkip[i]);
    return 0;
}
