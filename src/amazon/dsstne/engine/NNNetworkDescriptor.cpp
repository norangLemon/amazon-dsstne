/*
   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#include "NNNetworkDescriptor.h"

#include <netcdf>

#include "NNLayerDescriptor.h"
#include "NNWeightDescriptor.h"

using namespace netCDF;
using namespace netCDF::exceptions;

NNNetworkDescriptor::NNNetworkDescriptor() :
_kind(NNNetwork::Kind::FeedForward),
_errorFunction(ErrorFunction::CrossEntropy),
_bShuffleIndices(true),
_maxout_k(2),
_LRN_k(2),
_LRN_n(5),
_LRN_alpha((NNFloat)0.0001),
_LRN_beta((NNFloat)0.75),
_bSparsenessPenalty(false),
_sparsenessPenalty_p((NNFloat)0.0),
_sparsenessPenalty_beta((NNFloat)0.0),
_bDenoising(false),
_denoising_p((NNFloat)0.0),
_deltaBoost_one((NNFloat)1.0),
_deltaBoost_zero((NNFloat)1.0),
_SMCE_oneTarget((NNFloat)0.9),
_SMCE_zeroTarget((NNFloat)0.1),
_SMCE_oneScale((NNFloat)1.0),
_SMCE_zeroScale((NNFloat)1.0),
_name(""),
_checkpoint_name("checkpoint"),
_checkpoint_interval(0),
_checkpoint_epochs(0),
_bConvLayersCalculated(false)
{
}

ostream& operator<< (ostream& out, NNNetworkDescriptor& d)
{
    out << "Name:                    " << d._name << endl;
    out << "Kind:                    " << d._kind << endl;
    out << "bShuffleIndices          " << std::boolalpha << d._bShuffleIndices << endl;
    out << "Error Function:          " << d._errorFunction << endl;
    out << "MaxOut_k:                " << d._maxout_k << endl;
    out << "LRN_k:                   " << d._LRN_k << endl;
    out << "LRN_n:                   " << d._LRN_n << endl;
    out << "LRN_beta:                " << d._LRN_beta << endl;
    out << "LRN_alpha:               " << d._LRN_alpha << endl;
    out << "bSparsenessPenalty:      " << std::boolalpha << d._bSparsenessPenalty << endl;
    out << "sparsenessPenalty_beta:  " << d._sparsenessPenalty_beta << endl;
    out << "sparsenessPenalty_p:     " << d._sparsenessPenalty_p << endl;
    out << "bDenoising:              " << std::boolalpha << d._bDenoising << endl;
    out << "denoising_p:             " << d._denoising_p << endl;
    out << "deltaBoost_one:          " << d._deltaBoost_one << endl;
    out << "deltaBoost_zero:         " << d._deltaBoost_zero << endl;
    out << "SMCE_oneTarget:          " << d._SMCE_oneTarget << endl;
    out << "SMCE_zeroTarget:         " << d._SMCE_zeroTarget << endl;
    out << "SMCE_oneScale:           " << d._SMCE_oneScale << endl;
    out << "SMCE_zeroScale:          " << d._SMCE_zeroScale << endl;
    out << "checkpoint_name:         " << d._checkpoint_name << endl;
    out << "checkpoint_interval:     " << d._checkpoint_interval << endl;

    // Dump layers
    out << endl << "Layers:" << endl;
    for (uint32_t i = 0; i < d._vLayerDescriptor.size(); i++)
    {
        out << "Layer " << i << endl << d._vLayerDescriptor[i];
    }

    // Dump Weights
    out << endl << "Weights:" << endl;
    for (uint32_t i = 0; i < d._vWeightDescriptor.size(); i++)
    {
        out << "Weight " << i << endl << d._vWeightDescriptor[i];
    }
    return out;
}
