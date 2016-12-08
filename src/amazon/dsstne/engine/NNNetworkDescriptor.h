/*
   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#ifndef NNNETWORK_DESCRIPTOR_H
#define NNNETWORK_DESCRIPTOR_H
#ifndef __NVCC__

#include <iostream>
#include <string>

#include "NNNetwork.h"

using std::ostream;
using std::string;

struct NNNetworkDescriptor
{
    string                      _name;                      // Optional name for neural network
    NNNetwork::Kind             _kind;                      // Either AutoEncoder or FeedForward (default)
    ErrorFunction               _errorFunction;             // Error function for training
    vector<NNLayerDescriptor>   _vLayerDescriptor;          // Vector containing neural network layers
    vector<NNWeightDescriptor>  _vWeightDescriptor;         // Vector containing preloaded weight data
    bool                        _bShuffleIndices;           // Flag to signal whether to shuffle training data or not
    uint32_t                    _maxout_k;                  // Size of Maxout (default 2)
    NNFloat                     _LRN_k;                     // Local Response Normalization offset (default 2)
    uint32_t                    _LRN_n;                     // Local Response Normalization spread (default 5)
    NNFloat                     _LRN_alpha;                 // Local Response Normalization scaling (default 0.0001)
    NNFloat                     _LRN_beta;                  // Local Response Normalization exponent (default 0.75)
    bool                        _bSparsenessPenalty;        // Specifies whether to use sparseness penalty on hidden layers or not
    NNFloat                     _sparsenessPenalty_p;       // Target sparseness probability for hidden layers
    NNFloat                     _sparsenessPenalty_beta;    // Sparseness penalty weight
    bool                        _bDenoising;                // Specifies whether to use denoising on input layers
    NNFloat                     _denoising_p;               // Probability of denoising inputs (for sparse layers, only denoise on non-zero values)
    NNFloat                     _deltaBoost_one;            // Adjusts scaling of nonzero-valued outputs
    NNFloat                     _deltaBoost_zero;           // Adjusts scaling of zero-valued outputs
    NNFloat                     _SMCE_oneTarget;            // Relaxed target for non-zero target values (Default 0.9)
    NNFloat                     _SMCE_zeroTarget;           // Relaxed target for zero target values (Default 0.1)
    NNFloat                     _SMCE_oneScale;             // Scaling factor for non-zero target values (Default 1.0)
    NNFloat                     _SMCE_zeroScale;            // Scaling factor for zero target values (Default 1.0)
    string                      _checkpoint_name;           // Checkpoint file name
    int32_t                     _checkpoint_interval;       // Number of epochs between checkpoints
    int32_t                     _checkpoint_epochs;         // Number of epochs since last checkpoint
    bool                        _bConvLayersCalculated;     // Have convolution layer dimensions been calculated?
    NNNetworkDescriptor();
};

ostream& operator<< (ostream& out, NNNetworkDescriptor& d);

#endif // __NVCC__
#endif // NNNETWORK_DESCRIPTOR_H
