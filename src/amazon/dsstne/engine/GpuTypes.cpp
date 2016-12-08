/*


   Copyright 2016  Amazon.com, Inc. or its affiliates. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at

   http://aws.amazon.com/apache2.0/

   or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
 */

#include "GpuTypes.h"

#include <vector>

#include "GpuBuffer.h"

using std::vector;

void verifySGEMM(GpuBuffer<NNFloat>* pbA, GpuBuffer<NNFloat>* pbB, GpuBuffer<NNFloat>* pbC, uint32_t m, uint32_t k, uint32_t n)
{

    vector<NNFloat> vA(m * k);
    vector<NNFloat> vB(k * n);
    vector<NNFloat> vC(m * n);
    
    pbA->Download(vA.data());
    pbB->Download(vB.data());
    pbC->Download(vC.data());
    
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            NNFloat sum         = (NNFloat)0.0;
            NNFloat* pA         = vA.data() + i * k;            
            NNFloat* pB         = vB.data() + j;
            for (size_t kk = 0; kk < k; kk++)
            {
                sum            += *pA * (*pB);
                pA++;
                pB             += n;
            }
            if (fabs(sum - vC[i * n + j]) > 0.000001f)
                printf("%3lu %3lu %16.8f %16.8f\n", i, j, sum, vC[i * n + j]);
        }
    }
    exit(-1);
}

void verifySGEMMNT(GpuBuffer<NNFloat>* pbA, GpuBuffer<NNFloat>* pbB, GpuBuffer<NNFloat>* pbC, uint32_t m, uint32_t k, uint32_t n)
{

    vector<NNFloat> vA(m * k);
    vector<NNFloat> vB(k * n);
    vector<NNFloat> vC(m * n);
    
    pbA->Download(vA.data());
    pbB->Download(vB.data());
    pbC->Download(vC.data());

    
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            NNFloat sum         = (NNFloat)0.0;
            NNFloat* pA         = vA.data() + i * k;            
            NNFloat* pB         = vB.data() + j * k;
            for (size_t kk = 0; kk < k; kk++)
            {
                sum            += *pA * (*pB);
                pA++;
                pB++;
            }
            if (fabs(sum - vC[i * n + j]) / (fabs(sum) + 0.00000000000001f) > 0.000002f)
                printf("%3lu %3lu %16.8f %16.8f\n", i, j, sum, vC[i * n + j]);
        }
    }
    printf("%u %u %u\n", m, k, n);
    exit(-1);
}

void verifySGEMMTN(GpuBuffer<NNFloat>* pbA, GpuBuffer<NNFloat>* pbB, GpuBuffer<NNFloat>* pbC, uint32_t m, uint32_t k, uint32_t n)
{
    printf("%u %u %u\n", m, k, n);  
    vector<NNFloat> vA(m * k);
    vector<NNFloat> vB(k * n);
    vector<NNFloat> vC(m * n);
    
    pbA->Download(vA.data());
    pbB->Download(vB.data());
    pbC->Download(vC.data());
    
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            NNFloat sum         = (NNFloat)0.0;
            NNFloat* pA         = vA.data() + i;            
            NNFloat* pB         = vB.data() + j;
            for (size_t kk = 0; kk < k; kk++)
            {
                sum            += *pA * (*pB);
                pA             += m;
                pB             += n;
            }
            if (fabs(sum - vC[i * n + j]) / (fabs(sum) + 0.00000000000001f) > 0.000005f)
                printf("%3lu %3lu %16.8f %16.8f\n", i, j, sum, vC[i * n + j]);
        }
    }
    printf("%u %u %u\n", m, k, n);    
    exit(-1);    
}
