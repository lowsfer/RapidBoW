/*
Copyright [2024] [Yao Yao]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

*/

//
// Created by yao on 10/25/19.
//

#pragma once
#include "kmeans.cuh"
namespace rbow {
void launchVocTreeLookUp(
        const KArray<KMeansTraits::Vec, KMeansTraits::vecsPerDesc, KMeansTraits::nbCenters> *__restrict__ nodes,
        const BitSet<KMeansTraits::nbCenters> *__restrict__ masks, const uint32_t nbLevels,
        const typename KMeansTraits::Descriptor *__restrict__ descriptors, const uint32_t nbDesc,
        uint32_t *__restrict__ indicesInLeafLevel, cudaStream_t stream);
}