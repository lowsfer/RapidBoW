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