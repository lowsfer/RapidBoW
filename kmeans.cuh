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
// Created by yao on 10/19/19.
//

#pragma once
#include <cstdint>
#include "KArray.h"
#include "BitSet.h"
#include <cuda_runtime_api.h>
#include <limits>
#include <numeric>
#include <cuda_utils.h>

constexpr uint32_t idxInvalid = std::numeric_limits<uint32_t>::max();

struct KMeansTraits{
    using DescElemType = uint8_t;
    static constexpr uint32_t elemsPerWord = sizeof(uint32_t) / sizeof(DescElemType);
    static constexpr uint32_t wordsPerVec = 1;
    static constexpr uint32_t vecsPerDesc = 32;
    static constexpr uint32_t warp_size = 32;
    static constexpr uint32_t wordsPerDesc = wordsPerVec * vecsPerDesc;
    static constexpr uint32_t descDims = wordsPerDesc * sizeof(uint32_t) / sizeof(DescElemType);

    static constexpr uint32_t thrdM = 4;
    static constexpr uint32_t nbCenters = 16;
    static constexpr uint32_t thrdN = nbCenters;

    static constexpr uint32_t grpSize = warp_size / thrdM;

    static constexpr uint32_t thrdLdsN = 4; // number of center involved per uniform smem load. One Vec per center

    static constexpr uint32_t nbSMemBanks = 32u;
    static constexpr uint32_t nbSMemCenterAcc = nbSMemBanks / nbCenters;

    using Vec = KArray<uint32_t, wordsPerVec>;
    static constexpr uint32_t tileSize = warp_size;
    using DescTile = KArray<Vec, vecsPerDesc, tileSize>; // A group of transposed descriptors
    using Descriptor = KArray<Vec, vecsPerDesc>;
    using Centers = KArray<Vec, vecsPerDesc, nbCenters>;

    using Distance = uint32_t;
#ifdef __CUDACC__
    __device__ __forceinline__
    static Distance accumulate(const Distance& init, const Vec& a, const Vec& b)
    {
        Distance acc = init;
        for (uint32_t i = 0; i < Vec::dimension; i++)
        {
            const uint32_t diff = __vabsdiffu4(a[i], b[i]);
            acc = __dp4a(diff, diff, acc);
        }
        return acc;
    }

    template <typename DstType = DescElemType> __device__ __forceinline__
    static DstType extractElem(const Vec& vec, uint32_t idxInVec){
        const uint32_t idxWord = idxInVec / elemsPerWord;
        const uint32_t idxInWord = idxInVec % elemsPerWord;
        const uint32_t result = __dp4a(vec[idxWord], 1u << (8 * idxInWord), 0u);
        return static_cast<DstType>(result);
    }
#endif
    using CtaAccType = uint32_t;
    using GlbAccType = uint32_t;
    using GlbCenterAcc = KArray<GlbAccType, descDims, nbCenters>;
    using Counters = KArray<uint32_t, nbCenters>;
};

struct alignas(128) ArgsKMeansAssignCenter
{
    using Traits = KMeansTraits;
    const Traits::DescTile* __restrict__ descTiles;
    uint32_t nbDescTiles;
    uint32_t nbDesc;
    const Traits::Centers* __restrict__ centers; // length = 1
    uint8_t* __restrict__ idxNearestCenter; // length = nbDescTiles * tileSize
    Traits::Counters* __restrict__ counter; // descriptors per center, length = 1
    Traits::GlbCenterAcc* __restrict__ centerAcc; // length = 1
    uint32_t* __restrict__ nbChanged; // length = 1
    const uint32_t* __restrict__ nbChagnedLast; // length = 1
    const BitSet<Traits::nbCenters>* __restrict__ validMask; // length = 1
};
//@info: ArgsKMeansAssignCenter does not have to be passed in. We could use SoA arguments and build ArgsKMeansAssignCenter in shared memory. Then arguments can be simplified. But that relies on the assumption that tasks are contiguous, and would not handle sparsity (i.e. remove tasks that are empty or converged early in the future).
void launchAssignCenter(const ArgsKMeansAssignCenter* __restrict__ tasks, uint32_t nbTasks, cudaStream_t stream);
void launchKMeansUpdateCenters(const ArgsKMeansAssignCenter* __restrict__ tasks, uint32_t nbTasks, cudaStream_t stream);
