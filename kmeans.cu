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
// Created by yao on 4/11/18.
//

#include "cuda_hint.cuh"
#include <cuda_runtime_api.h>
#include <array>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>
#include <cuda_utils.h>
#include "KArray.h"
#include "kmeans.cuh"
#include <mutex>
#include "cuda_utils.h"

namespace cg = cooperative_groups;

#define PREFETCH_FIRST_DESC_VEC 0 // seems this prefetch is not useful
class GrpTask : public KMeansTraits
{
public:
    __device__ __forceinline__
    GrpTask(const cg::thread_block_tile<warp_size / thrdM>& grp, uint32_t idxGrp, uint32_t nbGrps, const ArgsKMeansAssignCenter& args, const Vec (&centers)[vecsPerDesc][nbCenters], KArray<CtaAccType, nbSMemCenterAcc> (&ctaCenterAcc)[descDims][nbCenters], uint32_t(&ctaNbChanged)[warp_size])
    : mGrp{grp}
    , mThrdRank{grp.thread_rank()}
    , mIdxGrp{idxGrp}
    , mNbGrps{nbGrps}
    , mArgs{args}
    , mCenters{centers}
    , mCtaCenterAcc{ctaCenterAcc}
    , mCtaNbChanged{ctaNbChanged}
    {}

    //@fixme: add cache control template argument for default/last-use
    __device__ __forceinline__
    void loadDescVec(KArray<Vec, thrdM>& descVecs, const uint32_t idxDescTile, const uint32_t idxVec){
        assert(idxDescTile < mArgs.nbDescTiles);
        descVecs = reinterpret_cast<const KArray<Vec, thrdM>*>(&mArgs.descTiles[idxDescTile][idxVec])[mThrdRank];
    }

    __device__ __forceinline__
    void computeVec(const KArray<Vec, thrdM>& descVecs, const uint32_t idxVec)
    {
        static_assert(thrdN == nbCenters, "fatal error");
        KArray<Vec, thrdLdsN> vecCenter[2]; // double buffer
        auto loadCenterVec = [this, &vecCenter, idxVec](uint32_t idxSmemLd/* = idxCenter/thrdLdsN*/){
            vecCenter[idxSmemLd % 2] = reinterpret_cast<const KArray<Vec, thrdLdsN>&>(mCenters[idxVec][idxSmemLd * thrdLdsN]);
        };
        loadCenterVec(0);
#pragma unroll
        for (uint32_t idxSmemLd = 0; idxSmemLd < thrdN / thrdLdsN; idxSmemLd++) {
            static_assert(thrdN % thrdLdsN == 0, "fatal error");
            if (idxSmemLd + 1 < thrdN / thrdLdsN) {
                loadCenterVec(idxSmemLd + 1);
            }
#pragma unroll
            for (uint32_t m = 0; m < thrdM; m++) {
#pragma unroll
                for (uint32_t inner_n = 0; inner_n < thrdLdsN; inner_n++) {
                    const uint32_t n = thrdLdsN * idxSmemLd + inner_n;
                    mThrdAcc[m][n] = accumulate(mThrdAcc[m][n],
                                                descVecs[m], vecCenter[idxSmemLd % 2][inner_n]);
//                    if (threadIdx.x == 0 && m == 1 && n == 0){
//                        for (uint32_t i = 0; i < 4u; i++){
//                            printf("%u - %u\n", (uint32_t)reinterpret_cast<const uint8_t(&)[4]>(descVecs[m])[i],
//                                   (uint32_t)reinterpret_cast<const uint8_t(&)[4]>(vecCenter[idxSmemLd % 2][inner_n])[i]);
//                        }
//                    }
                }
            }
        }
    }
    __device__ __forceinline__
    void computeTile(uint32_t idxDescTile)
    {
        assert(idxDescTile < mArgs.nbDescTiles);
#if !PREFETCH_FIRST_DESC_VEC
        loadDescVec(mThrdDescVec[0], idxDescTile, 0);
#endif
        memset(&mThrdAcc, 0, sizeof(mThrdAcc));
#pragma unroll(1)
        for (uint32_t idxVecOuter = 0; idxVecOuter < vecsPerDesc / 2; idxVecOuter++){
#pragma unroll
            for (uint32_t idxBuf = 0; idxBuf < 2; idxBuf++){
                const uint32_t idxVec = idxVecOuter * 2 + idxBuf;
                const KArray<Vec, thrdM>& descVecs = mThrdDescVec[idxBuf];
                KArray<Vec, thrdM>& descVecsNext = mThrdDescVec[(idxBuf + 1) % 2];
                if (idxVec + 1 < vecsPerDesc){
                    loadDescVec(descVecsNext, idxDescTile, idxVec + 1);
                }
                computeVec(descVecs, idxVec);
            }
        }
#if PREFETCH_FIRST_DESC_VEC
        if (idxDescTile + 1 < mArgs.nbDescTiles){
            loadDescVec(mThrdDescVec[0], idxDescTile + 1, 0);
        }
#endif
        struct {
            uint32_t minDistance;
            uint32_t idxCenter;
        } nearestCenter[thrdM];
#pragma unroll
        for (uint32_t m = 0; m < thrdM; m++){
#pragma unroll
            for (uint32_t n = 0; n < thrdN; n++){
//                if (threadIdx.x == 0 && m == 1) {
//                    printf("\t%u", mThrdAcc[m][n]);
//                }
                //@info: It's important that centers later in N-dim does not update nearestCenter when equally close.
                // This makes sure that for duplicate centers, descriptors are assigned to the first one
                if (n == 0 || mThrdAcc[m][n] < nearestCenter[m].minDistance){
                    nearestCenter[m] = {.minDistance = mThrdAcc[m][n], .idxCenter = n };
                }
            }
//            if (threadIdx.x == 0 && m == 1) {printf("\n");}
        }
        using CenterIndex = std::decay_t<decltype(*mArgs.idxNearestCenter)>;
        KArray<CenterIndex, thrdM> idxNearestCenter{};
        bool isInRange[thrdM];
        for (bool& e : isInRange) {
            e = true;
        }
        if (idxDescTile + 1 == mArgs.nbDescTiles) {
#pragma unroll
            for (uint32_t m = 0; m < thrdM; m++) {
                isInRange[m] = (tileSize * idxDescTile + thrdM * mThrdRank + m < mArgs.nbDesc);
            }
        }
#pragma unroll
        for (uint32_t m = 0; m < thrdM; m++){
            if (isInRange[m]) {
                atomicAdd(&(*mArgs.counter)[nearestCenter[m].idxCenter], 1u);
            }
            idxNearestCenter[m] = uint8_t(nearestCenter[m].idxCenter);
        }
        KArray<CenterIndex, thrdM>& glbCenterIdxRef = reinterpret_cast<KArray<CenterIndex, thrdM>(*)[grpSize]>(mArgs.idxNearestCenter)[idxDescTile][mThrdRank];
        if (mArgs.nbChanged != nullptr) {
            KArray<CenterIndex, thrdM> glbCenterIdx = glbCenterIdxRef;
            for (uint32_t m = 0; m < thrdM; m++) {
                if (isInRange[m]) {
                    if (glbCenterIdx[m] != nearestCenter[m].idxCenter) {
                        atomicAdd(&mCtaNbChanged[lane_id()], 1u);
                    }
                }
            }
        }
        glbCenterIdxRef = idxNearestCenter;
        if (mArgs.centerAcc != nullptr)
        {// this implementation causes too much bank conflict.
            assert(PREFETCH_FIRST_DESC_VEC == 0); // This implementation uses the same buffer for prefetch, so prefetch must be disabled.
#pragma unroll(1)
            for (int idxVecOuter = vecsPerDesc / 2 - 1; idxVecOuter >= 0; idxVecOuter--) {
#pragma unroll
                for (int idxBuf = 1; idxBuf >= 0; idxBuf--) {
                    const int idxVec = idxVecOuter * 2 + idxBuf;
                    const KArray<Vec, thrdM> &descVecs = mThrdDescVec[idxBuf];
                    KArray<Vec, thrdM> &descVecsNext = mThrdDescVec[(idxBuf + 1) % 2];
                    if (idxVec >= 1) {
                        loadDescVec(descVecsNext, idxDescTile, idxVec - 1u);// @todo: set cache control to last-use
                    }
#pragma unroll
                    for (uint32_t m = 0; m < thrdM; m++) {
                        constexpr uint32_t elemsPerVec = elemsPerWord * wordsPerVec;
                        if (isInRange[m]) {
#pragma unroll
                            for (unsigned idxInVec = 0; idxInVec < elemsPerVec; idxInVec++) {
                                const unsigned idxElem = elemsPerVec * idxVec + idxInVec;
                                const auto elem = extractElem<uint32_t>(descVecs[m], idxInVec);
                                atomicAdd(&mCtaCenterAcc[idxElem][idxNearestCenter[m]][mIdxGrp % nbSMemCenterAcc], elem);
                            }
                        }
                    }
                }
            }
        }
    }

    __device__ __forceinline__
    void compute()
    {
#if PREFETCH_FIRST_DESC_VEC
        if (mIdxGrp < mArgs.nbDescTiles){
            loadDescVec(mThrdDescVec[0], mIdxGrp, 0u);
        }
#endif
        for (uint32_t idxDescTile = mIdxGrp; idxDescTile < mArgs.nbDescTiles; idxDescTile += mNbGrps){
            computeTile(idxDescTile);
        }
    }

private:
    const cg::thread_block_tile<grpSize>& mGrp;
    const uint32_t mThrdRank; // save mGrp.thread_rank() to register costs 6 more registers, but it seems to be faster
    const uint32_t mIdxGrp;
    const uint32_t mNbGrps;
    const ArgsKMeansAssignCenter& mArgs;
    const Vec (&mCenters)[vecsPerDesc][nbCenters];
    KArray<Vec, thrdM> mThrdDescVec[2]; // double buffer
    Distance mThrdAcc[thrdM][thrdN] {};
    KArray<CtaAccType, nbSMemCenterAcc> (&mCtaCenterAcc)[descDims][nbCenters];
    uint32_t (&mCtaNbChanged)[warp_size];
};

__global__ void kernel_kmeansAssignCenter(const ArgsKMeansAssignCenter* __restrict__ tasks, uint32_t nbTasks)
{
    if (blockIdx.y >= nbTasks){
        return;
    }
    __shared__ ArgsKMeansAssignCenter args;
    assert(sizeof(ArgsKMeansAssignCenter) / sizeof(uint32_t) <= blockDim.x);
    if (threadIdx.x < sizeof(ArgsKMeansAssignCenter) / sizeof(uint32_t)){
        reinterpret_cast<uint32_t*>(&args)[threadIdx.x] = reinterpret_cast<const uint32_t*>(&tasks[blockIdx.y])[threadIdx.x];
    }
    __syncthreads();

    if (args.nbChagnedLast != nullptr && *args.nbChagnedLast == 0u){
        return;
    }
    using Traits = KMeansTraits;
    constexpr auto warp_size = Traits::warp_size;
    assert(warp_size == warpSize);
    constexpr auto grpSize = Traits::grpSize;
    const auto& cta = cg::this_thread_block();
    const cg::thread_block_tile<grpSize> g = cg::tiled_partition<grpSize>(cta);

    __shared__ typename Traits::Vec centers[Traits::vecsPerDesc][Traits::nbCenters];
    for (uint32_t i = cta.thread_rank(); i < Traits::vecsPerDesc * Traits::nbCenters; i += cta.size()){
        centers[0][i] = (*args.centers)[0][i];
    }
    static_assert(Traits::nbSMemCenterAcc > 0, "fatal error");
    __shared__ KArray<typename Traits::CtaAccType, Traits::nbSMemCenterAcc> ctaCenterAcc[Traits::descDims][Traits::nbCenters];
    for (uint32_t i = cta.thread_rank(); i < Traits::descDims * Traits::nbCenters * Traits::nbSMemCenterAcc; i += cta.size()){
         ctaCenterAcc[0][0][i] = Traits::CtaAccType{0};
    }
    __shared__ uint32_t ctaNbChanged[warp_size];
    if (cta.thread_rank() < warp_size){
        ctaNbChanged[lane_id()] = 0u;
    }

    cta.sync();

    const uint32_t nbGrps = blockDim.x * gridDim.x / grpSize;
    const uint32_t idxGrp = (blockDim.x * blockIdx.x + threadIdx.x) / grpSize;
    if (idxGrp < args.nbDescTiles){
        GrpTask task{g, idxGrp, nbGrps, args, centers, ctaCenterAcc, ctaNbChanged};
        task.compute();
    }
    cta.sync();
    if (args.nbChanged != nullptr && cta.thread_rank() < warp_size)
    {
#if 0
        const auto warp = cg::tiled_partition<warp_size>(cta);
        uint32_t sum = ctaNbChanged[warp.thread_rank()];
#pragma unroll
        for (unsigned i = warp_size / 2; i != 0; i /= 2) {
            const uint32_t sum_other = warp.shfl_xor(sum, i);
            sum += sum_other;
        }
        if (cta.thread_rank() == 0){
            atomicAdd(args.nbChanged, sum);
        }
#else
        atomicAdd(args.nbChanged, ctaNbChanged[lane_id()]);// seems compiler automatically does warp reduction
#endif
    }

    if (args.centerAcc != nullptr){
#pragma unroll(1)
        for (uint32_t i = cta.thread_rank(); i < Traits::descDims * Traits::nbCenters; i += cta.size()){
            uint32_t sum = 0u;
#pragma unroll
            for (const auto e : ctaCenterAcc[0][i].data){
                sum += e;
            }
            atomicAdd(&(*args.centerAcc)[0][i], sum);
        }
    }
}

void launchAssignCenter(const ArgsKMeansAssignCenter* __restrict__ tasks, uint32_t nbTasks, cudaStream_t stream)
{
    using Traits = KMeansTraits;
//    assert((nbDesc + Traits::tileSize - 1u) / Traits::tileSize == nbDescTiles);

    static const auto ctaSizeAndNbCtasPerSM = [](){
        int maxThrdsPerSM = 0;
        int bestCtaSize;
        int bestNbCtasPerSM;
        for (int testCtaSize = 1024; testCtaSize > 0; testCtaSize-= 32){
            int nbCtasPerSM = 0;
            cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nbCtasPerSM, &kernel_kmeansAssignCenter, testCtaSize, 0));
            const int thrdPerSM = testCtaSize * nbCtasPerSM;
            if (thrdPerSM > maxThrdsPerSM || (thrdPerSM == maxThrdsPerSM && bestNbCtasPerSM == 1)){
                bestCtaSize = testCtaSize;
                bestNbCtasPerSM = nbCtasPerSM;
                maxThrdsPerSM = thrdPerSM;
            }
        }
        return std::make_pair(bestCtaSize, bestNbCtasPerSM);
    }();
    const int ctaSize = ctaSizeAndNbCtasPerSM.first;
    const int nbCtasPerSM = ctaSizeAndNbCtasPerSM.second;
    const int device = getCudaDevice();
    cudaDeviceProp devProp;
    cudaCheck(cudaGetDeviceProperties(&devProp, device));
    const uint32_t nbCtasPerTask = std::max(devProp.multiProcessorCount * nbCtasPerSM / nbTasks, 1u);
    kernel_kmeansAssignCenter<<<dim3{nbCtasPerTask, nbTasks}, ctaSize, 0, stream>>>(tasks, nbTasks);
    cudaCheck(cudaGetLastError());
}

namespace {
__global__ void kernel_kmeansUpdateCenters(const ArgsKMeansAssignCenter* __restrict__ tasks, uint32_t nbTasks){
    using Traits = KMeansTraits;

    if (blockIdx.y >= nbTasks){
        return;
    }

    if (*tasks[blockIdx.y].nbChanged == 0){
        return;
    }
    __shared__ ArgsKMeansAssignCenter args;
    assert(sizeof(ArgsKMeansAssignCenter) / sizeof(uint32_t) <= blockDim.x);
    if (threadIdx.x < sizeof(ArgsKMeansAssignCenter) / sizeof(uint32_t)){
        reinterpret_cast<uint32_t*>(&args)[threadIdx.x] = reinterpret_cast<const uint32_t*>(&tasks[blockIdx.y])[threadIdx.x];
    }
    __syncthreads();

    const auto grp = cg::tiled_partition<Traits::nbCenters>(cg::this_thread_block());
    const auto idxGrp = (blockDim.x * blockIdx.x + threadIdx.x) / grp.size();
    const auto nbGrps = blockDim.x * gridDim.x / grp.size();
    const auto elemsPerVec = Traits::descDims / Traits::vecsPerDesc;
    const auto idxCenter = grp.thread_rank();
    const uint32_t count = (*args.counter)[idxCenter];
    const bool hasDesc = (count != 0u);
    if (hasDesc) {
        assert(args.validMask->test(idxCenter));//@fixme: When !hasDesc, the mask may still be true, in case there are duplicate centoids. Better avoid that when picking kmeans seeds.
        for (uint32_t idxVec = idxGrp; idxVec < Traits::vecsPerDesc; idxVec += nbGrps) {
            KArray<Traits::GlbAccType, elemsPerVec> src;
            KArray<Traits::DescElemType, elemsPerVec> dst;
            for (uint32_t i = 0; i < elemsPerVec; i++) {
                src[i] = (*args.centerAcc)[elemsPerVec * idxVec + i][idxCenter];
            }
            for (uint32_t i = 0; i < elemsPerVec; i++) {
                dst[i] = static_cast<Traits::DescElemType>(std::round(float(src[i]) / float(count)));
            }
            static_assert(sizeof(dst) == sizeof(Traits::Vec), "fatal error");
            Traits::Vec tmp;
            memcpy(&tmp, &dst, sizeof(dst));
            const_cast<typename Traits::Vec &>((*args.centers)[idxVec][idxCenter]) = tmp;
        }
    }
}
}

void launchKMeansUpdateCenters(const ArgsKMeansAssignCenter* __restrict__ tasks, uint32_t nbTasks, cudaStream_t stream){
    constexpr uint32_t ctaSize = 512;
    static_assert(sizeof(ArgsKMeansAssignCenter) % sizeof(uint32_t) == 0);
    static_assert(sizeof(ArgsKMeansAssignCenter) / sizeof(uint32_t) <= ctaSize);
    kernel_kmeansUpdateCenters << < dim3(1, nbTasks), ctaSize, 0, stream >> > (tasks, nbTasks);
    cudaCheck(cudaGetLastError());
}
