//
// Created by yao on 10/23/19.
//
#include "cuda_hint.cuh"
#include <cuda_runtime.h>
#include "kmeans.cuh"
#include "FullTree.h"
#include <cooperative_groups.h>
#include <cassert>
#include "vocTreeLookUp.h"

using namespace rbow;
namespace cg = cooperative_groups;

namespace {
using Traits = KMeansTraits;
//using Voc = typename Vocabulary<DataType::kUInt8, DistanceType::kL2, Traits::descDims, Traits::nbCenters>;
//using Node = Voc::Node;
using Node = KArray<Traits::Vec, Traits::vecsPerDesc, Traits::nbCenters>;
constexpr uint32_t lg2BranchFactor = 4u;
static_assert((1<<lg2BranchFactor) == Traits::nbCenters);

constexpr uint32_t ctaSize = 128;
constexpr uint32_t nbVecsPerLdSt = 4;
using LdStType = KArray<typename Traits::Vec, nbVecsPerLdSt>;
constexpr uint2 thrdsPerDesc = {4, 2};
static_assert(Traits::nbCenters == nbVecsPerLdSt * thrdsPerDesc.x);
constexpr uint32_t nbDescPerCta = ctaSize / (thrdsPerDesc.x * thrdsPerDesc.y);
constexpr uint32_t cacheLineWidth = 128;
static_assert(cacheLineWidth == sizeof(std::declval<Node>()[0]) * thrdsPerDesc.y);
__global__ void __launch_bounds__(ctaSize) kernel_VocTreeLookUp(
        const Node *__restrict__ nodes, const BitSet<Traits::nbCenters> *__restrict__ masks, const uint32_t nbLevels,
        const Traits::Descriptor *__restrict__ descriptors, const uint32_t nbDesc,
        uint32_t *__restrict__ indicesInLeafLevel) {
    const uint32_t& idxCta = blockIdx.x;
    if (nbDescPerCta * idxCta > nbDesc){
        return;
    }
    constexpr uint32_t nbLdStPerDesc = sizeof(Traits::Descriptor) / sizeof(LdStType);
    __shared__ LdStType smemDescStorage[nbDescPerCta][nbLdStPerDesc];
    auto smemDesc = [](uint32_t idxSMemDesc, uint32_t idxLdSt) -> LdStType& {
        return smemDescStorage[idxSMemDesc][(idxLdSt + idxSMemDesc) % nbLdStPerDesc];
    };


    static_assert(nbDescPerCta * nbLdStPerDesc % ctaSize == 0);
#pragma unroll
    for (uint32_t i = 0; i < nbDescPerCta * nbLdStPerDesc / ctaSize; i++) {
        const uint32_t idx = ctaSize * i + threadIdx.x;
        const uint32_t idxSMemDesc = idx / nbLdStPerDesc;
        const uint32_t idxLdSt = idx % nbLdStPerDesc;
        if (nbDescPerCta * idxCta + idxSMemDesc < nbDesc){
            smemDesc(idxSMemDesc, idxLdSt) =
                    reinterpret_cast<const LdStType *>(&descriptors[nbDescPerCta * idxCta])[idx];
        }
    }
    __syncthreads();

    using Tree = FullTree<lg2BranchFactor>;

    const auto g = cg::tiled_partition<thrdsPerDesc.x * thrdsPerDesc.y>(cg::this_thread_block());
    const uint2 idxThrdInGrp{g.thread_rank() % thrdsPerDesc.x, g.thread_rank() / thrdsPerDesc.x};
    const uint32_t idxGrpInCta = threadIdx.x / g.size();

    uint32_t idxNodeInLevel = 0;
    for (uint32_t idxLevel = 0; idxLevel < nbLevels; idxLevel++){
        KArray<Traits::Distance, nbVecsPerLdSt> distances{};
        const uint32_t idxNode = Tree::levelBeg(idxLevel) + idxNodeInLevel;
#pragma unroll
        for (uint32_t idxVecIter = 0; idxVecIter < Traits::vecsPerDesc / thrdsPerDesc.y; idxVecIter++){
            const uint32_t idxVec = idxVecIter * thrdsPerDesc.y + idxThrdInGrp.y;
            // we rely on unroll and compiler to generate non-duplicate LDS.128
            const Traits::Vec descVec = smemDesc(idxGrpInCta, idxVec / nbVecsPerLdSt)[idxVec % nbVecsPerLdSt];
            using NodeLdStType = KArray<LdStType, Traits::vecsPerDesc, Traits::nbCenters / nbVecsPerLdSt>;
            const KArray<typename Traits::Vec, nbVecsPerLdSt> nodeVecs =
                    reinterpret_cast<const NodeLdStType&>(nodes[idxNode])[idxVec][idxThrdInGrp.x];
            for (uint32_t i = 0; i < nbVecsPerLdSt; i++) {
                distances[i] = Traits::accumulate(distances[i], descVec, nodeVecs[i]);
            }
        }
#pragma unroll
        for (uint32_t xorMask = 1; xorMask < thrdsPerDesc.y; xorMask *= 2){
            for (uint32_t i = 0; i < distances.dimension; i++) {
                distances[i] += g.shfl_xor(distances[i], xorMask * thrdsPerDesc.x);
            }
        }

        Traits::Distance minDistance;
        uint32_t idxBestBranch;
#pragma unroll
        for (uint32_t i = 0; i < distances.dimension; i++) {
            if (i == 0 || distances[i] < minDistance) {
                minDistance = distances[i];
                idxBestBranch = nbVecsPerLdSt * idxThrdInGrp.x + i;
            }
        }
        assert(idxBestBranch < (1u<<8));
        assert(minDistance < (1u<<24));
        uint32_t combinedBestDistanceIdxBranch = ((minDistance << 8) | (idxBestBranch & 0xFFu));
        const auto innerGrp = cg::tiled_partition<thrdsPerDesc.x>(g);
#pragma unroll
        for (uint32_t xorMask = 1; xorMask < thrdsPerDesc.x; xorMask *= 2){
            combinedBestDistanceIdxBranch = std::min(combinedBestDistanceIdxBranch,
                    innerGrp.shfl_xor(combinedBestDistanceIdxBranch, xorMask));
        }
        assert(g.shfl(combinedBestDistanceIdxBranch, 0) == combinedBestDistanceIdxBranch);
        minDistance = (combinedBestDistanceIdxBranch >> 8);
        idxBestBranch = (combinedBestDistanceIdxBranch & 0xFFu);
        if (nbDescPerCta * idxCta + idxGrpInCta < nbDesc) {
            assert(masks[Tree::levelBeg(idxLevel) + idxNodeInLevel].test(idxBestBranch));
            unused(masks);
        }

#ifndef NDEBUG
        {
            const uint32_t idxDesc = nbDescPerCta * idxCta + idxGrpInCta;
            if (idxDesc < nbDesc) {
                uint32_t refIdxBestBranch;
                Traits::Distance refMinDistance = std::numeric_limits<Traits::Distance>::max();
                for (uint32_t idxBranch = 0; idxBranch < Traits::nbCenters; idxBranch++) {
                    uint32_t refDistance = 0;
                    for (uint32_t idxVec = 0; idxVec < Traits::vecsPerDesc; idxVec++) {
                        refDistance = Traits::accumulate(refDistance, descriptors[idxDesc][idxVec],
                                                         nodes[idxNode][idxVec][idxBranch]);
                    }
                    if (refDistance < refMinDistance) {
                        refIdxBestBranch = idxBranch;
                        refMinDistance = refDistance;
                    }
                }
                assert(idxBestBranch == refIdxBestBranch);
                assert(minDistance == refMinDistance);
            }
        }
#endif
        const uint32_t idxNodeInNextLevel = Tree::gotoChild(idxNodeInLevel, idxBestBranch);
        idxNodeInLevel = idxNodeInNextLevel;
    }
    const uint32_t idxDesc = nbDescPerCta * idxCta + idxGrpInCta;
    if (g.thread_rank() == 0 && idxDesc < nbDesc){
        indicesInLeafLevel[idxDesc] = idxNodeInLevel;
    }
}

}

namespace rbow{
void launchVocTreeLookUp(const Node *__restrict__ nodes, const BitSet<Traits::nbCenters> *__restrict__ masks, const uint32_t nbLevels,
                         const Traits::Descriptor *__restrict__ descriptors, const uint32_t nbDesc,
                         uint32_t *__restrict__ indicesInLeafLevel, cudaStream_t stream)
{
    kernel_VocTreeLookUp <<< divUp(nbDesc, nbDescPerCta), ctaSize, 0, stream >>> (nodes, masks, nbLevels, descriptors, nbDesc, indicesInLeafLevel);
    cudaCheck(cudaGetLastError());
}
}
