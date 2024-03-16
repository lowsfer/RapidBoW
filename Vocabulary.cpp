//
// Created by yao on 10/2/19.
//

#include "Vocabulary.h"
#include "cuda_utils.h"
#include <numeric>
#include "vocTreeLookUp.h"
#include <cassert>

using namespace rbow;
using Traits = KMeansTraits;

namespace rbow {
template<unsigned lg2BranchFactor>
Vocabulary<DataType::kUInt8, DistanceType::kL2, 128u, lg2BranchFactor> buildSiftVocabulary(
        const KMeansTraits::Descriptor* devDesc, uint32_t nbDesc,
        uint32_t nbDoc,
        uint32_t nbLevels, cudaStream_t stream) {
    constexpr uint32_t branchFactor = (1u << lg2BranchFactor);
    static_assert(branchFactor == Traits::nbCenters);
    if (nbDesc < branchFactor) {
        throw std::runtime_error("Insufficient descriptors");
    }
    if (nbDoc <= 0) {
        throw std::runtime_error("nbDoc must be positive");
    }

    using TreeType = FullTree<lg2BranchFactor>;
    const uint32_t nbTreeNodes = TreeType::levelBeg(nbLevels);
    auto centersAllLevel = allocCudaMem<typename Traits::Centers, CudaMemType::kDevice>(nbTreeNodes);
    auto validMaskAllLevel = allocCudaMem<BitSet<branchFactor>, CudaMemType::kDevice, true>(nbTreeNodes);

    std::vector<std::vector<uint32_t>> groups{std::vector<uint32_t>(nbDesc)};
    std::iota(groups.at(0).begin(), groups.at(0).begin() + nbDesc, 0u);

    std::vector<std::array<float, branchFactor>> weights(nbTreeNodes);
    union OccurenceWeightUnion
    {
        uint32_t occurence;
        float weight;
    };
    std::array<OccurenceWeightUnion, branchFactor>* const occurenceWeightTree
        = reinterpret_cast<std::array<OccurenceWeightUnion, branchFactor>*>(weights.data());
    assert(std::all_of(occurenceWeightTree[0].data(), occurenceWeightTree[0].data() + branchFactor*nbTreeNodes,
            [](auto a){return a.occurence == 0;}));
    static_assert(sizeof(occurenceWeightTree[0]) == sizeof(weights[0]));

    KMeans kmeans{devDesc, nbDesc, stream};
    for (uint32_t level = 0; level < nbLevels; level++) {
        // run kmeans
        const DescGrouping grouping{groups};
        const uint32_t nbTiles = divUp(grouping.idxEnd(grouping.getNbGrps() - 1u), Traits::tileSize);
        const auto idxNearestCenter = allocCudaMem<KArray<uint8_t, Traits::tileSize>, CudaMemType::kDevice>(nbTiles);
        typename Traits::Centers *const centersThisLevel = &centersAllLevel[TreeType::levelBeg(level)];
        BitSet<branchFactor> *const validMaskThisLevel = &validMaskAllLevel[TreeType::levelBeg(level)];
        kmeans.setGrouping(&grouping, idxNearestCenter.get(), centersThisLevel, validMaskThisLevel);
        //@fixme: consider using multiple streams. Maybe useful for deep levels
        kmeans.compute(0u, grouping.getNbGrps());

        std::vector<uint8_t, CudaHostAllocator<uint8_t>> idxCenter(Traits::tileSize *nbTiles);
        cudaCheck(cudaMemcpyAsync(idxCenter.data(), idxNearestCenter.get(), sizeof(uint8_t) * idxCenter.size(),
                                  cudaMemcpyDeviceToHost, stream));
        cudaCheck(cudaStreamSynchronize(stream));

        groups.clear();
        groups.resize(grouping.getNbGrps() * branchFactor);
        for (uint32_t idxGrpOld = 0; idxGrpOld < grouping.getNbGrps(); idxGrpOld++) {
            const auto grpsNew = groups.begin() + idxGrpOld * branchFactor;
            for (uint32_t i = grouping.idxBeg(idxGrpOld); i < grouping.idxEnd(idxGrpOld); i++) {
                grpsNew[idxCenter.at(i)].push_back(grouping.getIndices()[i]);
            }
            for (uint32_t idxBranch = 0; idxBranch < branchFactor; idxBranch++) {
                occurenceWeightTree[TreeType::levelBeg(level) + idxGrpOld][idxBranch].occurence = static_cast<uint32_t>(grpsNew[idxBranch].size());
                //@fixme: This assertion may fail when there are two identical descriptors picked as the initial centers.
                // Initialize non-duplicate centers require storage of descriptors in host memory, which we are not doing currently.
                // When we change to kmeans++ init of 16 centers from some (e.g. 256) randomly sampled descriptors,
                // we will need host descriptors anyway and we should make it non-duplicate
//                assert(validMaskAllLevel[TreeType::levelBeg(level) + idxGrpOld].test(idxBranch) == (occurenceWeightTree[TreeType::levelBeg(level) + idxGrpOld][idxBranch].occurence != 0));
            }
        }
    }
    // @info : This is similar to IDF used in the paper, but not exactly, as we are not removing
    // duplicate word occurrences in the same document (image). The paper author claimed that they
    // also tried this and there is no difference.
    for (uint32_t i = 0; i < nbTreeNodes; i++) {
        for (auto& v : occurenceWeightTree[i]) {
            //@fixme: when v.occurence is 0, v.weight should not matter, as no descriptor should reach this branch. But actually it is used sometimes. Likely it's a bug.
            v.weight = v.occurence == 0 ? std::numeric_limits<float>::min() : std::max(std::numeric_limits<float>::min(), std::log(float(nbDoc) / float(v.occurence)));
        }
    }

    using VocType = Vocabulary<DataType::kUInt8, DistanceType::kL2, 128u, lg2BranchFactor>;
    using Node = typename VocType::Node;
    static_assert(sizeof(typename VocType::Node) == sizeof(Traits::Centers)
                  && sizeof(Node[0]) == sizeof(Traits::Centers[0])
                  && sizeof(Node[0][0]) == sizeof(Traits::Centers[0][0]));
    return VocType{
        nbLevels,
        CudaMem<Node, CudaMemType::kDevice>{reinterpret_cast<Node *>(centersAllLevel.release())},
        std::move(validMaskAllLevel),
        std::move(weights)
    };
}

template Vocabulary<DataType::kUInt8, DistanceType::kL2, 128u, 4u>
buildSiftVocabulary<4u>(const Traits::Descriptor *devDesc, uint32_t nbDesc, uint32_t nbDoc, uint32_t nbLevels, cudaStream_t stream);

template <unsigned lg2BranchFactor>
Vocabulary<DataType::kUInt8, DistanceType::kL2, 128u, lg2BranchFactor>
deserializeSiftVocabulary(const std::uint8_t* blob, const size_t blobSize){
    if (blobSize < 4u){
        throw std::runtime_error("invalid blob");
    }
    using Voc = Vocabulary<DataType::kUInt8, DistanceType::kL2, 128u, lg2BranchFactor>;
    size_t offset = 0;
    uint32_t nbLevels{};
    size_t size = sizeof(nbLevels);
    std::copy_n(blob + offset, size, reinterpret_cast<std::uint8_t*>(&nbLevels));
    offset += size;
    const uint32_t nbNodes = Voc::Tree::levelBeg(nbLevels);
    auto nodes = allocCudaMem<typename Voc::Node>(nbNodes);
    auto masks = allocCudaMem<BitSet<Voc::branchFactor>>(nbNodes);
    std::vector<std::array<float, Voc::branchFactor>> weights(nbNodes);

    if (blobSize != offset + sizeof(nodes[0]) * nbNodes + sizeof(masks[0]) * nbNodes + sizeof(weights[0]) * nbNodes){
        throw std::runtime_error("invalid blob");
    }

    size = sizeof(nodes[0]) * nbNodes;
    cudaCheck(cudaMemcpy(nodes.get(), &blob[offset], size, cudaMemcpyHostToDevice));
    offset += size;
    size = sizeof(masks[0]) * nbNodes;
    cudaCheck(cudaMemcpy(masks.get(), &blob[offset], size, cudaMemcpyHostToDevice));
    offset += size;
    size = sizeof(weights[0]) * nbNodes;
    assert(weights.size() == Voc::Tree::levelBeg(nbLevels));
    std::copy_n(&blob[offset], size, reinterpret_cast<std::uint8_t*>(weights.data()));
    return Voc(nbLevels, std::move(nodes), std::move(masks), std::move(weights));
}

template Vocabulary<DataType::kUInt8, DistanceType::kL2, 128u, 4u>
deserializeSiftVocabulary<4u>(const std::uint8_t* blob, size_t blobSize);

template <DataType dataType, DistanceType distanceType, size_t nbDims, uint32_t lg2BranchFactor>
void Vocabulary<dataType, distanceType, nbDims, lg2BranchFactor>::lookUpImpl(uint32_t nbLevels,
            const std::array<std::byte, getDescriptorBytes(descAttr)>* __restrict__ descriptors, uint32_t nbDesc,
            uint32_t* __restrict__ indicesInLeafLevel, //indices in the leaf level (idxLevel == nbLevels)
            cudaStream_t stream) const
{
    launchVocTreeLookUp(
            reinterpret_cast<const KArray<Traits::Vec, Traits::vecsPerDesc, Traits::nbCenters>*>(mNodes.get()),
            mNodeMasks.get(), nbLevels,
            reinterpret_cast<const typename Traits::Descriptor*>(descriptors), nbDesc,
            indicesInLeafLevel, stream);
}

template<DataType dataType, DistanceType distanceType, size_t nbDims, uint32_t lg2BranchFactor_>
std::vector<std::uint8_t> Vocabulary<dataType, distanceType, nbDims, lg2BranchFactor_>::serialize() const {
    std::vector<std::uint8_t> result(
        sizeof(mNbLevels) +
        sizeof(mNodes[0]) * Tree::levelBeg(mNbLevels) +
        sizeof(mNodeMasks[0]) * Tree::levelBeg(mNbLevels) +
        sizeof(mWeights[0]) * Tree::levelBeg(mNbLevels)
    );
    size_t offset = 0;
    size_t size = sizeof(mNbLevels);
    std::copy_n(reinterpret_cast<const std::uint8_t*>(&mNbLevels), size, result.begin());
    offset += size;
    size = sizeof(mNodes[0]) * Tree::levelBeg(mNbLevels);
    cudaCheck(cudaMemcpy(&result[offset], mNodes.get(), size, cudaMemcpyDeviceToHost));
    offset += size;
    size = sizeof(mNodeMasks[0]) * Tree::levelBeg(mNbLevels);
    cudaCheck(cudaMemcpy(&result[offset], mNodeMasks.get(), size, cudaMemcpyDeviceToHost));
    offset += size;
    size = sizeof(mWeights[0]) * Tree::levelBeg(mNbLevels);
    assert(mWeights.size() == Tree::levelBeg(mNbLevels));
    std::copy_n(reinterpret_cast<const std::uint8_t*>(mWeights.data()), size, &result[offset]);
    return result;
}

} // namespace rbow

template class rbow::Vocabulary<DataType::kUInt8, DistanceType::kL2, Traits::descDims, 4u>;
