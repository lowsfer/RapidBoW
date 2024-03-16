//
// Created by yao on 10/2/19.
//

#pragma once
#include <cuda_utils.h>
#include <cassert>
#include "RapidBoWInternal.h"
#include "kmeans.h"
#include "BitSet.h"
#include "FullTree.h"
#include <array>

namespace rbow {

template <DataType dataType, DistanceType distanceType, size_t nbDims, uint32_t lg2BranchFactor_>
class Vocabulary : public IVocabularyInternal
{
public:
    static constexpr DescriptorAttributes descAttr {dataType, distanceType, nbDims};
    static constexpr uint32_t lg2BranchFactor = lg2BranchFactor_;
    static constexpr uint32_t branchFactor = (1u << lg2BranchFactor);
    using Tree = FullTree<lg2BranchFactor>;

    using Node = KArray<uint32_t, divUp(getDescriptorBytes(descAttr), sizeof(uint32_t)), branchFactor>;

    Vocabulary(uint32_t nbLevels, CudaMem<Node, CudaMemType::kDevice> nodes,
            CudaMem<BitSet<branchFactor>, CudaMemType::kDevice> nodeValidMasks,
            std::vector<std::array<float, branchFactor>> nodeWeights)
    : mNbLevels{nbLevels}
    , mNodes{std::move(nodes)}
    , mNodeMasks{std::move(nodeValidMasks)}
    , mWeights{std::move(nodeWeights)}
    {}

    DescriptorAttributes getDescriptorAttributes() const override {return descAttr;}
    uint32_t getBranchFactor() const override {return branchFactor;}
    uint32_t getNbLevels() const override {return mNbLevels;}
    const Node* getTree() const {return mNodes.get();}
    const BitSet<branchFactor>* getMask() const {return mNodeMasks.get();}

    void lookUp(const void *descriptors, uint32_t nbDesc, uint32_t *indicesInLeafLevel, cudaStream_t stream) const override {
        lookUpImpl(getNbLevels(),
                static_cast<const std::array<std::byte, getDescriptorBytes(descAttr)>*>(descriptors), nbDesc,
                indicesInLeafLevel, stream);
    }
    const void* getLevelWeights(uint32_t idxLevel) const override {
        return &mWeights.at(Tree::levelBeg(idxLevel));
    }

    std::vector<std::uint8_t> serialize() const override;

private:
    void lookUpImpl(uint32_t nbLevels,
            const std::array<std::byte, getDescriptorBytes(descAttr)>* __restrict__ descriptors, uint32_t nbDesc,
                uint32_t* __restrict__ indicesInLeafLevel, //indices in the leaf level (idxLevel == nbLevels)
            cudaStream_t stream) const;
    uint32_t getNbNodes() const {return Tree::levelBeg(getNbLevels());}
private:
    // leaf nodes are not counted as they do not contain any information
    uint32_t mNbLevels{0}; // mNbLevels == 1 means one node.

    // Breadth-first traversal of the tree. Length is 1 + branchFactor + pow(branchFactor, 2) + pow(branchFactor, 3) + ... + pow(branchFactor, mNbLevels - 1)
    // Use Tree to help traversing. level is in [0, mNbLevels) range
    CudaMem<Node, CudaMemType::kDevice> mNodes;
    // A dynamic bitset. Same structure as mNodes
    CudaMem<BitSet<branchFactor>, CudaMemType::kDevice> mNodeMasks;
    // node weights, same structure as mNodes. Stored on host.
    std::vector<std::array<float, branchFactor>> mWeights;
};

// Intiantiated for lg2BranchFactor = 4u
template <unsigned lg2BranchFactor>
Vocabulary<DataType::kUInt8, DistanceType::kL2, 128u, lg2BranchFactor> buildSiftVocabulary(
        const KMeansTraits::Descriptor* devDesc, uint32_t nbDesc,
        uint32_t nbDoc,
        uint32_t nbLevels, cudaStream_t stream);

template <unsigned lg2BranchFactor>
Vocabulary<DataType::kUInt8, DistanceType::kL2, 128u, lg2BranchFactor> deserializeSiftVocabulary(const std::uint8_t* blob, size_t blobSize);
}

