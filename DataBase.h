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
// Created by yao on 10/26/19.
//

#pragma once
#include "RapidBoW.h"
#include "FullTree.h"
#ifdef NDEBUG
#include "RapidBoWInternal.h"
#else
#include "Vocabulary.h"
#endif
#include <shared_mutex>
#include <unordered_map>

namespace rbow {

class DataBase : public IDataBase
{
public:
    using IdxDoc = uint32_t;
    static constexpr uint32_t lg2BranchFactor = 4u;
    static constexpr uint32_t branchFactor = (1u << lg2BranchFactor);
    using Tree = FullTree<lg2BranchFactor>;

    explicit DataBase(const IVocabulary& vocabulary)
    :mVoc{dynamic_cast<decltype(mVoc)>(vocabulary)}
    {assert(mVoc.getBranchFactor() == branchFactor);}
    const IVocabulary* getVocabulary() const override {return &mVoc;}
    void addDoc(uint32_t idxDoc, const uint32_t *hostIndicesInLeafLevel, uint32_t nbDesc) override;
    std::vector<uint32_t>
    query(const uint32_t *hostIndicesInLeafLevel, uint32_t nbDesc, uint32_t maxNbResults) const override;
    std::vector<uint32_t>
    queryAndAddDoc(uint32_t idxDoc, const uint32_t *hostIndicesInLeafLevel, uint32_t nbDesc, uint32_t maxNbResults) override;
private:
#ifdef NDEBUG
    const IVocabularyInternal& mVoc;
#else
    const Vocabulary<DataType::kUInt8, DistanceType::kL2, KMeansTraits::descDims, 4u>& mVoc;
#endif
    using InvertedFile = std::vector<IdxDoc>;// If an inverted file contains duplicate IdxDoc's, it means multiple descriptors in that image is clustered to this leaf node.
    std::vector<std::array<InvertedFile, branchFactor>> mLastLvlIvF{Tree::levelSize(mVoc.getNbLevels() - 1)}; // use virtual inverted file for other levels
    uint32_t mStopThresholdOffset = 16;
    uint32_t mStopThresholdScale = 4;// we drop non-leaf inverted files with more than (mStopThresholdOffset + maxNbResults * mStopThresholdScale) documents
    std::unordered_map<IdxDoc, float, DirectMappingHash<IdxDoc>> mWeightVecRcpL1Norm;
    mutable std::shared_mutex mIvFLock;
};

}



