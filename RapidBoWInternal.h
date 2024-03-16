//
// Created by yao on 10/26/19.
//

#pragma once
#include "RapidBoW.h"

namespace rbow{
class IVocabularyInternal : public IVocabulary
{
public:
    // returns host pointer. The real type should be const std::array<float, branchFactor>[levelSize(idxLevel)].
    virtual const void* getLevelWeights(uint32_t idxLevel) const = 0;
};
}
