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
// Created by yao on 10/23/19.
//

#pragma once
#include "BitSet.h"

namespace rbow {

template<uint32_t lg2BranchFactor_>
struct FullTree {
    static constexpr uint32_t lg2BranchFactor = lg2BranchFactor_;
    static constexpr uint32_t branchFactor = (1u << lg2BranchFactor);

    // in number of nodes, not number of branches
    static constexpr uint32_t levelSize(uint32_t level) {
        return 1u << (lg2BranchFactor * level);
    }

    template<int impl = 0>
    static constexpr uint32_t levelBeg(uint32_t level) {
        if (impl == 0)
            return level == 0u ? 0u : 1u + ((1u << (lg2BranchFactor * (level - 1u))) - 1u) / (branchFactor - 1u) *
                                           branchFactor;
        else if (impl == 1)
            return ((1u << (lg2BranchFactor * level)) - 1u) /
                   (branchFactor - 1u); // this method is more likely to overflow
        else {
            uint32_t acc = 0u;
            for (uint32_t i = 0u; i < level; i++) {
                acc = (acc << lg2BranchFactor) + 1u;
            }
            return acc;
        }
    }

    static_assert(levelBeg<0>(3) == 1 + branchFactor + branchFactor * branchFactor
                  && levelBeg<1>(3) == 1 + branchFactor + branchFactor * branchFactor
                  && levelBeg<2>(3) == 1 + branchFactor + branchFactor * branchFactor);

    //! \return New idxNodeInLevel in the next level
    static constexpr uint32_t gotoChild(uint32_t idxNodeInLevel, uint32_t idxBranch) {
        return (idxNodeInLevel << lg2BranchFactor) + idxBranch;
    }

    static constexpr uint32_t idxNodeLocal2Global(uint32_t level, uint32_t idxNodeInLevel) {
        return levelBeg(level) + idxNodeInLevel;
    }
};
}