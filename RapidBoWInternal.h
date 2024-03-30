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

namespace rbow{
class IVocabularyInternal : public IVocabulary
{
public:
    // returns host pointer. The real type should be const std::array<float, branchFactor>[levelSize(idxLevel)].
    virtual const void* getLevelWeights(uint32_t idxLevel) const = 0;
};
}
