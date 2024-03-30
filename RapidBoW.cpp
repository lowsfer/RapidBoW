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

#include <iostream>
#include "RapidBoW.h"
#include "Vocabulary.h"
#include "DataBase.h"

namespace rbow{
IVocabulary::~IVocabulary() = default;

IVocabulary* impl::buildVocabularyImpl(
        DescriptorAttributes descAttr, uint32_t branchFactor, const void* devDesc, uint32_t nbDesc,
        uint32_t nbDoc, uint32_t nbLevels, cudaStream_t stream){
    if (descAttr.dataType == DataType::kUInt8
        && descAttr.distanceType == DistanceType::kL2
        && descAttr.nbDims == 128u
        && branchFactor == 16u){
        constexpr uint32_t lg2BranchFactor= 4u;
        return new Vocabulary<DataType::kUInt8, DistanceType::kL2, 128u, lg2BranchFactor>(
                buildSiftVocabulary<4u>(static_cast<const KMeansTraits::Descriptor*>(devDesc), nbDesc, nbDoc, nbLevels, stream));
    }
    else{
        std::cout << "Only {DataType::kUInt8, DistanceType::kL2, 128}, i.e. quantized SIFT, and branchFactor = 16 is supported" << std::endl;
        return nullptr;
    }
}

std::unique_ptr<IVocabulary> deserializeVocabulary(const std::uint8_t* blob, size_t blobSize){
    return std::make_unique<Vocabulary<DataType::kUInt8, DistanceType::kL2, 128u, 4u>>(
            deserializeSiftVocabulary<4u>(blob, blobSize));
}

IDataBase::~IDataBase() = default;

std::unique_ptr<IDataBase> createDataBase(const IVocabulary* voc){
    return std::make_unique<DataBase>(*voc);
}

}
