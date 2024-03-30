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

#ifndef RAPIDBOW_RAPIDBOW_H
#define RAPIDBOW_RAPIDBOW_H

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <array>
#include <cuda_runtime_api.h>
#include <vector>

namespace rbow
{

enum class DataType{
    kBinary,
    kUInt8
};

enum class DistanceType{
    kHamming,
    kL1,
    kL2
};

struct DescriptorAttributes
{
    DataType dataType;
    DistanceType distanceType;
    size_t nbDims;
};

constexpr DescriptorAttributes SiftAttr{DataType::kUInt8, DistanceType::kL2, 128u};

constexpr inline size_t getDescriptorBytes(DescriptorAttributes descAttr)
{
    switch (descAttr.dataType){
        case DataType::kBinary: return (descAttr.nbDims + 7) / 8;
        case DataType::kUInt8: return descAttr.nbDims;
    }
}

class IVocabulary
{
public:
    virtual DescriptorAttributes getDescriptorAttributes() const = 0;
    virtual uint32_t getBranchFactor() const = 0;
    virtual uint32_t getNbLevels() const = 0;
    // descriptors and indicesInLeafLevel must be accessible from GPU
    virtual void
    lookUp(const void *descriptors, uint32_t nbDesc, uint32_t *indicesInLeafLevel, cudaStream_t stream) const = 0;
    virtual std::vector<std::uint8_t> serialize() const = 0;
    virtual ~IVocabulary();
};

namespace impl {
//@fixme: Consider applying PCA to reduce descriptor dimensionality
IVocabulary *buildVocabularyImpl(
        DescriptorAttributes descAttr, uint32_t branchFactor,
        const void *devDesc, uint32_t nbDesc,
        uint32_t nbDoc,
        uint32_t nbLevels, cudaStream_t stream);
} // namespace impl

inline std::unique_ptr<IVocabulary> buildVocabulary(
        DescriptorAttributes descAttr, uint32_t branchFactor,
        const void* devDesc, uint32_t nbDesc,
        uint32_t nbDoc,
        uint32_t nbLevels, cudaStream_t stream){
    return std::unique_ptr<IVocabulary>{impl::buildVocabularyImpl(descAttr, branchFactor, devDesc, nbDesc, nbDoc, nbLevels, stream)};
}

std::unique_ptr<IVocabulary> deserializeVocabulary(const std::uint8_t* blob, size_t size);

class IDataBase
{
public:
    virtual const IVocabulary* getVocabulary() const = 0;
    // indicesInLeafLevel is host memory, unlike in IVocabulary::lookUp
    virtual void addDoc(uint32_t idxDoc, const uint32_t *hostIndicesInLeafLevel, uint32_t nbDesc) = 0;
    // indicesInLeafLevel is host memory, unlike in IVocabulary::lookUp
    virtual std::vector<uint32_t>
    query(const uint32_t *hostIndicesInLeafLevel, uint32_t nbDesc, uint32_t maxNbResults) const = 0;
    // indicesInLeafLevel is host memory, unlike in IVocabulary::lookUp
    virtual std::vector<uint32_t>
    queryAndAddDoc(uint32_t idxDoc, const uint32_t *hostIndicesInLeafLevel, uint32_t nbDesc, uint32_t maxNbResults) = 0;
    virtual ~IDataBase();
};

std::unique_ptr<IDataBase> createDataBase(const IVocabulary* voc);

}

#endif
