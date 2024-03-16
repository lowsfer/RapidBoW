//
// Created by yao on 10/23/19.
//

//@fixme: a lot of duplicate code here. refactor!

#include "../Vocabulary.h"
#include <gtest/gtest.h>


using Traits = KMeansTraits;
uint32_t squaredNorm(const typename Traits::Descriptor& a, const typename Traits::Descriptor& b){
    uint32_t acc = 0;
    for (uint32_t i = 0; i < sizeof(a) / sizeof(typename Traits::DescElemType); i++){
        const int diff = int(reinterpret_cast<const uint8_t*>(&a)[i]) - int(reinterpret_cast<const uint8_t*>(&b)[i]);
        acc += diff * diff;
    }
    return acc;
}

TEST(VocTest, refCheck2d)
{
    constexpr uint32_t nbLevels = 2;
    const uint32_t nbDesc = 4096;
    const uint32_t nbDoc = 16;
    const uint32_t offset = 126;
    const cudaStream_t stream = nullptr;
    std::mt19937_64 rng{};
    std::uniform_int_distribution<uint16_t> dist{};

    std::vector<typename Traits::Descriptor, CudaManagedAllocator<typename Traits::Descriptor>> descriptors(nbDesc);
    for (auto& desc : descriptors){
        const auto val = dist(rng);
        memcpy(reinterpret_cast<uint8_t*>(&desc) + offset, &val, sizeof(val));
    }

    cudaCheck(cudaMemPrefetchAsync(descriptors.data(), sizeof(descriptors[0]) * descriptors.size(), getCudaDevice(), stream));
    const auto voc = rbow::buildSiftVocabulary<4u>(descriptors.data(), nbDesc, nbDoc, nbLevels, stream);
    using VocType = std::remove_const_t<decltype(voc)>;
    cudaCheck(cudaStreamSynchronize(stream));

    std::vector<uint32_t, CudaManagedAllocator<uint32_t>> indicesInLevel(nbDesc);
    cudaCheck(cudaMemPrefetchAsync(indicesInLevel.data(), sizeof(indicesInLevel[0]) * indicesInLevel.size(), getCudaDevice(), stream));
    voc.lookUp(reinterpret_cast<const std::array<std::byte, sizeof(typename Traits::Descriptor)> *>(descriptors.data()),
               nbDesc, indicesInLevel.data(), stream);
    cudaCheck(cudaStreamSynchronize(stream));

    cudaCheck(cudaMemPrefetchAsync(descriptors.data(), sizeof(descriptors[0]) * descriptors.size(), getCudaDevice(), stream));
    cudaCheck(cudaMemPrefetchAsync(indicesInLevel.data(), sizeof(indicesInLevel[0]) * indicesInLevel.size(), cudaCpuDeviceId, stream));
    cudaCheck(cudaStreamSynchronize(stream));

    std::vector<typename VocType::Node> tree(VocType::Tree::levelBeg(nbLevels));
    cudaCheck(cudaMemcpy(tree.data(), voc.getTree(), sizeof(tree[0]) * tree.size(), cudaMemcpyDeviceToHost));
    const typename VocType::Node* nodesLastLevel = &tree.at(VocType::Tree::levelBeg(nbLevels - 1));
    std::vector<typename Traits::Descriptor> leafCentroids{VocType::Tree::levelSize(nbLevels)};
    for (uint32_t i = 0; i < leafCentroids.size(); i++){
        for (uint32_t j = 0; j < Traits::vecsPerDesc; j++) {
            leafCentroids[i][j][0] = nodesLastLevel[i / Traits::nbCenters][j][i % Traits::nbCenters];
        }
    }

    uint64_t inertia = 0;
    for (uint32_t i = 0; i < nbDesc; i++){
        inertia += squaredNorm(descriptors.at(i), leafCentroids.at(indicesInLevel.at(i)));
    }
    EXPECT_LE(std::sqrt(inertia/nbDesc), 10);// should be ~9.95
}

TEST(VocTest, refCheckFull)
{
    constexpr uint32_t nbLevels = 3;
    const uint32_t nbDesc = 32;
    const uint32_t nbDoc = 4;
    const cudaStream_t stream = nullptr;
    std::mt19937_64 rng{};
    std::uniform_int_distribution<uint64_t> dist{};

    std::vector<typename Traits::Descriptor, CudaManagedAllocator<typename Traits::Descriptor>> descriptors(nbDesc);
    for (auto& desc : descriptors){
        std::array<uint64_t, sizeof(desc)/sizeof(uint64_t)> val;
        std::generate(val.begin(), val.end(), [&](){return dist(rng);});
        memcpy(&desc, &val, sizeof(val));
    }

    cudaCheck(cudaMemPrefetchAsync(descriptors.data(), sizeof(descriptors[0]) * descriptors.size(), getCudaDevice(), stream));
    const auto voc = rbow::buildSiftVocabulary<4u>(descriptors.data(), nbDesc, nbDoc, nbLevels, stream);
    using VocType = std::remove_const_t<decltype(voc)>;
    cudaCheck(cudaStreamSynchronize(stream));

    std::vector<uint32_t, CudaManagedAllocator<uint32_t>> indicesInLevel(nbDesc);
    cudaCheck(cudaMemPrefetchAsync(indicesInLevel.data(), sizeof(indicesInLevel[0]) * indicesInLevel.size(), getCudaDevice(), stream));
    voc.lookUp(reinterpret_cast<const std::array<std::byte, sizeof(typename Traits::Descriptor)> *>(descriptors.data()),
               nbDesc, indicesInLevel.data(), stream);
    cudaCheck(cudaStreamSynchronize(stream));

    cudaCheck(cudaMemPrefetchAsync(descriptors.data(), sizeof(descriptors[0]) * descriptors.size(), getCudaDevice(), stream));
    cudaCheck(cudaMemPrefetchAsync(indicesInLevel.data(), sizeof(indicesInLevel[0]) * indicesInLevel.size(), cudaCpuDeviceId, stream));
    cudaCheck(cudaStreamSynchronize(stream));

    std::vector<typename VocType::Node> tree(VocType::Tree::levelBeg(nbLevels));
    cudaCheck(cudaMemcpy(tree.data(), voc.getTree(), sizeof(tree[0]) * tree.size(), cudaMemcpyDeviceToHost));
    const typename VocType::Node* nodesLastLevel = &tree.at(VocType::Tree::levelBeg(nbLevels - 1));
    std::vector<typename Traits::Descriptor> leafCentroids{VocType::Tree::levelSize(nbLevels)};
    for (uint32_t i = 0; i < leafCentroids.size(); i++){
        for (uint32_t j = 0; j < Traits::vecsPerDesc; j++) {
            leafCentroids[i][j][0] = nodesLastLevel[i / Traits::nbCenters][j][i % Traits::nbCenters];
        }
    }

    uint64_t inertia = 0;
    for (uint32_t i = 0; i < nbDesc; i++){
        inertia += squaredNorm(descriptors.at(i), leafCentroids.at(indicesInLevel.at(i)));
    }
    EXPECT_LE(std::sqrt(inertia/nbDesc), 0);// should be ~9.95
}


TEST(VocTest, perf)
{
    constexpr uint32_t nbLevels = 4;
    const uint32_t nbDesc = 1<<20;
    const uint32_t nbDoc = 100;
    const cudaStream_t stream = nullptr;
    std::mt19937_64 rng{3};
    std::uniform_int_distribution<uint64_t> dist{};

    std::vector<typename Traits::Descriptor, CudaManagedAllocator<typename Traits::Descriptor>> descriptors(nbDesc);
    for (auto& desc : descriptors){
        std::array<uint64_t, sizeof(desc)/sizeof(uint64_t)> val;
        std::generate(val.begin(), val.end(), [&](){return dist(rng);});
        memcpy(&desc, &val, sizeof(val));
    }

    cudaCheck(cudaMemPrefetchAsync(descriptors.data(), sizeof(descriptors[0]) * descriptors.size(), getCudaDevice(), stream));
    const auto voc = rbow::buildSiftVocabulary<4u>(descriptors.data(), nbDesc, nbDoc, nbLevels, stream);
    cudaCheck(cudaStreamSynchronize(stream));

    std::vector<uint32_t, CudaManagedAllocator<uint32_t>> indicesInLevel(nbDesc);
    cudaCheck(cudaMemPrefetchAsync(indicesInLevel.data(), sizeof(indicesInLevel[0]) * indicesInLevel.size(), getCudaDevice(), stream));
    voc.lookUp(reinterpret_cast<const std::array<std::byte, sizeof(typename Traits::Descriptor)> *>(descriptors.data()),
               nbDesc, indicesInLevel.data(), stream);
    cudaCheck(cudaStreamSynchronize(stream));

    cudaCheck(cudaMemPrefetchAsync(descriptors.data(), sizeof(descriptors[0]) * descriptors.size(), getCudaDevice(), stream));
    cudaCheck(cudaMemPrefetchAsync(indicesInLevel.data(), sizeof(indicesInLevel[0]) * indicesInLevel.size(), cudaCpuDeviceId, stream));
    cudaCheck(cudaStreamSynchronize(stream));
}
