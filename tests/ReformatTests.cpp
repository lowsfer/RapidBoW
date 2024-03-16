//
// Created by yao on 10/19/19.
//

#include <gtest/gtest.h>
#include <cuda_utils.h>
#include <cstdint>
#include <random>
#include "../kmeans.h"

template <uint32_t tileSize>
class ReformatTestTemplate : public testing::Test
{
public:
    using Traits = KMeansTraits;
    using Descriptor = typename Traits::Descriptor;
    using DescTile = KArray<typename Traits::Vec, Traits::vecsPerDesc, tileSize>;

    void test(size_t nbDesc = 136, size_t nbTiles = 3, bool refCheck = true);
protected:
    std::vector<Descriptor, CudaManagedAllocator<Descriptor>> mDescriptors;
    std::vector<KArray<uint32_t, tileSize>, CudaManagedAllocator<KArray<uint32_t, tileSize>>> mIndices;
    std::vector<DescTile, CudaManagedAllocator<DescTile>> mTiles;
};


void launchReformatDesc(typename KMeansTraits::DescTile *__restrict__ tiles,
                        const typename KMeansTraits::Descriptor *__restrict__ descriptors,
                        const KArray<uint32_t, KMeansTraits::tileSize> *__restrict__ indices, // desc indices for each tile
                        uint32_t nbTiles, cudaStream_t stream);
void launchKMeansInitCenters(typename KMeansTraits::Centers*__restrict__ tiles,
                             const typename KMeansTraits::Descriptor *__restrict__ descriptors,
                             const KArray<uint32_t, KMeansTraits::nbCenters> *__restrict__ indices, // desc indices for each tile
                             uint32_t nbTiles, cudaStream_t stream);

template <uint32_t tileSize>
void ReformatTestTemplate<tileSize>::test(size_t nbDesc, size_t nbTiles, bool refCheck){
    std::mt19937_64 rng{};
    std::uniform_int_distribution<uint64_t> dist64{};
    std::uniform_int_distribution<uint32_t> distIdx{0u, uint32_t(nbDesc - 1)};
    std::bernoulli_distribution distBin{0.125f};

    mDescriptors.resize(nbDesc);
    std::generate(mDescriptors.begin(), mDescriptors.end(), [&](){
        Descriptor desc{};
        std::generate_n(reinterpret_cast<uint64_t*>(&desc), sizeof(desc) / sizeof(uint64_t), [&](){return dist64(rng);});
        return desc;
    });
    mIndices.resize(nbTiles);
    mTiles.resize(nbTiles);

    for (auto& tileIndices : mIndices){
        for (auto& idx : tileIndices){
            idx = distBin(rng) ? idxInvalid : distIdx(rng);
        }
    }

    cudaStream_t stream = nullptr;
    const int device = getCudaDevice();
    cudaCheck(cudaMemPrefetchAsync(mTiles.data(), sizeof(mTiles[0]) * mTiles.size(), device, stream));
    cudaCheck(cudaMemPrefetchAsync(mDescriptors.data(), sizeof(mDescriptors[0]) * mDescriptors.size(), device, stream));
    cudaCheck(cudaMemPrefetchAsync(mIndices.data(), sizeof(mIndices[0]) * mIndices.size(), device, stream));

    if constexpr(tileSize == Traits::tileSize){
        launchReformatDesc(mTiles.data(), mDescriptors.data(), mIndices.data(), nbTiles, stream);
    }
    else{
        static_assert(tileSize == Traits::nbCenters);
        launchKMeansInitCenters(mTiles.data(), mDescriptors.data(), mIndices.data(), nbTiles, stream);
    }
    cudaCheck(cudaStreamSynchronize(stream));

    if (refCheck) {
        Traits::Vec invalidVec{{~0u}};
        for (unsigned i = 0; i < nbTiles; i++) {
            for (unsigned j = 0; j < tileSize; j++) {
                for (unsigned k = 0; k < Traits::vecsPerDesc; k++) {
                    const auto idxDesc = mIndices[i][j];
                    EXPECT_EQ(mTiles[i].data[k][j], idxDesc == idxInvalid ? invalidVec : mDescriptors[idxDesc][k]);
                }
            }
        }
    }
}

using ReformatTest = ReformatTestTemplate<KMeansTraits::tileSize>;

TEST_F(ReformatTest, func){
    test(136, 3, true);
}

TEST_F(ReformatTest, perf){
    test(1000000, 31250, false);
}

using CenterInitTest = ReformatTestTemplate<KMeansTraits::nbCenters>;

TEST_F(CenterInitTest, func){
    test(136, 3, true);
}

TEST_F(CenterInitTest, perf){
    test(1000000, 31250, false);
}

