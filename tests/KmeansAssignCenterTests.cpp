//
// Created by yao on 10/2/19.
//

#include <gtest/gtest.h>
#include <cuda_utils.h>
#include <cstdint>
#include <random>
#include "../kmeans.h"

class KMeansAssignCenterTestTask
{
public:
    void SetUp(){
        mIdxNearestCenterRef = std::vector<uint8_t>(mNbDesc, uint8_t{255});
        mCounterRef = std::vector<uint32_t>(mNbCenters, 0u);
        mCenterAccRef = std::vector<uint32_t>(mDescDims * mNbCenters, 0u);
        mNbChangedRef = 0;
    }
    void init(uint32_t nbDesc, cudaStream_t stream) {
        mStream = stream;
        mNbDesc = nbDesc;
        mNbTiles = (mNbDesc + mTileSize - 1) / mTileSize;
        mIdxNearestCenterRef = std::vector<uint8_t>(mNbDesc, uint8_t{255});

        const auto descElems = mDescVecs * mTileSize * mNbTiles;
        mDescTiles = allocCudaMem<uint32_t, CudaMemType::kManaged>(descElems);
        const auto centerElems = mDescVecs * mNbCenters;
        mCenters = allocCudaMem<uint32_t, CudaMemType::kManaged>(centerElems);
        mValidMask = allocCudaMem<BitSet<mNbCenters>, CudaMemType::kManaged, true>(1);
        for(uint32_t i = 0; i < mNbCenters; i++)
            mValidMask[0].set(i, i < nbDesc);
        std::mt19937_64 rng{};//{std::random_device{}()};
        std::uniform_int_distribution<uint64_t> dist{};
        for (uint32_t i = 0; i < descElems / 2; i++){
            reinterpret_cast<uint64_t*>(mDescTiles.get())[i] = rng();// dist(rng);
        }
        //@fixme: pick centers from descriptors
        using Traits = KMeansTraits;
        typename Traits::Centers& centers = *reinterpret_cast<typename Traits::Centers*>(mCenters.get());
        typename Traits::DescTile* descTiles = reinterpret_cast<typename Traits::DescTile*>(mDescTiles.get());
        for (uint32_t i = 0; i < mNbCenters; i++){
            for (uint32_t j = 0; j < Traits::vecsPerDesc; j++){
                const uint32_t idxDesc = std::min(i, nbDesc - 1);
                centers[j][i] = descTiles[idxDesc / Traits::tileSize][j][idxDesc % Traits::tileSize];
            }
        }
        mIdxNearestCenter = allocCudaMem<uint8_t, CudaMemType::kManaged>(mTileSize * mNbTiles);
        for (uint32_t i = 0; i < mNbDesc; i++){
            mIdxNearestCenter[i] = mIdxNearestCenterRef.at(i) = uint8_t(dist(rng) % mNbCenters);
        }
        mCounter = allocCudaMem<uint32_t, CudaMemType::kManaged>(mNbCenters);
        mCenterAcc = allocCudaMem<uint32_t, CudaMemType::kManaged>(mDescDims * mNbCenters);
        mNbChanged = allocCudaMem<uint32_t, CudaMemType::kManaged>(2);
        mNbChanged[1] = mNbDesc;
    }
    void TearDown() {
        cudaCheck(cudaDeviceSynchronize());
        mDescTiles.reset();
        mCenters.reset();
        mIdxNearestCenter.reset();
        mCounter.reset();
        mCenterAcc.reset();
    }
    void computeRef();
    void migrateToDevice(){
        int device;
        cudaCheck(cudaGetDevice(&device));
        cudaCheck(cudaMemPrefetchAsync(mDescTiles.get(), sizeof(mDescTiles[0]) * mDescVecs * mTileSize * mNbTiles, device, mStream));
        cudaCheck(cudaMemPrefetchAsync(mCenters.get(), sizeof(mCenters[0]) * mDescVecs * mNbCenters, device, mStream));
        cudaCheck(cudaMemPrefetchAsync(mCounter.get(), sizeof(mCounter[0]) * mNbCenters, device, mStream));
        cudaCheck(cudaMemPrefetchAsync(mIdxNearestCenter.get(), sizeof(mIdxNearestCenter[0]) * mTileSize * mNbTiles, device, mStream));
        cudaCheck(cudaMemPrefetchAsync(mCenterAcc.get(), sizeof(mCenterAcc[0]) * mNbCenters * mDescDims, device, mStream));
        cudaCheck(cudaMemPrefetchAsync(mNbChanged.get(), sizeof(mNbChanged[0]), device, mStream));
    }

    ArgsKMeansAssignCenter makeTaskArgs() const{
        using Traits = KMeansTraits;
        ArgsKMeansAssignCenter args{
            reinterpret_cast<const typename Traits::DescTile*>(mDescTiles.get()),
            mNbTiles, mNbDesc, reinterpret_cast<const typename Traits::Centers*>(mCenters.get()),
            mIdxNearestCenter.get(),
            reinterpret_cast<typename Traits::Counters*>(mCounter.get()),
            reinterpret_cast<typename Traits::GlbCenterAcc*>(mCenterAcc.get()), mNbChanged.get(), mNbChanged.get() + 1,
            mValidMask.get()
        };
        return args;
    }

    // call computeRef and check.
    void refCheck();

    void refCheckUpdateCenter(){
        using Traits = KMeansTraits;
        for (uint32_t idxElem = 0; idxElem < Traits::descDims; idxElem++){
            for (uint32_t idxCenter = 0; idxCenter < Traits::nbCenters; idxCenter++){
                if (mCounter[idxCenter] == 0){
                    continue;
                }
                constexpr uint32_t elemsPerVec = Traits::elemsPerWord * Traits::wordsPerVec;
                const uint8_t val = reinterpret_cast<const uint8_t(&)[elemsPerVec]>(reinterpret_cast<const typename Traits::Centers&>(*mCenters.get())[idxElem / elemsPerVec][idxCenter])[idxElem % elemsPerVec];
                const uint8_t ref = (uint8_t)std::round(float(reinterpret_cast<typename Traits::GlbCenterAcc&>(*mCenterAcc.get())[idxElem][idxCenter]) / mCounter[idxCenter]);
                EXPECT_EQ(val, ref);
            }
        }
    }

protected:
    static constexpr uint32_t mDescDims = 128;
    static constexpr uint32_t mDescVecs = 32;
    uint32_t mNbDesc = 123;
//    uint32_t mNbDesc = 4u<<20;
    static constexpr uint32_t mTileSize = 32;
    uint32_t mNbTiles = (mNbDesc + mTileSize - 1) / mTileSize;
    CudaMem<uint32_t, CudaMemType::kManaged> mDescTiles;
    static constexpr uint32_t mNbCenters = 16;
    CudaMem<uint32_t, CudaMemType::kManaged> mCenters; // centers are transposed as uint32_t[mDescVecs][mNbCenters]
    CudaMem<BitSet<mNbCenters>, CudaMemType::kManaged> mValidMask;
    std::vector<uint8_t> mIdxNearestCenterRef = std::vector<uint8_t>(mNbDesc, uint8_t{255});
    std::vector<uint32_t> mCounterRef = std::vector<uint32_t>(mNbCenters, 0u);
    std::vector<uint32_t> mCenterAccRef = std::vector<uint32_t>(mDescDims * mNbCenters, 0u);
    uint32_t mNbChangedRef = 0u;
    CudaMem<uint8_t, CudaMemType::kManaged> mIdxNearestCenter;
    CudaMem<uint32_t, CudaMemType::kManaged> mCounter;
    CudaMem<uint32_t, CudaMemType::kManaged> mCenterAcc;
    CudaMem<uint32_t, CudaMemType::kManaged> mNbChanged;
    cudaStream_t mStream = nullptr;
};

void KMeansAssignCenterTestTask::computeRef() {
    for (uint32_t idxDesc = 0; idxDesc < mNbDesc; idxDesc++){
        const uint32_t idxTile = idxDesc / mTileSize;
        const uint32_t idxTileDesc = idxDesc % mTileSize;
        uint32_t distance[mNbCenters]{};
        for (uint32_t idxVec = 0; idxVec < mDescVecs; idxVec++){
            const uint32_t vec = mDescTiles[mDescVecs * mTileSize * idxTile + mTileSize * idxVec + idxTileDesc];
            for (uint32_t idxCenter = 0; idxCenter < mNbCenters; idxCenter++){
                const uint32_t vecCenter = mCenters[mNbCenters * idxVec + idxCenter];
                for (int i = 0; i < 4; i++){
                    const int a = reinterpret_cast<const uint8_t (&)[4]>(vec)[i];
                    const int b = reinterpret_cast<const uint8_t (&)[4]>(vecCenter)[i];
//                    if (idxDesc == 1 && idxCenter == 0){
//                        printf("%u - %u\n", a, b);
//                    }
                    const int diff = a - b;
                    distance[idxCenter] += uint32_t(diff*diff);
                }
            }
        }
//        for (uint32_t i = 0; i < mNbCenters; i++){
//            printf("\t%u", distance[i]);
//        }
//        printf("\n");
        const auto iter_min = std::min_element(std::begin(distance), std::end(distance));
        const uint32_t idxNearest = iter_min - std::begin(distance);
        if (mIdxNearestCenterRef.at(idxDesc) != idxNearest){
            mNbChangedRef++;
        }
        mIdxNearestCenterRef.at(idxDesc) = idxNearest;
        mCounterRef.at(idxNearest)++;
        constexpr uint32_t elemsPerVec = mDescDims / mDescVecs;
        for (uint32_t idxVec = 0; idxVec < mDescVecs; idxVec++) {
            const uint32_t vec = mDescTiles[mDescVecs * mTileSize * idxTile + mTileSize * idxVec + idxTileDesc];
            for (uint32_t i = 0; i < elemsPerVec; i++) {
                const uint32_t a = reinterpret_cast<const uint8_t (&)[4]>(vec)[i];
                const uint32_t idxElem = idxVec * elemsPerVec + i;
                mCenterAccRef.at(mNbCenters * idxElem + idxNearest) += a;
            }
        }
    }
//    printf("_______________________________________________________________\n");
}

void KMeansAssignCenterTestTask::refCheck() {
    computeRef();
    for (uint32_t i = 0; i < mNbDesc; i++){
        EXPECT_EQ(mIdxNearestCenter.get()[i], mIdxNearestCenterRef.at(i));
    }
    for (uint32_t i = 0; i < mNbCenters; i++) {
        EXPECT_EQ(mCounter.get()[i], mCounterRef.at(i));
    }
    for (uint32_t idxElem = 0; idxElem < mDescDims; idxElem++){
        for (uint32_t idxCenter = 0; idxCenter < mNbCenters; idxCenter++){
            const auto idx = idxElem * mNbCenters + idxCenter;
            EXPECT_EQ(mCenterAcc[idx], mCenterAccRef.at(idx));
        }
    }
    EXPECT_EQ(mNbChanged[0], mNbChangedRef);
}

class KMeansAssignCenterTest : public testing::Test
{
public:
    void SetUp() override {
        for (auto& task : mTasks){
            task.SetUp();
        }
    }
    void TearDown() override{
        for (auto& task : mTasks){
            task.TearDown();
        }
    }
    void migrateToDevice() {
        for (auto& task : mTasks)
            task.migrateToDevice();
        const int device = getCudaDevice();
        cudaCheck(cudaMemPrefetchAsync(mArgs.data(), sizeof(mArgs[0]) * nbTasks, device, mStream));
    }
    static constexpr uint32_t nbTasks = 4;
    void init(std::array<uint32_t, nbTasks> nbDesc){
        for (uint32_t i = 0; i < nbTasks; i++){
            mTasks.at(i).init(nbDesc.at(i), mStream);
        }
        mArgs.resize(nbTasks);
        for (uint32_t i = 0; i < nbTasks; i++){
            mArgs.at(i) = mTasks.at(i).makeTaskArgs();
        }
    }
    void computeRef(){
        for (auto& task : mTasks)
            task.computeRef();
    }
    void compute(){
        launchAssignCenter(mArgs.data(), nbTasks, mStream);
    }
    void refCheck(){
        cudaCheck(cudaStreamSynchronize(mStream));
        for (uint32_t i = 0; i < nbTasks; i++){
            mTasks.at(i).refCheck();
        }
    }

    void computeUpdateCenter(){
        launchKMeansUpdateCenters(mArgs.data(), nbTasks, mStream);
    }
    void refCheckUpdateCenter(){
        cudaCheck(cudaStreamSynchronize(mStream));
        for (uint32_t i = 0; i < nbTasks; i++){
            mTasks.at(i).refCheckUpdateCenter();
        }
    }
protected:
    std::array<KMeansAssignCenterTestTask, nbTasks> mTasks;
    std::vector<ArgsKMeansAssignCenter, CudaManagedAllocator<ArgsKMeansAssignCenter>> mArgs;
    cudaStream_t mStream{nullptr};
};

TEST_F(KMeansAssignCenterTest, random)
{
    init({478u, 15u, 39u, 134u});
    migrateToDevice();
    compute();
    refCheck();
}

TEST_F(KMeansAssignCenterTest, perf)
{
    init({2u<<20, 2u<<20, 2u<<20, 2u<<20});
    migrateToDevice();

    for (int i = 0; i < 32; i++) {
        compute();
    }
    cudaCheck(cudaStreamSynchronize(mStream));
}

TEST_F(KMeansAssignCenterTest, updateCenter)
{
    init({478u, 15u, 39u, 134u});
    migrateToDevice();
    compute();
    computeUpdateCenter();
    refCheckUpdateCenter();
}