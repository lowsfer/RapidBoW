//
// Created by yao on 10/19/19.
//

#pragma once
#include "kmeans.cuh"
#include <random>
#include <algorithm>
#include "DeviceKernelArg.h"
#include "BitSet.h"

class DescGrouping
{
public:
    using Traits = KMeansTraits;
    explicit DescGrouping(const std::vector<std::vector<uint32_t>>& src) {init(src);}
    void init(const std::vector<std::vector<uint32_t>>& src);
    uint32_t idxBeg(uint32_t idxGrp) const {
        return idxGrp == 0 ? 0 : roundUp(mIdxEnds.at(idxGrp - 1), Traits::tileSize);
    }
    uint32_t idxEnd(uint32_t idxGrp) const {
        return mIdxEnds.at(idxGrp);
    }
    uint32_t getNbGrps() const {
        return static_cast<uint32_t>(mIdxEnds.size());
    }
    uint32_t* getIndices() { return mIndices.data(); }
    const uint32_t* getIndices() const { return mIndices.data(); }
private:
    std::vector<uint32_t, CudaHostAllocator<uint32_t>> mIndices;
    std::vector<uint32_t> mIdxEnds;
};

class KMeans
{
public:
    using Traits = KMeansTraits;
    KMeans(const Traits::Descriptor* devDesc, uint32_t nbDesc, cudaStream_t stream)
    : mDescriptors{devDesc}
    , mNbDesc{nbDesc}
    , mStream{stream}
    {}
    cudaStream_t getStream() const {return mStream;}
    // Must set grouping and output for all tasks in this level. Called when entering a new level
    void setGrouping(const DescGrouping* grouping, KArray<uint8_t, Traits::tileSize>* idxNearestCenter,
            typename Traits::Centers* centersAll, BitSet<Traits::nbCenters>* validMask);
    //@fixme: should repeat k-means multiple times and choose the best clustering minimizing inertia
    //@fixme: implement kmeans++ to accelerate convergence
    // May compute only part of the tasks
    void compute(uint32_t idxGrpBeg, uint32_t nbGrps);
    void sync() const {cudaCheck(cudaStreamSynchronize(mStream));}

protected:
    void computeSome(uint32_t idxGrpBeg, uint32_t nbGrps);

    std::pair<std::vector<KArray<uint32_t, Traits::nbCenters>>, std::vector<BitSet<Traits::nbCenters>>>
        makeRandCenters(uint32_t idxGrpBeg, uint32_t nbGrps);

    std::vector<ArgsKMeansAssignCenter> makeKMeansArgs(uint32_t idxGrpBeg, uint32_t nbGrps);

protected:
    const typename Traits::Descriptor* mDescriptors;
    uint32_t mNbDesc;

    const DescGrouping* mGroups {nullptr};
    CudaMem<KArray<uint32_t, Traits::tileSize>, CudaMemType::kDevice> mIndices;
    // *All means data for all data at this tree level
    KArray<uint8_t, Traits::tileSize>* mIdxNearestCenterAll {nullptr};
    typename Traits::Centers* mCentersAll {nullptr};
    BitSet<Traits::nbCenters>* mValidMaskAll {nullptr};
#ifndef NDEBUG
    std::vector<Traits::Counters> mCounterAllDbg;
    std::vector<bool> mCounterAllDgbSet;
#endif
    CudaMem<typename Traits::DescTile, CudaMemType::kDevice> mDescTiles; // May only hold part of the groups (mMaxNbGrpsPerLaunch) in mGroups, for one computeSome call only.
    uint32_t mNbAllocatedTiles{0u};

    uint32_t mMaxNbGrpsPerLaunch = 128u;
    // Note that mCounter, mCenterAcc, mNbChanged and mNbChangedLast may all be zero for most groups after stopping, because most of them converged and stopped early.
    CudaMem<typename Traits::Counters, CudaMemType::kDevice> mCounter
        = allocCudaMem<typename Traits::Counters, CudaMemType::kDevice>(mMaxNbGrpsPerLaunch);
    CudaMem<typename Traits::GlbCenterAcc, CudaMemType::kDevice> mCenterAcc
        = allocCudaMem<typename Traits::GlbCenterAcc, CudaMemType::kDevice>(mMaxNbGrpsPerLaunch);
    CudaMem<uint32_t, CudaMemType::kDevice> mNbChanged
        = allocCudaMem<uint32_t, CudaMemType::kDevice>(mMaxNbGrpsPerLaunch);
    CudaMem<uint32_t, CudaMemType::kDevice> mNbChangedLast
        = allocCudaMem<uint32_t, CudaMemType::kDevice>(mMaxNbGrpsPerLaunch);

    CudaMem<uint32_t, CudaMemType::kPinned> mNbChangedHostCopy
            = allocCudaMem<uint32_t, CudaMemType::kPinned>(mMaxNbGrpsPerLaunch);

    cudaStream_t mStream {nullptr};
    DeviceKernelArg<KArray<uint32_t, Traits::nbCenters>> mCenterInitArgs;
    DeviceKernelArg<ArgsKMeansAssignCenter> mKMeansIterArgs;
#ifndef NDEBUG
    uint32_t mConvergenceCheckInterval = 1;
#else
    uint32_t mConvergenceCheckInterval = 4;
#endif
    uint32_t mConvergeMaxNbChanges = 0u; // convergence threshold
    uint32_t mMaxNbIterations = 1000u;
};
