//
// Created by yao on 10/20/19.
//

#include <unordered_set>
#include "kmeans.h"

using Traits = KMeansTraits;

void DescGrouping::init(const std::vector<std::vector<uint32_t>>& src){
    mIndices.clear();
    mIdxEnds.clear(); mIdxEnds.reserve(src.size());
    const size_t nbIndicesRoundedUp = std::accumulate(src.begin(), src.end(), size_t{0u},
                                                      [](uint32_t a, const std::vector<uint32_t>& b){return a + roundUp(b.size(), Traits::tileSize);});
    mIndices.reserve(nbIndicesRoundedUp);
    for (const auto& g : src){
        const uint32_t beg = mIdxEnds.empty() ? 0u : roundUp(mIdxEnds.back(), Traits::tileSize);
        mIdxEnds.push_back(beg + g.size());
        assert(mIndices.begin() + beg == mIndices.end());
        mIndices.insert(mIndices.end(), g.begin(), g.end());
        mIndices.insert(mIndices.end(), roundUp(g.size(), Traits::tileSize) - g.size(), idxInvalid);
        assert(mIndices.size() == roundUp(mIdxEnds.back(), Traits::tileSize));
    }
    assert(nbIndicesRoundedUp == roundUp(mIdxEnds.back(), Traits::tileSize));
}

// Must set all grouping
void KMeans::setGrouping(const DescGrouping* grouping, KArray<uint8_t, Traits::tileSize>* idxNearestCenter,
        typename Traits::Centers* centersAll, BitSet<Traits::nbCenters>* validMask){
    mGroups = grouping;
    const uint32_t nbTiles = divUp(grouping->idxEnd(grouping->getNbGrps() - 1u), Traits::tileSize); // rounded-up indices
    mIndices = allocCudaMem<KArray<uint32_t, Traits::tileSize>, CudaMemType::kDevice>(nbTiles);
    cudaCheck(cudaMemcpyAsync(mIndices.get(), grouping->getIndices(), sizeof(uint32_t) * Traits::tileSize * nbTiles, cudaMemcpyDeviceToDevice, mStream));
    mIdxNearestCenterAll = idxNearestCenter;
    mCentersAll = centersAll;
    mValidMaskAll = validMask;
#ifndef NDEBUG
    mCounterAllDbg.assign(grouping->getNbGrps(), {});
    mCounterAllDgbSet.assign(grouping->getNbGrps(), false);
#endif
}
// May compute only part of the tasks
void KMeans::compute(uint32_t idxGrpBeg, uint32_t nbGrps){
    uint32_t nbFinished = 0;
    while (nbFinished < nbGrps){
        const uint32_t batchSize = std::min(nbGrps - nbFinished, mMaxNbGrpsPerLaunch);
        computeSome(idxGrpBeg + nbFinished, batchSize);
        nbFinished += batchSize;
    }
}


void launchReformatDesc(typename KMeansTraits::DescTile *__restrict__ tiles,
                        const typename KMeansTraits::Descriptor *__restrict__ descriptors,
                        const KArray<uint32_t, KMeansTraits::tileSize> *__restrict__ indices, // desc indices for each tile
                        uint32_t nbTiles, cudaStream_t stream);
void launchKMeansInitCenters(typename KMeansTraits::Centers*__restrict__ tiles,
                             const typename KMeansTraits::Descriptor *__restrict__ descriptors,
                             const KArray<uint32_t, KMeansTraits::nbCenters> *__restrict__ indices, // desc indices for each tile
                             uint32_t nbTiles, cudaStream_t stream);

void KMeans::computeSome(uint32_t idxGrpBeg, uint32_t nbGrps){
    if (nbGrps > mMaxNbGrpsPerLaunch){
        throw std::runtime_error("Too many tasks in one kernel");
    }
    const uint32_t nbRequiredTiles = (roundUp(mGroups->idxEnd(idxGrpBeg + nbGrps - 1), Traits::tileSize) - mGroups->idxBeg(idxGrpBeg)) / Traits::tileSize;
    if (nbRequiredTiles > mNbAllocatedTiles){
        const uint32_t nbTiledToAllocate = std::max(nbRequiredTiles, mNbAllocatedTiles * 2);
        mDescTiles = allocCudaMem<typename Traits::DescTile, CudaMemType::kDevice>(nbTiledToAllocate);
        mNbAllocatedTiles = nbTiledToAllocate;
    }
    const uint32_t idxTileBeg = mGroups->idxBeg(idxGrpBeg) / Traits::tileSize;
    launchReformatDesc(mDescTiles.get(), mDescriptors, &mIndices[idxTileBeg], nbRequiredTiles, mStream);
    cudaCheck(cudaMemsetAsync(mNbChanged.get(), 255, sizeof(mNbChanged[0]) * mMaxNbGrpsPerLaunch, mStream));

    // set up centers
    const auto [centerIndices, centerValidMask] = makeRandCenters(idxGrpBeg, nbGrps);
    cudaCheck(cudaMemcpyAsync(&mValidMaskAll[idxGrpBeg], centerValidMask.data(), sizeof(centerValidMask[0]) * nbGrps, cudaMemcpyHostToDevice, mStream));
    mCenterInitArgs.dispatch(
            [this, idxGrpBeg, nbGrps](const KArray<uint32_t, Traits::nbCenters>* indices, cudaStream_t stream){
                launchKMeansInitCenters(mCentersAll + idxGrpBeg, mDescriptors, indices, nbGrps, stream);
            },
            centerIndices.data(), size_t(nbGrps), mStream);
    // init kmeans args
    const auto kmeansArgs = makeKMeansArgs(idxGrpBeg, nbGrps);
    mKMeansIterArgs.asyncAllocAndFill(kmeansArgs.data(), nbGrps, mStream);
    const auto delayedFree = makeScopeGuard([&](){mKMeansIterArgs.asyncFree(mStream);});
    auto checkConvergence = [&](){
        cudaCheck(cudaMemcpyAsync(mNbChangedHostCopy.get(), mNbChanged.get(), sizeof(mNbChanged[0]) * nbGrps, cudaMemcpyDeviceToHost, mStream));
        cudaCheck(cudaStreamSynchronize(mStream));
#ifndef NDEBUG
        for (uint32_t i = 0; i < nbGrps; i++){
            if (mNbChangedHostCopy.get()[i] <= mConvergeMaxNbChanges && !mCounterAllDgbSet.at(idxGrpBeg + i)){
                mCounterAllDbg.at(idxGrpBeg + i) = mCounter[i];
                mCounterAllDgbSet.at(idxGrpBeg + i) = true;
            }
        }
#endif
        return std::count_if(mNbChangedHostCopy.get(), mNbChangedHostCopy.get() + nbGrps, [this](uint32_t nbChanged){ return nbChanged <= mConvergeMaxNbChanges;});
    };
    for (uint32_t i = 0; i < mMaxNbIterations; i++){
        cudaCheck(cudaMemsetAsync(mCounter.get(), 0, sizeof(mCounter[0]) * mMaxNbGrpsPerLaunch, mStream));// fixme: only resets unconverged ones. May use a kernel to do all four
        cudaCheck(cudaMemsetAsync(mCenterAcc.get(), 0, sizeof(mCenterAcc[0]) * mMaxNbGrpsPerLaunch, mStream));
        cudaCheck(cudaMemcpyAsync(mNbChangedLast.get(), mNbChanged.get(), sizeof(mNbChanged[0]) * mMaxNbGrpsPerLaunch, cudaMemcpyDeviceToDevice, mStream));
        cudaCheck(cudaMemsetAsync(mNbChanged.get(), 0, sizeof(mNbChanged[0]) * mMaxNbGrpsPerLaunch, mStream));

        launchAssignCenter(mKMeansIterArgs.getDevicePtr(), nbGrps, mStream);
        if (i % mConvergenceCheckInterval == mConvergenceCheckInterval - 1u){
            const auto nbConverged = checkConvergence();
            if (nbConverged == nbGrps){
//                printf("Converged with %u iterations\n", i);
                break;
            }
        }
        launchKMeansUpdateCenters(mKMeansIterArgs.getDevicePtr(), nbGrps, mStream);
    }
    cudaCheck(cudaDeviceSynchronize());
    const auto nbConverged = checkConvergence();
    if (nbConverged != nbGrps){
        printf("Warning: only %lu/%u kmeans tasks converged after %u iterations.\n", nbConverged, nbGrps, mMaxNbIterations);
    }
}

uint32_t randPick(uint32_t* dst, uint32_t randSpace, uint32_t nbToPick)
{
    const uint32_t nbPick = std::min(randSpace, nbToPick);
//    thread_local static std::mt19937_64 rng{std::random_device{}()};
    thread_local static std::mt19937_64 rng{};
    thread_local static std::unordered_set<uint32_t> results;
    results.clear();
    if (nbPick * 16 < randSpace){
        while(results.size() != nbPick){
            results.insert(rng() % randSpace);
        }
        std::copy_n(results.begin(), nbPick, dst);
        std::sort(dst, dst + nbPick);
    }
    else {
        std::vector<uint32_t> space(randSpace);
        std::iota(space.begin(), space.end(), 0u);
        std::shuffle(space.begin(), space.end(), rng);
        std::copy_n(space.begin(), nbPick, dst);
        std::sort(dst, dst + nbPick);
    }
    if (nbPick < nbToPick){
        //@info: the kmeans kernel guaranatees that when multiple centers are identical, descriptors are assigned to the first one.
        //@info: when randSpace and nbPick is 0, the value is not important, as there is no descriptor for clusterring. So we set it to 0.
        std::fill(dst+nbPick, dst+nbToPick, nbPick != 0 ? dst[nbPick - 1u] : 0u);
    }
    return nbPick;
}

std::pair<std::vector<KArray<uint32_t, Traits::nbCenters>>, std::vector<BitSet<Traits::nbCenters>>>
        KMeans::makeRandCenters(uint32_t idxGrpBeg, uint32_t nbGrps){
//    thread_local std::mt19937_64 rng{std::random_device{}()};
    thread_local static std::mt19937_64 rng{};
    thread_local static std::uniform_int_distribution<uint64_t> dist{};
    std::vector<KArray<uint32_t, Traits::nbCenters>> centerIndices;
    std::vector<BitSet<Traits::nbCenters>> centerValidMask;
    centerIndices.reserve(nbGrps); centerValidMask.reserve(nbGrps);
    for (uint32_t idxGrp = idxGrpBeg; idxGrp < idxGrpBeg + nbGrps; idxGrp++){
        const uint32_t grpNbDesc = mGroups->idxEnd(idxGrp) - mGroups->idxBeg(idxGrp);
        const uint32_t* grpIndices = &mGroups->getIndices()[mGroups->idxBeg(idxGrp)];
        std::array<uint32_t, Traits::nbCenters> tmp;
        //@fixme: implement kmeans++ for faster convergence. In case there are a lot of descriptors, to accelerate kmeans++ init, we can first randomly pick 256 descriptors, and use kmeans++ to pick centers from these 256 descriptors.
        // When we implement kmeans++, make sure we make centers non-duplicate. Currently we may have identical centers.
        const uint32_t nbValidCenters = randPick(tmp.begin(), grpNbDesc, Traits::nbCenters);
        KArray<uint32_t, Traits::nbCenters> descIndices;
        std::transform(tmp.begin(), tmp.end(), descIndices.begin(), [grpIndices](uint32_t idxInGrpIndices){return grpIndices[idxInGrpIndices];});
        centerIndices.emplace_back(descIndices);
        static_assert(sizeof(centerValidMask[0]) == sizeof(uint16_t));
        const uint16_t mask = (std::numeric_limits<uint16_t>::max() >> (Traits::nbCenters - nbValidCenters));
        centerValidMask.emplace_back(reinterpret_cast<const BitSet<Traits::nbCenters>&>(mask));
        for (uint32_t i = 0; i < Traits::nbCenters; i++){
            assert((i < nbValidCenters) == centerValidMask.back().test(i));
        }
    }
    return std::make_pair(std::move(centerIndices), std::move(centerValidMask));
}

std::vector<ArgsKMeansAssignCenter> KMeans::makeKMeansArgs(uint32_t idxGrpBeg, uint32_t nbGrps){
    std::vector<ArgsKMeansAssignCenter> tasks(nbGrps);
    for (uint32_t i = 0; i < nbGrps; i++){
        const uint32_t idxGrp = idxGrpBeg + i;
        const auto idxDescBegGlobal = mGroups->idxBeg(idxGrp);
        const auto idxDescBegLocal = idxDescBegGlobal - mGroups->idxBeg(idxGrpBeg);
        const auto idxDescEndGlobal = mGroups->idxEnd(idxGrp);
        const auto idxTileBegGlobal = idxDescBegGlobal / Traits::tileSize;
        const auto idxTileBegLocal = idxDescBegLocal / Traits::tileSize;
        const auto idxTileEndGlobal = divUp(idxDescEndGlobal, Traits::tileSize);
        const auto nbDescInGrp = mGroups->idxEnd(idxGrp) - mGroups->idxBeg(idxGrp);
        const auto nbTilesInGrp = idxTileEndGlobal - idxTileBegGlobal;
        tasks.at(i) = ArgsKMeansAssignCenter{
                &mDescTiles[idxTileBegLocal],
                nbTilesInGrp,
                nbDescInGrp,
                &mCentersAll[idxGrpBeg + i],
                &mIdxNearestCenterAll[idxTileBegGlobal][0],
                &mCounter[i],
                &mCenterAcc[i],
                &mNbChanged[i],
                &mNbChangedLast[i],
                &mValidMaskAll[idxGrpBeg + i],
        };
    }
    return tasks;
}

