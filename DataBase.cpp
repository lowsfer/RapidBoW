//
// Created by yao on 10/26/19.
//

#include <numeric>
#include <algorithm>
#include "DataBase.h"
#include <cmath>
#include <VectorMap.h>

namespace rbow {

static inline float getWeight(const std::array<float, DataBase::branchFactor> *levelWeights,
        // indicesInLevel = idxInLeafLevel/branchFactor; idxBranch =idxInLeafLevel%branchFactor
        uint32_t idxInNextLevel){
//        constexpr auto branchFactor = DataBase::branchFactor;
//        return levelWeights[idxInNextLevel/ branchFactor][idxInNextLevel % branchFactor];
        return levelWeights[0].data()[idxInNextLevel];
}


void DataBase::addDoc(uint32_t idxDoc, const uint32_t *hostIndicesInLeafLevel, uint32_t nbDesc) {
    std::lock_guard<std::shared_mutex> lk{mIvFLock}; // need exclusive locking
    if (mWeightVecRcpL1Norm.find(idxDoc) != mWeightVecRcpL1Norm.end()){
        throw std::runtime_error("Duplicate document index");
    }
    auto insertIvFEntry = [](InvertedFile& ivf, IdxDoc docIdx){
        const auto iter = std::upper_bound(ivf.begin(), ivf.end(), docIdx);
        ivf.insert(iter, docIdx); // ivf is assumed to be small, so high std::vector complexity here should be fine.
        assert(std::is_sorted(ivf.begin(), ivf.end()));
    };
    for (uint32_t i = 0; i < nbDesc; i++) {
        const uint32_t idx = hostIndicesInLeafLevel[i];
//        insertIvFEntry(mLastLvlIvF.at(idx / branchFactor)[idx % branchFactor], idxDoc);
        insertIvFEntry(mLastLvlIvF[0][idx], idxDoc);
    }
    const uint32_t nbLevels = mVoc.getNbLevels();
    double weightL1Norm = 0;
    std::vector<uint32_t> indicesInNextLevel{hostIndicesInLeafLevel, hostIndicesInLeafLevel+nbDesc};
    std::sort(indicesInNextLevel.begin(), indicesInNextLevel.end()); // Optional for L1 norm. We don't have to aggregate same indices together.
    for (uint32_t idxIter = 0; idxIter < nbLevels; idxIter++) {
        const uint32_t idxLevel = nbLevels - 1u - idxIter;
        const auto lvlWeights = static_cast<const std::array<float, branchFactor> *>(mVoc.getLevelWeights(idxLevel));
        for (uint32_t idxDesc = 0; idxDesc < nbDesc; idxDesc++) {
            const float w = getWeight(lvlWeights, indicesInNextLevel[idxDesc]);
            assert(w >= 0);
            weightL1Norm += w;
            indicesInNextLevel[idxDesc] /= branchFactor;
        }
    }
    mWeightVecRcpL1Norm[idxDoc] = float(1/weightL1Norm);
}

std::vector<uint32_t>
DataBase::query(const uint32_t *hostIndicesInLeafLevel, uint32_t nbDesc, uint32_t maxNbResults) const {
    std::shared_lock<std::shared_mutex> lk{mIvFLock};

    const uint32_t nbLevels = mVoc.getNbLevels();

    const float queryWeightRcpL1Norm = [&](){
        double weightL1Norm = 0;
        std::vector<uint32_t> indicesInNextLevel{hostIndicesInLeafLevel, hostIndicesInLeafLevel+nbDesc};
        std::sort(indicesInNextLevel.begin(), indicesInNextLevel.end()); // Optional for L1 norm. We don't have to aggregate same indices together.
        for (uint32_t idxIter = 0; idxIter < nbLevels; idxIter++) {
            const uint32_t idxLevel = nbLevels - 1u - idxIter;
            const auto lvlWeights = static_cast<const std::array<float, branchFactor> *>(mVoc.getLevelWeights(idxLevel));
            for (uint32_t idxDesc = 0; idxDesc < nbDesc; idxDesc++) {
                const float w = getWeight(lvlWeights, indicesInNextLevel[idxDesc]);
//                assert(w > 0); // we have a bug WAR that may cause w == 0.
                weightL1Norm += w;
                assert(std::isfinite(weightL1Norm));
                indicesInNextLevel[idxDesc] /= branchFactor;
            }
        }
        assert(std::isfinite(weightL1Norm));
        return float(1 / weightL1Norm);
    }();

    VectorMap<IdxDoc, double> distanceL1;
    const InvertedFile* leafInvertedFiles = &mLastLvlIvF[0][0];
    std::vector<uint32_t> indicesInLeafLevel{hostIndicesInLeafLevel, hostIndicesInLeafLevel+nbDesc};
    std::sort(indicesInLeafLevel.begin(), indicesInLeafLevel.end());// @fixme: May use unordered_map . We only need aggregation, not order.
    std::vector<uint32_t> indicesInNextLevel = indicesInLeafLevel;
//    double tmp[2]{};
//    double tmpNoNorm{};
    for (uint32_t idxIter = 0; idxIter < nbLevels; idxIter++) {
        const uint32_t idxLevel = nbLevels - 1u - idxIter;
        const auto lvlWeights = static_cast<const std::array<float, branchFactor> *>(mVoc.getLevelWeights(idxLevel));
        for (uint32_t idxDesc = 0; idxDesc < nbDesc;) {
            const uint32_t idxInLeafLevel = indicesInLeafLevel.at(idxDesc);
            const uint32_t idxInNextLevel = (idxInLeafLevel >> (idxIter * lg2BranchFactor));
            assert(idxInNextLevel == indicesInNextLevel[idxDesc]);
            const float w = getWeight(lvlWeights, idxInNextLevel);
            assert(w >= 0);
            // normalized weights
            uint32_t nbIdenticalQueryNodeEntries = 1u;
            while (idxDesc + nbIdenticalQueryNodeEntries < nbDesc && indicesInNextLevel.at(idxDesc + nbIdenticalQueryNodeEntries) == idxInNextLevel){
                nbIdenticalQueryNodeEntries++;
            }
            const float queryEntry = nbIdenticalQueryNodeEntries * w * queryWeightRcpL1Norm;
            const uint32_t nbInvertedFiles = (1u << (lg2BranchFactor * (nbLevels - 1 - idxLevel)));
            const uint32_t idxLeafIvfBeg = (idxInLeafLevel & ~(nbInvertedFiles - 1));
            assert(idxLeafIvfBeg == idxInLeafLevel / nbInvertedFiles * nbInvertedFiles);
            const uint32_t idxLeafIvfEnd = idxLeafIvfBeg + nbInvertedFiles;
            const uint32_t nbIvfEntries = std::accumulate(
                    leafInvertedFiles+idxLeafIvfBeg, leafInvertedFiles+idxLeafIvfEnd, 0u,
                    [](uint32_t acc, const InvertedFile& ivf){return acc + (uint32_t)ivf.size();});
            if (idxLevel >= nbLevels - 1u || nbIvfEntries < mStopThresholdOffset + maxNbResults * mStopThresholdScale)
            {
                auto accumulateDistanceL1 = [&](IdxDoc idxDoc, uint32_t nbIdenticalIvfEntries){
                    const float trainWeightRcpL1Norm = mWeightVecRcpL1Norm.at(idxDoc);
                    const float trainEntry = nbIdenticalIvfEntries * w * trainWeightRcpL1Norm;
                    distanceL1[idxDoc] += std::abs(queryEntry - trainEntry) - (queryEntry + trainEntry);
//                    if (idxDoc == 200) {
//                        assert(nbIdenticalIvfEntries == nbIdenticalQueryNodeEntries);
//                        tmp[0] += queryEntry;
//                        tmp[1] += trainEntry;
//                        tmpNoNorm += nbIdenticalIvfEntries * w;
//                    }
                };
                if (idxLeafIvfEnd - idxLeafIvfBeg == 1u){
                    const InvertedFile& ivf = leafInvertedFiles[idxLeafIvfBeg];
                    for (uint32_t idxIvfEntry = 0; idxIvfEntry < ivf.size();){
                        const IdxDoc idxDoc = ivf[idxIvfEntry];
                        uint32_t nbIdenticalIvfEntries = 1u;
                        while (idxIvfEntry + nbIdenticalIvfEntries < ivf.size() && ivf.at(idxIvfEntry + nbIdenticalIvfEntries) == idxDoc){
                            nbIdenticalIvfEntries++;
                        }
                        accumulateDistanceL1(idxDoc, nbIdenticalIvfEntries);
                        idxIvfEntry += nbIdenticalIvfEntries;
                    }
                }
                else {
                    thread_local static VectorMap<IdxDoc, uint32_t> occurenceInIvf;
                    assert(occurenceInIvf.empty());
                    for (uint32_t idxIvf = idxLeafIvfBeg; idxIvf < idxLeafIvfEnd; idxIvf++){
                        const InvertedFile& ivf = leafInvertedFiles[idxIvf];
                        for (const IdxDoc idxDoc : ivf) {
                            occurenceInIvf[idxDoc]++;
                        }
                    }
                    for (const auto& nbIdenticalIvfEntries : occurenceInIvf){
                        const auto idxDoc = occurenceInIvf.getKey(nbIdenticalIvfEntries);
                        accumulateDistanceL1(idxDoc, nbIdenticalIvfEntries);
                    }
                    occurenceInIvf.clear();
                }
            }
            idxDesc += nbIdenticalQueryNodeEntries;
        }
        std::for_each(indicesInNextLevel.begin(), indicesInNextLevel.end(), [](uint32_t& v){v /= branchFactor;});
    }
    std::vector<std::pair<IdxDoc, float>> orderedDistanceL1(distanceL1.size());
    std::transform(distanceL1.begin(), distanceL1.end(), orderedDistanceL1.begin(),
            [&](double& dist)->std::pair<IdxDoc, float>{
        return {distanceL1.getKey(dist), dist + 2.f};// +2.f is not necessary but we do it to make it match the paper.
    });
    std::sort(orderedDistanceL1.begin(), orderedDistanceL1.end(),
            [](const auto& a, const auto& b){return a.second < b.second;});
//    printf("%lf - %lf | %lf - %lf\n", tmp[0], tmp[1], tmpNoNorm, 1/queryWeightRcpL1Norm);
//    printf("first match: %u - %f\n", orderedDistanceL1[0].first, orderedDistanceL1[0].second);
//    for (const auto& item : orderedDistanceL1){
//        printf("%f\t", item.second);
//    }
//    printf("\n");
    if (orderedDistanceL1.size() > maxNbResults){
        orderedDistanceL1.resize(maxNbResults);
    }
    std::vector<IdxDoc> result(orderedDistanceL1.size());
    std::transform(orderedDistanceL1.begin(), orderedDistanceL1.end(), result.begin(),
            [](const auto a){return a.first;});
    return result;
}

std::vector<uint32_t>
DataBase::queryAndAddDoc(uint32_t idxDoc, const uint32_t *hostIndicesInLeafLevel, uint32_t nbDesc, uint32_t maxNbResults) {
    const auto result = query(hostIndicesInLeafLevel, nbDesc, maxNbResults);
    addDoc(idxDoc, hostIndicesInLeafLevel, nbDesc);
    return result;
}
}
