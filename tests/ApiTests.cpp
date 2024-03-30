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

#include "../RapidBoW.h"
#include <gtest/gtest.h>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <filesystem>
#include <cuda_utils.h>
#include "../KArray.h"
namespace fs = std::filesystem;

void adaptiveNonMaximalSuppresion(std::vector<cv::KeyPoint>& keypoints,
                                  uint32_t numToKeep);

TEST(ApiTest, golf)
{
#if CV_VERSION_MAJOR < 4 || CV_VERSION_MINOR < 5
	printf("Skipped because insufficient opencv version");
#else
    const auto sift = cv::SIFT::create();
    const char filelist[] = "/home/yao/projects/RapidBoW/tests/data/golf/list.txt";
    const fs::path resultPath = "/home/yao/projects/RapidBoW/tests/data/golf/result/";

//    const char filelist[] = "/home/yao/projects/RapidBoW/tests/data/seq/list.txt";
//    const fs::path resultPath = "/home/yao/projects/RapidBoW/tests/data/seq/result";

//    const char filelist[] = "/home/yao/projects/RapidBoW/tests/data/generated/list.txt";
//    const fs::path resultPath = "/home/yao/projects/RapidBoW/tests/data/generated/result";

    std::ifstream fin;
    fin.open(filelist);
    std::string imgFile, descFile;
    using Descriptor = KArray<uint8_t, 128>;
    struct Item{
        std::string imgFile;
        std::string descFile;
        cv::Mat desc;
        Descriptor* devDesc;
    };
    std::vector<Item> data;
    uint32_t nbDesc = 0u;
    while(fin >> imgFile >> descFile){
        std::vector<cv::KeyPoint> kpoints;
        cv::Mat desc(1, 128, CV_8U);
        if (fs::exists(descFile))
        {
            std::ifstream descFin(descFile, std::ios::binary);
            std::vector<char> descData{std::istreambuf_iterator<char>(descFin),
                                       std::istreambuf_iterator<char>()};
            desc = cv::Mat(descData.size() / 128, 128, CV_8U, descData.data()).clone();
        }
        else {
            cv::Mat img = cv::imread(imgFile, cv::IMREAD_GRAYSCALE);
            assert(!img.empty());
            sift->detect(img, kpoints, cv::noArray());
            adaptiveNonMaximalSuppresion(kpoints, 2000);
            sift->compute(img, kpoints, desc);
//            sift->detectAndCompute(img, cv::noArray(), kpoints, desc);
            desc.convertTo(desc, CV_8U);
            std::ofstream descFout(descFile, std::ios::binary);
            descFout.write(reinterpret_cast<const char*>(desc.data), desc.rows * desc.cols);
        }
//        if (data.size() == 200){
//            cv::Mat img = cv::imread(imgFile, cv::IMREAD_GRAYSCALE);
//            assert(!img.empty());
//            sift->detect(img, kpoints, cv::noArray());
//            cv::Mat outImg0, outImg1;
//            cv::drawKeypoints(img, kpoints, outImg0);
//            adaptiveNonMaximalSuppresion(kpoints, 2000);
//            cv::drawKeypoints(img, kpoints, outImg1);
//            cv::imshow("kpoints", outImg0);
//            cv::imshow("kpoints filtered", outImg1);
//            cv::waitKey();
//        }
        if (desc.empty() || desc.type() != CV_8U || desc.cols != 128)
            throw std::runtime_error("wrong descriptor data type");
        nbDesc += desc.rows;
        data.emplace_back(Item{imgFile, descFile, desc, nullptr});
    }
    const auto devDesc = allocCudaMem<Descriptor, CudaMemType::kDevice>(nbDesc);
    uint32_t offset = 0u;
    cudaStream_t stream = nullptr;
    for (auto& item : data){
        item.devDesc = devDesc.get() + offset;
        cudaCheck(cudaMemcpyAsync(item.devDesc, item.desc.data, 128 * item.desc.rows, cudaMemcpyHostToDevice, stream));
        offset += item.desc.rows;
    }
    assert(offset == nbDesc);
    cudaCheck(cudaStreamSynchronize(stream));

    const uint32_t nbLevels = 4;
    const uint32_t nbDoc = data.size();
    const auto voc = rbow::buildVocabulary(rbow::SiftAttr, 16, devDesc.get(), nbDesc, nbDoc, nbLevels, stream);
    const auto db = rbow::createDataBase(voc.get());
    std::vector<uint32_t, CudaManagedAllocator<uint32_t>> indicesInLeafNodes;
    for (uint32_t i = 0; i < data.size(); i++) {
        const auto& item = data.at(i);
        const uint32_t imgNbDesc = item.desc.rows;
        indicesInLeafNodes.resize(imgNbDesc);
        voc->lookUp(item.devDesc, imgNbDesc, indicesInLeafNodes.data(), stream);
        cudaCheck(cudaMemPrefetchAsync(indicesInLeafNodes.data(), sizeof(indicesInLeafNodes[0]) * indicesInLeafNodes.size(), cudaCpuDeviceId, stream));
        cudaCheck(cudaStreamSynchronize(stream));
        db->addDoc(i, indicesInLeafNodes.data(), imgNbDesc);
    }
    {
        const uint32_t idxQuery = 67;
        const auto& item = data.at(idxQuery);
        const uint32_t imgNbDesc = item.desc.rows;
        indicesInLeafNodes.resize(imgNbDesc);
        voc->lookUp(item.devDesc, imgNbDesc, indicesInLeafNodes.data(), stream);
        cudaCheck(cudaMemPrefetchAsync(indicesInLeafNodes.data(), sizeof(indicesInLeafNodes[0]) * indicesInLeafNodes.size(), cudaCpuDeviceId, stream));
        cudaCheck(cudaStreamSynchronize(stream));
        const uint32_t maxNbMatches = 40;
        const auto results = db->query(indicesInLeafNodes.data(), imgNbDesc, maxNbMatches);
        EXPECT_EQ(idxQuery, results.at(0));
        if (!fs::exists(resultPath)){
            fs::create_directories(resultPath);
        }
        for (auto& item : fs::directory_iterator{resultPath})
            fs::remove_all(item);

        std::cout << "query: " << item.imgFile << std::endl;
        for (uint32_t i = 0; i < results.size(); i++){
            const uint32_t idxImg = results.at(i);
            const auto& matchItem = data.at(idxImg);
            std::cout << "match: " << matchItem.imgFile << std::endl;
            fs::copy(matchItem.imgFile, resultPath/("match_" + std::to_string(i) + ".jpg"));
        }
    }

}


//"Multi-Image Matching using Multi-Scale Oriented Patches" by Brown, Szeliski, and Winder.
void adaptiveNonMaximalSuppresion( std::vector<cv::KeyPoint>& keypoints,
                                   const uint32_t numToKeep )
{
    if(keypoints.size() < numToKeep) { return; }

    //
    // Sort by response
    //
    std::sort( keypoints.begin(), keypoints.end(),
               [&]( const cv::KeyPoint& lhs, const cv::KeyPoint& rhs )
               {
                   return lhs.response > rhs.response;
               } );

    std::vector<cv::KeyPoint> anmsPts;

    std::vector<double> radii;
    radii.resize( keypoints.size() );
    std::vector<double> radiiSorted;
    radiiSorted.resize( keypoints.size() );

    const float robustCoeff = 1.11; // see paper

    //@fixme: Use a grid to accelerate.
    for(uint32_t i = 0; i < keypoints.size(); ++i )
    {
        const float response = keypoints[i].response * robustCoeff;
        double radius = std::numeric_limits<double>::max();
        for( uint32_t j = 0; j < i && keypoints[j].response > response; ++j )
        {
            radius = std::min( radius, cv::norm( keypoints[i].pt - keypoints[j].pt ) );
        }
        radii[i]       = radius;
        radiiSorted[i] = radius;
    }

    std::sort( radiiSorted.begin(), radiiSorted.end(),
               [&]( const double& lhs, const double& rhs )
               {
                   return lhs > rhs;
               } );

    const double decisionRadius = radiiSorted[numToKeep];
    for (uint32_t i = 0; i < radii.size(); ++i)
    {
        if (radii[i] >= decisionRadius)
        {
            anmsPts.push_back(keypoints[i]);
        }
    }

    anmsPts.swap(keypoints);
#endif
}

// another paper is https://www.researchgate.net/publication/323388062_Efficient_adaptive_non-maximal_suppression_algorithms_for_homogeneous_spatial_keypoint_distribution

