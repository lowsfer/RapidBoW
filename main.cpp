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

#include "RapidBoW.h"
#include <fstream>
#include <cassert>
#include <filesystem>
#include <cuda_utils.h>
namespace fs = std::filesystem;

const int nbDocs = 527;
const int nbLevels = 5;

int main(int argc, const char* argv[]) {
    if (argc != 3 || !fs::exists(argv[1])) {
        printf("makevoc $desc_file, $voc_file");
        return EXIT_FAILURE;
    }

    std::ifstream fin(argv[1], std::ios::binary);
    std::vector<char> allDesc(fs::file_size(argv[1]));
    fin.read(allDesc.data(), allDesc.size());
    fin.close();

    assert(allDesc.size() % 128 == 0);
    const auto devDesc = allocCudaMem<char>(allDesc.size());
    cudaCheck(cudaMemcpy(devDesc.get(), allDesc.data(), allDesc.size(), cudaMemcpyHostToDevice));
    const auto voc = rbow::buildVocabulary(rbow::SiftAttr, 16, devDesc.get(), allDesc.size() / 128, nbDocs, nbLevels, nullptr);
    const auto blob = voc->serialize();
    std::ofstream fout(argv[2], std::ios::binary | std::ios::trunc);
    fout.write(reinterpret_cast<const char*>(blob.data()), blob.size());
    fout.close();

    return EXIT_SUCCESS;
}