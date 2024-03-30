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
// Created by yao on 10/19/19.
//

#include "cuda_hint.cuh"
#include "kmeans.cuh"
#include <vector>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda_utils.h>

namespace cg = cooperative_groups;

using Traits = KMeansTraits;

namespace {

constexpr uint32_t ctaSize = 128;
constexpr uint32_t warpsPerCta = ctaSize / Traits::warp_size;

template <uint32_t tileSize>
__global__ void __launch_bounds__(ctaSize, 4) kernel_reformatDesc(
        KArray<Traits::Vec, Traits::vecsPerDesc, tileSize> *__restrict__ tiles,
        const typename Traits::Descriptor *__restrict__ descriptors,
        const KArray<uint32_t, tileSize> *__restrict__ indices, // desc indices for each tile
        uint32_t nbTiles) {
    assert(ctaSize == blockDim.x);
    static_assert(tileSize <= Traits::warp_size && Traits::vecsPerDesc == Traits::warp_size);
    const uint32_t idxWarp = warpsPerCta * blockIdx.x + threadIdx.x / Traits::warp_size;
    if (idxWarp >= nbTiles) {
        return;
    }
    const auto warp = cg::tiled_partition<Traits::warp_size>(cg::this_thread_block());
    const uint32_t idxLane = warp.thread_rank();

    const uint32_t storage = idxLane < tileSize ? indices[idxWarp][idxLane] : idxInvalid;
    __syncwarp(~0u);

    __shared__ KArray<Traits::Vec, warpsPerCta, tileSize, Traits::vecsPerDesc> ctaBuf;
    auto& warpBuf = ctaBuf[idxWarp % warpsPerCta];
#pragma unroll
    for (uint32_t i = 0; i < tileSize; i++){
        const uint32_t index = warp.shfl(storage, i);
        Traits::Vec tmp{{~0u}};
        if (index != idxInvalid){
            tmp = descriptors[index][idxLane];
        }
        warpBuf[i][(idxLane + i) % Traits::vecsPerDesc] = tmp;
    }
    __syncwarp(~0u);
    auto& dstTile = tiles[idxWarp];
    for (uint32_t i = 0; i < Traits::vecsPerDesc; i++) {
        if (idxLane < tileSize) {
            dstTile[i][idxLane] = warpBuf[idxLane][(idxLane + i) % Traits::vecsPerDesc];
        }
    }
}

}

void launchReformatDesc(typename Traits::DescTile *__restrict__ tiles,
                        const typename Traits::Descriptor *__restrict__ descriptors,
                        const KArray<uint32_t, Traits::tileSize> *__restrict__ indices, // desc indices for each tile
                        uint32_t nbTiles, cudaStream_t stream){
    if (nbTiles != 0){
        kernel_reformatDesc<Traits::tileSize><<<divUp(nbTiles, warpsPerCta), ctaSize, 0, stream>>>(tiles, descriptors, indices, nbTiles);
        cudaCheck(cudaGetLastError());
    }
}

void launchKMeansInitCenters(typename Traits::Centers*__restrict__ tiles,
                             const typename Traits::Descriptor *__restrict__ descriptors,
                             const KArray<uint32_t, Traits::nbCenters> *__restrict__ indices, // desc indices for each tile
                             uint32_t nbTiles, cudaStream_t stream){
    if (nbTiles != 0){
        kernel_reformatDesc<Traits::nbCenters><<<divUp(nbTiles, warpsPerCta), ctaSize, 0, stream>>>(tiles, descriptors, indices, nbTiles);
        cudaCheck(cudaGetLastError());
    }
}
