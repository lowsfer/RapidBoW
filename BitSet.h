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
// Created by yao on 10/21/19.
//

#pragma once
#include <cstddef>
#include <type_traits>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <cuda_utils.h>

template <size_t nbBits>
struct BitSet
{
    using StorageElemType = std::conditional_t<nbBits <= 8, uint8_t, std::conditional_t<nbBits <= 16, uint16_t, uint32_t >>;
    static constexpr uint32_t bitsPerElem = 8u * sizeof(StorageElemType);

    __forceinline__ __device__
    void set(uint32_t idx, bool val = true) {
        if (val) {
            StorageElemType& elem = data[idx / bitsPerElem];
            elem |= StorageElemType(1u << idx % bitsPerElem);
        } else{
            reset(idx);
        }
    }
    __forceinline__ __device__
    void reset(uint32_t idx) {
        StorageElemType& elem = data[idx / bitsPerElem];
        elem &= StorageElemType(~(1u << idx % bitsPerElem));
    }
    __forceinline__ __device__
    bool test(uint32_t idx) const { return (data[idx / bitsPerElem] & (1u << (idx % bitsPerElem))) != 0; }

    StorageElemType data[divUp(nbBits, bitsPerElem)];
};