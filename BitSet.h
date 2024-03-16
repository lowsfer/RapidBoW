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