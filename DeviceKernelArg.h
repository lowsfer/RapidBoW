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

#pragma once
#include <cuda_utils.h>
#include <algorithm>
#include <functional>
#include <CudaEventPool.h>

// For kernel arguments that need to be stored in global memory.
// Not thread-safe, only stream-safe.
template <typename T>
class DeviceKernelArg
{
public:
    void resize(size_t size, bool sync = true){
        if (sync){cudaCheck(cudaEventSynchronize(mEvent.get()));}
        if (mCapacity < size){
            mHostCopy = allocCudaMem<T, CudaMemType::kPinned>(size);
            mDevCopy = allocCudaMem<T, CudaMemType::kDevice>(size);
            mCapacity = size;
        }
        mSize = size;
    }
    size_t getSize() const {return mSize;}

    template <typename Func>
    void dispatch(Func&& kernelLauncher, const T* src, size_t size, cudaStream_t stream) {
        asyncAllocAndFill(src, size, stream);
        const auto delayedFree = makeScopeGuard([&](){asyncFree(stream);});
        kernelLauncher(getDevicePtr(), stream);
    }

    bool checkAvailable() const {
        const cudaError_t err = cudaEventQuery(mEvent.get());
        switch(err){
            case cudaSuccess: return true;
            case cudaErrorNotReady: return false;
            default: cudaCheck(err);
        }
        throw std::runtime_error("You should never reach here");
    }

    void asyncAllocAndFill(const T* src, size_t size, cudaStream_t stream) {
        cudaCheck(cudaEventSynchronize(mEvent.get()));
        resize(size, false);
        std::copy_n(src, getSize(), mHostCopy.get());
        cudaCheck(cudaMemcpyAsync(mDevCopy.get(), mHostCopy.get(), sizeof(T) * getSize(), cudaMemcpyDeviceToDevice, stream));
    }

    const T* getDevicePtr() const{
        return mDevCopy.get();
    }

    //! @param stream: must be the same stream where you filled and used the arguments. If you used it in multiple streams, this stream shall be sync with all those streams.
    void asyncFree(cudaStream_t stream) {
        cudaCheck(cudaEventRecord(mEvent.get(), stream));
    }

private:
    size_t mCapacity{0};
    CudaMem<T, CudaMemType::kPinned> mHostCopy;
    CudaMem<T, CudaMemType::kDevice> mDevCopy;
    size_t mSize{0};
    cudapp::PooledCudaEvent mEvent = cudapp::createPooledCudaEvent();
};
