cmake_minimum_required(VERSION 3.12)
project(RapidBoW LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75-real 86-real)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (NOT DEFINED DEP_BUILD_SUBFOLDER)
        set(DEP_BUILD_SUBFOLDER "qtcreator-release")
endif()

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${CMAKE_SOURCE_DIR}/../cuda-utils /usr/include/opencv4)
link_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../lib64 ${CMAKE_SOURCE_DIR}/../cuda-utils/build/${DEP_BUILD_SUBFOLDER})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -ffast-math -Wreturn-type -Wall -Wno-unknown-pragmas -Wextra  -Werror")

find_package(Eigen3 REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -keep --expt-relaxed-constexpr") #-res-usage
set(CUDA_PTXAS_FLAGS_RELEASE "${CUDA_PTXAS_FLAGS_RELEASE} -warn-lmem-usage -warn-spills -Werror") # -v
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} --use_fast_math -DNDEBUG -Xptxas='${CUDA_PTXAS_FLAGS_RELEASE}'")

add_library(RapidBoW RapidBoW.cpp RapidBoW.h
        Vocabulary.h Vocabulary.cpp
        kmeans.h kmeans.cu reformatDesc.cu cuda_hint.cuh ${CMAKE_SOURCE_DIR}/../cuda-utils/cuda_utils.h
        DeviceKernelArg.h
        kmeans.cpp BitSet.h vocTreeLookUp.cu FullTree.h vocTreeLookUp.h DataBase.cpp DataBase.h RapidBoWInternal.h)

add_executable(test tests/KmeansAssignCenterTests.cpp tests/ReformatTests.cpp tests/VocTests.cpp tests/ApiTests.cpp)
target_link_libraries(test PRIVATE RapidBoW gtest gtest_main opencv_core opencv_highgui opencv_features2d opencv_imgcodecs stdc++fs cuda cuda-utils jpeg turbojpeg heif exiv2)

add_executable(makevoc main.cpp)
target_link_libraries(makevoc PRIVATE RapidBoW stdc++fs cudart cuda-utils jpeg turbojpeg heif exiv2)
