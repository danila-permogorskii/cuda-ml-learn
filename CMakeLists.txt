cmake_minimum_required(VERSION 3.29.6)
project(CudaEnergyLearning LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.6/bin/nvcc)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++)

set(CMAKE_CUDA_ARCHITECTURES 86)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(CudaEnergyLearning main.cu)

set_target_properties(CudaEnergyLearning PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(CudaEnergyLearning
        ${CUDA_LIBRARIES}
        ${CUDA_CUDA_LIBRARY}
        ${CUDA_CUDART_LIBRARY}
)

install(TARGETS CudaEnergyLearning DESTINATION bin)

add_custom_target(generate_ptx
        COMMAND ${CMAKE_CUDA_COMPILER} -ptx -src-in-ptx -Xptxas=-v -ccbin=${CMAKE_CXX_COMPILER} ${CMAKE_CURRENT_SOURCE_DIR}/main.cu -o ${CMAKE_CURRENT_BINARY_DIR}/main.ptx
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating PTX code with detailed information for analysis"
        VERBATIM
)
