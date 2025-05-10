cmake_minimum_required(VERSION 3.28.3)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.9/bin/nvcc)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++)
project(CudaEnergyLearning LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



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

# Add custom target for PTX generation
add_custom_target(generate_time_series_ptx
        COMMAND ${CMAKE_CUDA_COMPILER} -ccbin g++ -ptx -src-in-ptx -Xptxas=-v
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cu
        -o ${CMAKE_CURRENT_BINARY_DIR}/main.ptx
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating PTX code for energy time-series"
        VERBATIM
)

# Generate comprehensive optimization report with SASS output
add_custom_target(comprehensive_report
        COMMAND ${CMAKE_CUDA_COMPILER} -ccbin g++ -arch=sm_86 --ptxas-options=-v
        --keep-device-functions -lineinfo
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cu
        -o ${CMAKE_CURRENT_BINARY_DIR}/main_optimized.ptx
        > ${CMAKE_CURRENT_BINARY_DIR}/ptxas_report.txt 2>&1
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating comprehensive optimization report with SASS code"
        VERBATIM
)