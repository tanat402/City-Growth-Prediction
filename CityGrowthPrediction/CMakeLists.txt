cmake_minimum_required(VERSION 3.10)
project(CityGrowthPrediction VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source files
set(SOURCES
    src/city_data.cpp
    src/neural_network.cpp
    src/city_growth_model.cpp
    main.cpp
)

# Create executable
add_executable(city_growth_prediction ${SOURCES})

# Include directories
target_include_directories(city_growth_prediction PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Output directory
set_target_properties(city_growth_prediction PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Print configuration
message(STATUS "City Growth Prediction Model Configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Source files: ${SOURCES}")
