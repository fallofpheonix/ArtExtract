cmake_minimum_required(VERSION 3.16)

project(ArtExtract VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(artextract STATIC
    src/extractor.cpp
)
target_include_directories(artextract
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(artextract_cli
    src/main.cpp
)
target_link_libraries(artextract_cli PRIVATE artextract)
set_target_properties(artextract_cli PROPERTIES OUTPUT_NAME artextract)

enable_testing()
add_executable(artextract_tests
    tests/test_extractor.cpp
)
target_link_libraries(artextract_tests PRIVATE artextract)
add_test(NAME artextract_tests COMMAND artextract_tests)
