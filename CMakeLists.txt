cmake_minimum_required(VERSION 3.16.3 FATAL_ERROR)

project(model)

set(CMAKE_PREFIX_PATH libtorch)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

add_executable(model model.cpp)
target_compile_features(model PUBLIC cxx_range_for)
target_link_libraries(model "${TORCH_LIBRARIES}")

set_property(TARGET model PROPERTY CXX_STANDARD 17)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET model
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:model>)
endif (MSVC)