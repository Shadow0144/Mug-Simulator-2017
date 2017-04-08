cmake_minimum_required(VERSION 3.1)

project(gplvm)

set (CMAKE_CXX_FLAGS "--std=gnu++11 ${CMAKE_CXX_FLAGS}")

find_package(Armadillo REQUIRED)
find_package(Ceres REQUIRED)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")

include_directories(
  include
  ${ARMADILLO_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
)

file(GLOB srcs "src/*.cpp")
file(GLOB hdrs "include/*.hpp")

add_executable(gplvm ${srcs} ${hdrs})

target_link_libraries(gplvm
   ${ARMADILLO_LIBRARIES}
   ${CERES_LIBRARIES}
)

