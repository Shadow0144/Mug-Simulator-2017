cmake_minimum_required(VERSION 3.1)

project(cpd_vtk)

set (CMAKE_CXX_FLAGS "--std=gnu++11 ${CMAKE_CXX_FLAGS}")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")

#find_package(Cpd REQUIRED)
find_package(Figtree REQUIRED)
find_package(VTK 7 REQUIRED)

find_package(Qt4 REQUIRED)
find_package(Armadillo REQUIRED)
find_package(Ceres REQUIRED)
find_package(BLAS)
find_package(LAPACK)

INCLUDE(${VTK_USE_FILE})

include_directories(
  include
  ${ARMADILLO_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${FIGTREE_INCLUDE_DIRS}
)

file(GLOB srcs "src/*.cpp" "src/cpd/*.cpp")
file(GLOB hdrs "include/*.hpp" "include/cpd/*.hpp")

add_executable(cpd_vtk ${srcs} ${hdrs})

target_link_libraries(cpd_vtk
   ${VTK_LIBRARIES}
   ${ARMADILLO_LIBRARIES}
   ${CERES_LIBRARIES}
   ${LAPACK_LIBRARIES}
   ${BLAS_LIBRARIES}
   ${FIGTREE_LIBRARY}
)
