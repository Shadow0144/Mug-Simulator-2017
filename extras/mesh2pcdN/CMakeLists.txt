cmake_minimum_required(VERSION 2.8.11)
project(mesh2pcdN)

find_package(PCL 1.8 REQUIRED)
find_package(VTK 7 REQUIRED)
find_package(Qt4 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include(${VTK_USE_FILE})

#file(GLOB hdrs "include/*.hpp")
file(GLOB srcs "src/*.cpp")

#add_executable(mesh2pcdN ${hdrs} ${srcs})
add_executable(mesh2pcdN ${srcs})

target_link_libraries(mesh2pcdN
   ${PCL_LIBRARIES}
   ${VTK_LIBRARIES}
)