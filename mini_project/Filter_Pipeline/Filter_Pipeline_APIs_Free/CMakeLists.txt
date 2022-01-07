cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(cloud_viewer)
project(cloud_viewer1)

find_package(PCL 1.2 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


add_executable (cloud_viewer Filter_Pipeline_APIs_Free.cpp)
target_link_libraries (cloud_viewer ${PCL_LIBRARIES})
add_executable (cloud_viewer1 Filter_Pipeline_APIs_Free_optimized_beta_version.cpp)
target_link_libraries (cloud_viewer1 ${PCL_LIBRARIES})
