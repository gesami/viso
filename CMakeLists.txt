cmake_minimum_required(VERSION 2.8)
project(viso)
set(CMAKE_CXX_FLAGS "-std=c++11 -O2")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# opencv
find_package(OpenCV 3 REQUIRED PATHS /work/ros/kinetic/)
include_directories(${OpenCV_INCLUDE_DIRS})

# pangolin
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

# eigen
include_directories("/usr/include/eigen3")  # change this if your eigen is not installed by apt-get

# sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# g2o
cmake_policy(SET CMP0015 NEW) # relative link directories
link_directories(lib/)
SET(G2O_LIBS g2o_core g2o_stuff g2o_types_sba)

include_directories(include)

add_executable(viso src/keyframe.cpp include/viso.h src/viso.cpp src/main.cpp include/timer.h include/map_point.h include/ring_buffer.h include/bundle_adjuster.h src/initializer.cpp)
target_link_libraries(viso ${OpenCV_LIBS} ${Pangolin_LIBRARIES} ${G2O_LIBS})

