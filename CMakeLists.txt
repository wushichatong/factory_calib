cmake_minimum_required(VERSION 2.8.3)
project(CornerDetect)

set(CMAKE_BUILD_TYPE "Release") # Release
set(CMAKE_CXX_FLAGS "-std=gnu++14 -g -Wall")

# 寻找Eigen库
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
# 包含头文件目录
include_directories(${CERES_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})


find_package(OpenCV REQUIRED)
link_directories(${OpenCV_LIBRARY_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})

find_package(Boost REQUIRED system)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

file(GLOB_RECURSE UTILS src/utils/*.cpp)
add_library(utils STATIC ${UTILS})

file(GLOB_RECURSE PARSER_PATH src/*.cpp src/*.cc)
add_library(${PROJECT_NAME} STATIC ${PARSER_PATH})
target_link_libraries(${PROJECT_NAME} ${CERES_LIBRARIES} libjsoncpp.a ${OpenCV_LIBS} ${Boost_SYSTEM_LIBRARY} ${PCL_LIBRARIES})



add_executable(run_ex src/run_ex.cpp)
target_link_libraries(run_ex ${PROJECT_NAME})
target_link_libraries(run_ex utils)
target_link_libraries(run_ex ${CERES_LIBRARIES})


add_executable(run_ceres src/run_ceres.cpp)
target_link_libraries(run_ceres ${PROJECT_NAME})
target_link_libraries(run_ceres utils)
target_link_libraries(run_ceres ${CERES_LIBRARIES})


add_executable(run_ex_cv src/run_ex_cv.cpp)
target_link_libraries(run_ex_cv ${PROJECT_NAME})
target_link_libraries(run_ex_cv utils)
target_link_libraries(run_ex_cv ${CERES_LIBRARIES})
