cmake_minimum_required(VERSION 3.10)
project(BallTrackCPP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 OpenCV
find_package(OpenCV REQUIRED)
# 查找文件系统（部分编译器需要）
find_package(Threads REQUIRED)

# 包含 OpenCV 头文件
include_directories(${OpenCV_INCLUDE_DIRS})
# 包含 nlohmann/json
include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(ball_track src/main.cpp)
# 链接 OpenCV 和 pthread
target_link_libraries(ball_track ${OpenCV_LIBS} pthread)
