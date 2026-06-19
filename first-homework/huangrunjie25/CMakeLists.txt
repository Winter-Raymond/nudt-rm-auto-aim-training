cmake_minimum_required(VERSION 3.10)
project(green_ball_tracker)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)

add_executable(ball_tracker main.cpp)
target_include_directories(ball_tracker PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(ball_tracker PRIVATE ${OpenCV_LIBS})
