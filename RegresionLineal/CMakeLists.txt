cmake_minimum_required(VERSION 3.5)

project(RegresionLineal LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(project_sources
    main.cpp
    linearregresion.cpp
    extraerdata/extraerdata.cpp
    )

set(project_headers
    linearregresion.h
    extraerdata/extraerdata.h
    )

add_executable(${PROJECT_NAME}
    ${project_sources}
    ${project_headers})
