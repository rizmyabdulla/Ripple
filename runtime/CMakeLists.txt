cmake_minimum_required(VERSION 3.20)

project(ripple_runtime VERSION 0.1.0 LANGUAGES CXX)

option(RIPPLE_BUILD_SHARED "Build Ripple as a shared library" OFF)
option(RIPPLE_BUILD_TESTS "Build native runtime tests" ON)

if(RIPPLE_BUILD_SHARED)
  set(RIPPLE_LIBRARY_TYPE SHARED)
else()
  set(RIPPLE_LIBRARY_TYPE STATIC)
endif()

add_library(ripple_runtime ${RIPPLE_LIBRARY_TYPE} src/ripple.cpp)
add_library(Ripple::runtime ALIAS ripple_runtime)
target_compile_features(ripple_runtime PUBLIC cxx_std_20)
target_include_directories(
  ripple_runtime
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
set_target_properties(
  ripple_runtime
  PROPERTIES
    CXX_EXTENSIONS OFF
    OUTPUT_NAME ripple
    POSITION_INDEPENDENT_CODE ON
)

if(RIPPLE_BUILD_SHARED)
  target_compile_definitions(
    ripple_runtime
    PUBLIC RIPPLE_SHARED
    PRIVATE RIPPLE_BUILDING_LIBRARY
  )
endif()

if(MSVC)
  target_compile_options(ripple_runtime PRIVATE /W4 /permissive- /EHsc)
else()
  target_compile_options(
    ripple_runtime
    PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow
  )
endif()

if(RIPPLE_BUILD_TESTS)
  include(CTest)
  add_executable(ripple_runtime_tests tests/runtime_test.cpp)
  target_link_libraries(ripple_runtime_tests PRIVATE Ripple::runtime)
  target_compile_features(ripple_runtime_tests PRIVATE cxx_std_20)
  add_test(NAME ripple_runtime_tests COMMAND ripple_runtime_tests)
endif()

include(GNUInstallDirs)
install(
  TARGETS ripple_runtime
  EXPORT RippleRuntimeTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(FILES include/ripple.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
