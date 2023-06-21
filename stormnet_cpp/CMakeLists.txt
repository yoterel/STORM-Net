cmake_minimum_required( VERSION 3.14 )
set(CMAKE_CXX_STANDARD 17)

project( stormnetcpp )

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/modules/" ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/SDL2-2.24.0" ${CMAKE_PREFIX_PATH})
set(CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/libtorch" ${CMAKE_PREFIX_PATH})
set(CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/opencv" ${CMAKE_PREFIX_PATH})

add_compile_definitions(USE_AVX_INSTRUCTIONS=ON)

include(FetchContent)
FetchContent_Declare(dlib
   GIT_REPOSITORY https://github.com/davisking/dlib.git
   GIT_TAG        v19.24
)
FetchContent_MakeAvailable(dlib)

FetchContent_Declare(json
   GIT_REPOSITORY https://github.com/nlohmann/json.git
   GIT_TAG        v3.11.2
)
FetchContent_MakeAvailable(json)

add_subdirectory(corrade EXCLUDE_FROM_ALL)

set(MAGNUM_WITH_SDL2APPLICATION ON CACHE BOOL "" FORCE)
set(MAGNUM_WITH_TGAIMPORTER ON CACHE BOOL "" FORCE)
set(IMGUI_DIR "${PROJECT_SOURCE_DIR}/imgui")
set(MAGNUM_WITH_IMGUI ON CACHE BOOL "" FORCE)

set_directory_properties(PROPERTIES CORRADE_USE_PEDANTIC_FLAGS ON)

add_subdirectory(magnum EXCLUDE_FROM_ALL)
add_subdirectory(magnum-integration EXCLUDE_FROM_ALL)
add_subdirectory(nativefiledialog-extended)
add_subdirectory(spdlog)

find_package(SDL2 CONFIG REQUIRED)
find_package(Magnum REQUIRED GL Sdl2Application)
find_package(MagnumIntegration REQUIRED ImGui)
find_package(Torch REQUIRED)
find_package(OpenCV 4.6.0 EXACT REQUIRED )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable( stormnetcpp 
  main.cpp 
  template.cpp
  template.h
  string_utils.cpp
  string_utils.h
  arc_ball.cpp
  arc_ball.h
  torch_data.cpp
  torch_data.h
  image_utils.cpp
  image_utils.h
  convd2d.h
  utils.cpp
  utils.h
  file_dialog.h
  file_dialog.cpp
  render_data.h
  render_data.cpp
  template_viewer.cpp
  template_viewer.h
  video_annotator.cpp
  video_annotator.h
  model_trainer.cpp
  model_trainer.h
  settings.h
  settings.cpp
  blob_detector.cpp
  blob_detector.h
)


# target_compile_definitions(stormnetcpp PRIVATE
    # MAGNUM_IMGUIINTEGRATION_USER_CONFIG="imgui/imconfig.h")

target_link_libraries(stormnetcpp PRIVATE SDL2::SDL2 SDL2::SDL2main)
target_link_libraries(stormnetcpp PRIVATE
    Magnum::Application
    Magnum::GL
    Magnum::Magnum
    Magnum::MeshTools
    Magnum::Primitives
    Magnum::Shaders
    )
target_link_libraries(stormnetcpp PRIVATE MagnumIntegration::ImGui)
target_link_libraries(stormnetcpp PRIVATE nfd)
target_link_libraries(stormnetcpp PRIVATE ole32.lib uuid.lib) # For nfd
target_link_libraries(stormnetcpp PRIVATE spdlog::spdlog)
target_link_libraries(stormnetcpp PRIVATE ${TORCH_LIBRARIES})
target_link_libraries(stormnetcpp PRIVATE ${OpenCV_LIBS})

target_link_libraries(stormnetcpp PRIVATE dlib::dlib)
target_link_libraries(stormnetcpp PRIVATE nlohmann_json::nlohmann_json)

target_compile_features(stormnetcpp PRIVATE cxx_std_14)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET stormnetcpp 
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:stormnetcpp>)
endif (MSVC)

if (MSVC)
  file(GLOB OPENCV_DLLS "${PROJECT_SOURCE_DIR}/opencv/x64/vc15/bin/*.dll")
  add_custom_command(TARGET stormnetcpp 
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${OPENCV_DLLS}
                     $<TARGET_FILE_DIR:stormnetcpp>)

  file(GLOB OPENCV_DLLS "${PROJECT_SOURCE_DIR}/opencv/bin/*.dll")
  add_custom_command(TARGET stormnetcpp 
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${OPENCV_DLLS}
                     $<TARGET_FILE_DIR:stormnetcpp>)
endif (MSVC)

