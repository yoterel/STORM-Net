#pragma once

#include <string>

#include <Magnum/Platform/Sdl2Application.h>
#include <Magnum/ImGuiIntegration/Widgets.h>

#include <filesystem>

#include "template.h"

#include <mutex>
#include <atomic>

#include <sstream>

#include <thread>
#include <vector>

#include <chrono>



using namespace std;

namespace fs = std::filesystem;

using namespace std::chrono_literals;


class RenderData
{
private:

    string render_data_model;
    string render_data_model_short;

    string render_log_file;
    string render_log_file_short;

    string render_synth_output;
    string render_synth_output_short;

    string render_executable;
    string render_executable_short;

    int render_iterations = 0;

    string cmd_path;
    FILE* render_data_pipe = NULL;

    thread log_thread;
    vector<string> log_lines;
    mutex log_mutex;
    atomic_bool stop_log_logthread = true;

public:

    RenderData();

    ~RenderData();

    void draw_gui();

    void join_threads();

    string getExecutablePathShort();

};

