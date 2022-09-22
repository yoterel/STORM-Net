#pragma once

#include "settings.h"

#include "file_dialog.h"

#include <string>
#include <spdlog/spdlog.h>
#include <Magnum/ImageView.h>
#include <Magnum/PixelFormat.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/ImageView.h>
#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/Platform/Sdl2Application.h>
#include <Magnum/Trade/AbstractImporter.h>
#include <Magnum/Trade/ImageData.h>
#include <Magnum/ImGuiIntegration/Widgets.h>
#include <torch/torch.h>

#include <array>
#include <vector>

#include <thread>

#include <chrono>

#include "utils.h"

#include "image_utils.h"
#include "config.h"

#include <atomic>

#include "convd2d.h"

#include "torch_data.h"

#include <mutex>

#include <sstream>
#include <filesystem>

#include <nlohmann/json.hpp>

#include <fstream>

#include <random>


using namespace std;

using namespace Magnum;
using namespace Math::Literals;

namespace fs = std::filesystem;

using json = nlohmann::json;



class ModelTrainer
{
private:

    Settings& settings;

    string train_synth_output;
    string train_synth_output_short;

    string train_log_file;
    string train_log_file_short;

    atomic_bool stop_stormnet_training = true;
    thread train_stormnet;

    vector<string> train_lines;

    mutex train_log_mutex;

public:

    ModelTrainer(Settings& settings);

    ~ModelTrainer();

    void draw_gui();

    void join_threads();

};

