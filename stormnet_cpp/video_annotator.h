#pragma once

#include "settings.h"

#include <Magnum/ImGuiIntegration/Context.hpp>
#include <Magnum/Platform/Sdl2Application.h>

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
#include <Magnum/Platform/Sdl2Application.h>
#include <Magnum/Trade/AbstractImporter.h>
#include <Magnum/Trade/ImageData.h>
#include <Magnum/ImGuiIntegration/Widgets.h>
#include <torch/torch.h>

#include <array>
#include <vector>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <thread>

#include <dlib/opencv.h>

#include <chrono>

#include "utils.h"

#include "image_utils.h"
#include "config.h"

#include <string>

#include <Magnum/Platform/Sdl2Application.h>
#include <Magnum/ImGuiIntegration/Widgets.h>
#include <Magnum/Math/Matrix4.h>

#include <torch/torch.h>

#include "convd2d.h"

#include "template.h"
#include "config.h"


using namespace cv;
using namespace std;

using namespace Magnum;
using namespace Math::Literals;

using namespace torch::indexing;


class VideoAnnotator
{
private:

    thread fetch_frames;
    bool update_video = false;
    bool is_fetching_frames = false;

    string video_path;

    GL::Texture2D frameView;
    bool hasFrameView = false;

    Mat frame;
    Mat annotated;

    bool frameViewCreated = false;

    int video_width, video_height;

    int marker_id = 0;
    vector<array<Vector2, 7>> markers;
    array<string, 7> marker_names;

    bool has_manually_annotated = false;
    int frame_idx = 0;

    dlib::frontal_face_detector detector;
    dlib::shape_predictor sp;
    bool dlib_loaded = false;
    std::thread dlib_load_thread;

    int face_detector_method = 0;

    CascadeClassifier face_cascade;
    thread haarcascade_load_thread;
    bool haarcascade_loaded = false;

    bool do_autoannotate = false;

    thread autoannotate;
    bool is_autoannotating = false;
    int autoannotating_progress = 0;

    vector<Mat> frames;

    string template_name;
    string template_path;

    string stormnet_model;
    string stormnet_model_path;

    Template online_template;

    Settings& settings;

    torch::Tensor marker_to_tensor(int frame_idx);

public:

    VideoAnnotator(Settings& settings);

    ~VideoAnnotator();

    void draw_gui(bool* show_online_step);

    void update();

    void join_threads();

    void updateVideoTexture();

    void updateAnnotations();

    void clearMarkers();

    void setup_dlib();

    bool is_dlib_loaded();

    void load_classifier();

    bool is_haarcascade_loaded();

    vector<array<Vector2, 7>>& getMarkers();


    bool fetchFrames(const char* filename, vector<int> frame_indices = {});

};

