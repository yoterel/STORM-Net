
#include <opencv2/opencv.hpp>
#include <thread>

#include <nfd.h>

#include "torch_data.h"
#include <spdlog/spdlog.h>

#include <Magnum/Math/Tags.h>

#include "video_annotator.h"


using namespace cv;
using namespace std;

using namespace Magnum;
using namespace Math::Literals;

using namespace torch::indexing;



VideoAnnotator::VideoAnnotator(Settings& settings) :
    settings(settings)
{

    marker_names[0] = "left eye";
    marker_names[1] = "nose tip";
    marker_names[2] = "right eye";
    marker_names[3] = "cap 1";
    marker_names[4] = "cap 2";
    marker_names[5] = "cap 3";
    marker_names[6] = "cap 4";

}

VideoAnnotator::~VideoAnnotator()
{

}

void VideoAnnotator::draw_gui(bool* show_online_step)
{
    ImGui::Begin("Online Step", show_online_step, ImGuiWindowFlags_MenuBar);

    if (ImGui::BeginMenuBar())
    {
        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::MenuItem("Open")) {

                nfdfilteritem_t filterItem[] = { { "Video", "mp4,avi" } };
                if (getPath(filterItem, video_path)) {
                    if (fetch_frames.joinable()) {
                        fetch_frames.join();
                    }

                    fetch_frames = thread([&]() {
                        is_fetching_frames = true;
                        fetchFrames(video_path.c_str());

                        if (frames.size() > 0) {
                            markers.resize(frames.size());
                            for (auto& markers_frame : markers) {
                                for (int i = 0; i < markers_frame.size(); ++i) {
                                    markers_frame[i][0] = 0.f;
                                    markers_frame[i][1] = 0.f;
                                }
                            }
                        }

                        update_video = true;
                        is_fetching_frames = false;

                        has_manually_annotated = false;

                        });
                }

            }
            ImGui::EndMenu();
        }
        ImGui::EndMenuBar();
    }


    if (is_autoannotating) {
        float progress = (float)autoannotating_progress / frames.size();
        ImGui::ProgressBar(progress, ImVec2(0.0f, 0.0f));
    }


    if (is_fetching_frames) {
        float progress = frames.size() / (float)settings.steps_per_datapoint;
        ImGui::ProgressBar(progress, ImVec2(0.0f, 0.0f));
    }

    if (hasFrameView && !is_autoannotating) {

        if (hasFrameView) {
            video_width = std::min(annotated.cols, (int)ImGui::GetWindowWidth() - 20);
            video_height = annotated.rows * video_width / annotated.cols;

            ImVec2 pos = ImGui::GetCursorScreenPos();
            ImGuiIntegration::image(frameView, Vector2(video_width, video_height), { Vector2{0.0f, 1.f}, Vector2{1.f, 0.f} });

            if (ImGui::IsItemClicked()) {
                ImGuiIO& io = ImGui::GetIO();
                float region_x = io.MousePos.x - pos.x;
                float region_y = io.MousePos.y - pos.y;

                markers[frame_idx][marker_id][0] = (float)region_x * annotated.cols / video_width;
                markers[frame_idx][marker_id][1] = (float)region_y * annotated.rows / video_height;
                marker_id = (marker_id + 1) % (int)(markers[frame_idx].size());
                has_manually_annotated = true;

                update_video = true;
            }
        }

        if (ImGui::Button("< Prev")) {
            frame_idx = std::max(0, frame_idx - 1);
            annotated = frames[frame_idx].clone();
            update_video = true;
            marker_id = 0;
        }

        ImGui::SameLine();

        if (ImGui::Button("Next >")) {
            frame_idx = std::min((int)frames.size() - 1, frame_idx + 1);
            annotated = frames[frame_idx].clone();
            update_video = true;
            marker_id = 0;
        }

        ImGui::SameLine();



        ImGui::Text("image size : %d x %d", frames[frame_idx].cols, frames[frame_idx].rows);

        const char* items[] = { "Cascade Classifier", "DLib" };
        if (ImGui::BeginCombo("face detector", items[face_detector_method], 0)) {
            for (int n = 0; n < IM_ARRAYSIZE(items); n++) {
                const bool is_selected = (face_detector_method == n);
                if (ImGui::Selectable(items[n], is_selected)) {
                    face_detector_method = n;
                }

                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        if (ImGui::Button("Automatically annotate")) {
            if (has_manually_annotated) {
                ImGui::OpenPopup("autoannotate_popup");
            }
            else {
                do_autoannotate = true;
            }
        }


        if (ImGui::BeginPopup("autoannotate_popup")) {
            ImGui::Text("Manual Annotation Detected");
            ImGui::Text("Are you sure you want to automaticaly annotate\n the frames? current manual annotations will be lost.");

            if (ImGui::Button("OK")) {
                do_autoannotate = true;
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }


        if (do_autoannotate) {

            if (dlib_loaded) {
                if (autoannotate.joinable()) { autoannotate.join(); }

                is_autoannotating = true;
                autoannotating_progress = 0;
                autoannotate = std::thread([&]() {
                    for (int k = 0; k < frames.size(); ++k) {
                        Mat frame = frames[k];

                        vector<dlib::rectangle> faces;
                        Mat bgr;
                        cvtColor(frame, bgr, COLOR_RGBA2BGR);

                        dlib::cv_image<dlib::bgr_pixel> img(bgr);
                        if (face_detector_method == 1) {
                            faces = detector(img);
                        }
                        else {

                            vector<cv::Rect> cv_faces;
                            Mat gray;
                            cvtColor(bgr, gray, COLOR_BGR2GRAY);
                            face_cascade.detectMultiScale(gray, cv_faces, 1.3, 5);

                            for (auto& face : cv_faces) {
                                faces.emplace_back(face.x, face.y, face.x + face.width, face.y + face.height);
                            }

                        }

                        int num_faces = faces.size();

                        if (num_faces == 0) {
                            spdlog::info("No face detected on frame {}", k);
                        }

                        int face_id = 0;
                        if (num_faces > 1) {
                            // Select bottommost
                            int top = faces[0].top();
                            for (int i = 0; i < num_faces; ++i) {
                                if (faces[i].top() > top) {
                                    top = faces[i].top();
                                    face_id = i;
                                }
                            }

                            spdlog::warn("Multiple face detected on frame {}", k);
                        }

                        if (num_faces > 0 && faces[face_id].top() < 100) {
                            spdlog::warn("Face detected top() < 100 on frame {}", k);
                            num_faces = 0;
                        }

                        if (num_faces > 0) {
                            dlib::full_object_detection shape =
                                sp(img, faces[0]);

                            torch::Tensor left_eye = torch::empty({ 6,2 });
                            torch::Tensor nose = torch::empty({ 2 });
                            torch::Tensor right_eye = torch::empty({ 6,2 });

                            // left eye
                            for (int i = 0; i < 6; ++i) {
                                left_eye.index_put_({ i, 0 }, shape.part(i + 42).x());
                                left_eye.index_put_({ i, 1 }, shape.part(i + 42).y());
                            }

                            // nose
                            nose.index_put_({ 0 }, shape.part(30).x());
                            nose.index_put_({ 1 }, shape.part(30).y());

                            // right eye
                            for (int i = 0; i < 6; ++i) {
                                right_eye.index_put_({ i, 0 }, shape.part(i + 36).x());
                                right_eye.index_put_({ i, 1 }, shape.part(i + 36).y());
                            }

                            left_eye = torch::mean(left_eye, 0);
                            right_eye = torch::mean(right_eye, 0);

                            markers[k][0][0] = left_eye.index({ 0 }).item<float>();
                            markers[k][0][1] = left_eye.index({ 1 }).item<float>();

                            markers[k][1][0] = nose.index({ 0 }).item<float>();
                            markers[k][1][1] = nose.index({ 1 }).item<float>();

                            markers[k][2][0] = right_eye.index({ 0 }).item<float>();
                            markers[k][2][1] = right_eye.index({ 1 }).item<float>();
                        }

                        autoannotating_progress++;
                    }
                    is_autoannotating = false;
                    update_video = true;
                    });
            }
            else {
                spdlog::warn("dlib is not ready yet.");
            }

            do_autoannotate = false;
            has_manually_annotated = false;
        }

        ImGui::Text("Stormnet model:");
        ImGui::SameLine();
        if (stormnet_model == "") {
            ImGui::Text("None");
        }
        else {
            ImGui::Text("%s", stormnet_model.c_str());
        }

        ImGui::SameLine();

        ImGui::PushID("Load model inference");
        if (ImGui::Button("Load")) {

            nfdchar_t* outPath;
            nfdfilteritem_t filterItem[] = { { "Model File", "*" } };
            nfdresult_t result = NFD_OpenDialog(&outPath, filterItem, 1, NULL);

            if (result == NFD_OKAY)
            {
                stormnet_model_path = outPath;
                stormnet_model = stormnet_model_path.substr(stormnet_model_path.find_last_of("/\\") + 1);
                NFD_FreePath(outPath);
            }

        }
        ImGui::PopID();

        ImGui::Text("Template:");
        ImGui::SameLine();
        if (stormnet_model == "") {
            ImGui::Text("None");
        }
        else {
            ImGui::Text("%s", template_name.c_str());
        }

        ImGui::SameLine();

        ImGui::PushID("Load template inference");
        if (ImGui::Button("Load")) {

            nfdchar_t* outPath;
            nfdfilteritem_t filterItem[] = { { "Template file", "txt" } };
            nfdresult_t result = NFD_OpenDialog(&outPath, filterItem, 1, NULL);

            if (result == NFD_OKAY)
            {
                template_path = outPath;
                template_name = template_path.substr(stormnet_model_path.find_last_of("/\\") + 1);
                online_template = Template::read(template_path);

                NFD_FreePath(outPath);
            }

        }
        ImGui::PopID();

        if (ImGui::Button("Co-register")) {




            torch::Tensor heatmaps = torch::empty({ 10, 256, 256 });

            for (int i = 0; i < 10; ++i) {
                torch::Tensor heatmap;
                auto marker_tensor = marker_to_tensor(i);
                center_data(marker_tensor);
                drawGMM(marker_tensor, heatmap, FRAME_WIDTH, FRAME_HEIGHT);
                heatmaps.index_put_({ i }, heatmap);
            }

            heatmaps = heatmaps.unsqueeze(0);


            Convd2d convd2d(10, 3);
            torch::Tensor pred;
            {
                torch::NoGradGuard no_grad;
                convd2d->eval();
                pred = convd2d->forward(heatmaps);
            }

            vector<torch::Tensor> rs, sc;

            for (int i = 0; i < pred.sizes()[0]; ++i) {
                torch::Tensor predi = pred.index({ i });

                Matrix4 scale{ Math::IdentityInit };
                Matrix4 rot =
                    Matrix4::rotationX(Deg(predi.index({ 0 }).item<float>())) *
                    Matrix4::rotationY(Deg(predi.index({ 1 }).item<float>())) *
                    Matrix4::rotationZ(Deg(predi.index({ 2 }).item<float>()));


                torch::Tensor rot_tensor = torch::empty({ 3,3 });
                torch::Tensor scale_tensor = torch::empty({ 3,3 });

                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        rot_tensor.index({ i, j }) = rot[i][j];
                        scale_tensor.index({ i, j }) = scale[i][j];
                    }
                }

                rs.push_back(rot_tensor);
                sc.push_back(scale_tensor);
            }



            online_template.toStandardCoordinateSystem();
            Template projected_data = online_template.applyRigidTransform(rs, sc);


            nfdchar_t* savePath;
            nfdfilteritem_t filterItem[] = { {"Output", "txt"} };
            nfdresult_t result = NFD_SaveDialog(&savePath, filterItem, 1, NULL, "output.txt");
            if (result == NFD_OKAY) {
                projected_data.writeTemporaryFile(savePath);
                spdlog::info("Saved in {}", savePath);
                NFD_FreePath(savePath);
            }
            else {
                spdlog::info("Save canceled!");
            }

        }


        static ImGuiTableFlags table_flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;
        if (ImGui::BeginTable("markers", 4, table_flags)) {
            ImGui::TableSetupColumn("Marker ID");
            ImGui::TableSetupColumn("Marker Name");
            ImGui::TableSetupColumn("Marker Pos");
            ImGui::TableHeadersRow();


            for (int row = 0; row < markers[frame_idx].size(); row++)
            {
                ImGui::TableNextRow();

                ImGui::PushID(row);
                ImGui::TableSetColumnIndex(0);

                ImGuiSelectableFlags selectable_flags = ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowItemOverlap; // allow overlap for the button, otherwise it can't be clicked

                static char label[32];
                sprintf(label, "%d", row);
                if (ImGui::Selectable(label, marker_id == row, selectable_flags)) {
                    marker_id = row;
                }

                ImGui::TableSetColumnIndex(1);
                ImGui::Text("%s", marker_names[row].c_str());

                ImGui::TableSetColumnIndex(2);
                ImGui::Text("(%d,%d)", (int)markers[frame_idx][row][0], (int)markers[frame_idx][row][1]);

                ImGui::TableSetColumnIndex(3);

                if (ImGui::SmallButton("clear")) {
                    markers[frame_idx][row][0] = 0;
                    markers[frame_idx][row][1] = 0;

                    update_video = true;
                }
                ImGui::PopID();
            }

            ImGui::EndTable();
        }

        if (ImGui::Button("Clear all")) {
            clearMarkers();
            update_video = true;
        }


    }
    ImGui::End();
}

void VideoAnnotator::update()
{

    if (update_video) {
        updateAnnotations();
        updateVideoTexture();
        hasFrameView = true;
        update_video = false;
    }

}

void VideoAnnotator::join_threads()
{

    if (fetch_frames.joinable()) {
        fetch_frames.join();
    }

    if (dlib_load_thread.joinable()) {
        dlib_load_thread.join();
    }

    if (haarcascade_load_thread.joinable()) {
        haarcascade_load_thread.join();
    }

    if (autoannotate.joinable()) {
        autoannotate.join();
    }

}

void VideoAnnotator::updateVideoTexture()
{
    ImageView2D img{ PixelFormat::RGBA8Unorm, {annotated.cols, annotated.rows},
      Corrade::Containers::ArrayView<const void>(annotated.ptr(), annotated.rows * annotated.cols * 4) };

    if (!frameViewCreated) {
        frameView.setWrapping(GL::SamplerWrapping::ClampToEdge)
            .setMagnificationFilter(GL::SamplerFilter::Linear)
            .setMinificationFilter(GL::SamplerFilter::Linear)
            .setStorage(1, GL::textureFormat(img.format()), img.size())
            .setSubImage(0, {}, img);
        frameViewCreated = true;
    }
    else {
        frameView.setSubImage(0, {}, img);
    }
}

void VideoAnnotator::updateAnnotations()
{
    if (frame_idx < 0 || frame_idx >= frames.size()) {
        spdlog::error("invalid frame_idx in updateAnnotations");
        return;
    }

    annotated = frames[frame_idx].clone();
    for (int i = 0; i < (int)markers[frame_idx].size(); ++i) {
        auto& marker = markers[frame_idx][i];
        if (marker[0] != 0.f || marker[1] != 0.f) {
            drawMarker(annotated,
                Point((int)marker[0], (int)marker[1]),
                Scalar(255, 0, 0, 255), MARKER_TILTED_CROSS, 20,
                2, LINE_AA);

            putText(annotated, marker_names[i],
                Point((int)marker[0], (int)marker[1] - 20),
                FONT_HERSHEY_PLAIN, 1.5,
                Scalar(255, 0, 0, 255),
                1, LINE_AA);
        }
    }
}


void VideoAnnotator::clearMarkers()
{
    for (auto& vec : markers[frame_idx]) {
        vec[0] = 0.f;
        vec[1] = 0.f;
    }
}

void VideoAnnotator::setup_dlib()
{

    dlib_load_thread = std::thread([&] {
        // We need a face detector.  We will use this to get bounding boxes for
        // each face in an image.
        detector = dlib::get_frontal_face_detector();

        // And we also need a shape_predictor.  This is the tool that will predict face
        // landmark positions given an image and face bounding box.  Here we are just
        // loading the model from the shape_predictor_68_face_landmarks.dat file you gave
        // as a command line argument.
        dlib::deserialize("models/shape_predictor_68_face_landmarks.dat") >> sp;
        dlib_loaded = true;
        });

}

bool VideoAnnotator::is_dlib_loaded()
{
    return dlib_loaded;
}

void VideoAnnotator::load_classifier()
{

    haarcascade_load_thread = std::thread([&]() {
        face_cascade.load("models/haarcascade_frontalface_default.xml");
        haarcascade_loaded = true;
        });

}

bool VideoAnnotator::is_haarcascade_loaded()
{
    return haarcascade_loaded;
}

vector<array<Vector2, 7>>& VideoAnnotator::getMarkers()
{
    return markers;
}

bool VideoAnnotator::fetchFrames(const char* filename, vector<int> frame_indices)
{

    VideoCapture cap(filename);
    STORM_ASSERT(cap.isOpened());


    int estimated_total_frames = cap.get(CAP_PROP_FRAME_COUNT);
    int frames_to_use = estimated_total_frames;
    STORM_ASSERT(frames_to_use > settings.starting_frame);

    frames_to_use -= settings.starting_frame;
    int stride = frames_to_use / settings.steps_per_datapoint;
    STORM_ASSERT(stride > 0);

    frames.clear();
    if (frame_indices.size() > 0) {
        for (int idx : frame_indices) {
            STORM_ASSERT(idx >= 0);
            STORM_ASSERT(idx < frames_to_use);

            cap.set(CAP_PROP_POS_FRAMES, idx);
            Mat frame;
            cap >> frame;

            STORM_ASSERT(!frame.empty());
            resize(frame, frame, Size(FRAME_WIDTH, FRAME_HEIGHT));
            cvtColor(frame, frame, COLOR_BGR2RGBA);
            frames.push_back(frame);
        }

    }
    else {
        for (int i = settings.starting_frame, j = 0; i < frames_to_use && j < settings.steps_per_datapoint; i += stride) {
            int base_site = i;

            int begin = max(base_site - settings.local_env_size / 2, 0);
            int end = begin + settings.local_env_size;
            STORM_ASSERT(end <= frames_to_use);

            Mat best_frame;
            float best_blur = 0.f;

            vector<float> site_blurness;
            vector<Mat> site_frames;
            for (int k = begin; k < end; ++k) {
                cap.set(CAP_PROP_POS_FRAMES, k);
                Mat frame;
                cap >> frame;

                STORM_ASSERT(!frame.empty());

                float blur = measure_blur(frame);
                site_frames.push_back(frame);

                if (blur >= best_blur) {
                    best_frame = frame;
                    best_blur = blur;
                }
            }

            STORM_ASSERT(!best_frame.empty());
            resize(best_frame, best_frame, Size(FRAME_WIDTH, FRAME_HEIGHT));
            cvtColor(best_frame, best_frame, COLOR_BGR2RGBA);
            frames.push_back(best_frame);
        }
    }

    return true;
}

torch::Tensor VideoAnnotator::marker_to_tensor(int frame_idx)
{
    torch::Tensor markers_tensor = torch::empty({ 7, 2 });
    auto& single_markers = markers[frame_idx];
    for (int i = 0; i < 7; ++i) {
        markers_tensor.index({ i, 0 }) = single_markers[i][0];
        markers_tensor.index({ i, 1 }) = single_markers[i][1];
    }
    return markers_tensor;
}


