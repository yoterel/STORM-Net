#include <Magnum/Math/Color.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/ImGuiIntegration/Context.hpp>
#include <Magnum/Platform/Sdl2Application.h>

#include <opencv2/opencv.hpp>

#include <spdlog/spdlog.h>

#include <opencv2/core/utils/logger.hpp>

#include <Magnum/ImageView.h>
#include <Magnum/PixelFormat.h>
#include <Magnum/GL/TextureFormat.h>
#include <Corrade/Containers/Optional.h>
#include <Corrade/Containers/StringView.h>
#include <Corrade/PluginManager/Manager.h>
#include <Corrade/Utility/Resource.h>
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

#include "arc_ball.h"

#include <Magnum/Image.h>


#include <array>
#include <vector>

#include "torch_data.h"
#include "render_data.h"

#include "utils.h"
#include "template_viewer.h"

#include "video_annotator.h"

#include "model_trainer.h"

#include "settings.h"



using namespace Magnum;
using namespace Math::Literals;


using namespace cv;






class ImGuiExample : public Platform::Application {
public:
    explicit ImGuiExample(const Arguments& arguments);

    void drawEvent() override;

    ~ImGuiExample();


    void viewportEvent(ViewportEvent& event) override;

    void keyPressEvent(KeyEvent& event) override;
    void keyReleaseEvent(KeyEvent& event) override;

    void mousePressEvent(MouseEvent& event) override;
    void mouseReleaseEvent(MouseEvent& event) override;
    void mouseMoveEvent(MouseMoveEvent& event) override;
    void mouseScrollEvent(MouseScrollEvent& event) override;
    void textInputEvent(TextInputEvent& event) override;

private:
    ImGuiIntegration::Context _imgui{ NoCreate };

    bool show_template_viewer = false;
    Color4 _clearColor = 0x72909aff_rgbaf;


    bool show_about = false;

    vector<string> console_lines;
    bool show_console = false;

    bool status_window = false;

    bool show_online_step = false;

    Containers::Optional<ArcBall> arc_ball;

    bool show_camera_window = false;

    bool camera_lagging = true;

    bool show_offline_step = false;

    RenderData render_data;

    TemplateViewer template_viewer;

    VideoAnnotator video_annotator;

    bool show_settings = false;

    ModelTrainer model_trainer;

    Settings settings;

};

ImGuiExample::ImGuiExample(const Arguments& arguments) : Platform::Application{ arguments,
  Configuration{}.setTitle("Cap calibrator")
    .setWindowFlags(Configuration::WindowFlag::Resizable) },

    model_trainer(settings),

    video_annotator(settings),

    template_viewer(GL::defaultFramebuffer.viewport())

{

    // For object picking
    MAGNUM_ASSERT_GL_VERSION_SUPPORTED(GL::Version::GL330);

    _imgui = ImGuiIntegration::Context(Vector2{ windowSize() } / dpiScaling(),
        windowSize(), framebufferSize());

    // The only thing needed to enable docking in imgui
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    // Set up proper blending to be used by ImGui. There's a great chance
    //   you'll need this exact behavior for the rest of your scene. If not, set
    //   this only for the drawFrame() call.
    GL::Renderer::setBlendEquation(GL::Renderer::BlendEquation::Add,
        GL::Renderer::BlendEquation::Add);
    GL::Renderer::setBlendFunction(GL::Renderer::BlendFunction::SourceAlpha,
        GL::Renderer::BlendFunction::OneMinusSourceAlpha);

    // Have some sane speed, please
    setMinimalLoopPeriod(16);



    utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR);




    NFD_Init();


    template_viewer.setup_geometry();


    template_viewer.setup_shader();



    const Vector3 eye = Vector3::zAxis(60.0f);
    const Vector3 center{};
    const Vector3 up = Vector3::yAxis();

    arc_ball.emplace(eye, center, up, 35.0_degf, windowSize());


    template_viewer.setup_framebuffer();


    video_annotator.setup_dlib();


    video_annotator.load_classifier();


}

void ImGuiExample::drawEvent() {

    video_annotator.update();


    arc_ball->updateTransformation();




    Matrix4 projectionMatrix = Matrix4::perspectiveProjection(35.0_degf,
        Vector2{ windowSize() }.aspectRatio(), 0.01f, 1000.0f);


    Matrix4 viewMatrix = arc_ball->viewMatrix();

    template_viewer.draw_offscreen(projectionMatrix, viewMatrix);



    GL::defaultFramebuffer.clear(GL::FramebufferClear::Color | GL::FramebufferClear::Depth);


    template_viewer.blit();


    _imgui.newFrame();


    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Exit")) {
                exit(0);
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("View")) {

            ImGui::MenuItem("Console", NULL, &show_console);

            ImGui::MenuItem("Status", NULL, &status_window);

            ImGui::MenuItem("Template Model Viewer", NULL, &show_template_viewer);

            ImGui::MenuItem("Online Step", NULL, &show_online_step);

            ImGui::MenuItem("Camera Controls", NULL, &show_camera_window);

            ImGui::MenuItem("Offline Step", NULL, &show_offline_step);

            ImGui::MenuItem("Settings", NULL, &show_settings);

            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Help")) {


            if (ImGui::MenuItem("About"))
            {
                show_about = true;
            }

            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }



    ImGui::DockSpaceOverViewport(ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);


    // Enable text input, if needed
    if (ImGui::GetIO().WantTextInput && !isTextInputActive())
        startTextInput();
    else if (!ImGui::GetIO().WantTextInput && isTextInputActive())
        stopTextInput();

    if (show_template_viewer) {

        template_viewer.draw_gui(&show_template_viewer);

    }



    if (status_window) {
        ImGui::Begin("Status", &status_window);

        ImGui::Text("dlib:");
        ImGui::SameLine();
        if (!video_annotator.is_dlib_loaded()) {
            ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Not ready.");
        }
        else {
            ImGui::Text("OK.");
        }

        ImGui::Text("haarcascade:");
        ImGui::SameLine();
        if (!video_annotator.is_haarcascade_loaded()) {
            ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Not ready.");
        }
        else {
            ImGui::Text("OK.");
        }

        ImGui::End();
    }

    // @private_data+=
    // bool showDemoWindow = true;

    // @show_demo_window+=
    // if(showDemoWindow) {
      // ImGui::ShowDemoWindow(&showDemoWindow);
    // }


    if (show_online_step) {

        video_annotator.draw_gui(&show_online_step);

    }


    if (show_offline_step) {
        ImGui::Begin("Offline Step", &show_offline_step);
        if (ImGui::CollapsingHeader("Render Data")) {

            render_data.draw_gui();

        }

        if (ImGui::CollapsingHeader("Train STORM-NET")) {

            model_trainer.draw_gui();

        }
        ImGui::End();
    }


    if (show_camera_window) {
        ImGui::Begin("Camera", &show_camera_window);

        ImGui::Checkbox("camera_lagging", &camera_lagging);
        if (camera_lagging) {
            arc_ball->setLagging(0.85f);
        }
        else {
            arc_ball->setLagging(0.f);
        }

        if (ImGui::Button("reset")) {
            arc_ball->reset();
        }

        ImGui::End();
    }



    if (show_settings) {

        settings.draw_gui(&show_settings);
    }


    if (show_console) {
        ImGui::Begin("Console", &show_console);

        static char console_buffer[512];
        ImGui::InputText("cmd", console_buffer, IM_ARRAYSIZE(console_buffer));
        ImGui::SameLine();
        if (ImGui::Button("run")) {
            string out = executeSync(console_buffer);

            istringstream iss(out);
            string line;
            while (std::getline(iss, line)) {
                console_lines.push_back(move(line));
            }
        }


        ImGui::BeginChild("Console Output", ImVec2(0, ImGui::GetFontSize() * 10.0f), true);
        for (const auto& line : console_lines) {
            ImGui::Text("%s", line.c_str());
        }
        ImGui::EndChild();

        ImGui::End();
    }


    if (show_about) {
        ImGui::Begin("About", &show_about);
        ImGui::Text(
            "STORM-Net: Simple and Timely Optode Registration Method for Functional Near-Infrared Spectroscopy (FNIRS).\n"
            "Research: Yotam Erel, Sagi Jaffe-Dax, Yaara Yeshurun-Dishon, Amit H. Bermano\n"
            "Implementation: Yotam Erel, Julien Burkhard\n"
            "This program is free for personal, non-profit or academic use.\n"
            "All Rights Reserved.");

        ImGui::Text("https://github.com/yoterel/STORM-Net");

        ImGui::End();
    }

    /* Update application cursor */
    _imgui.updateApplicationCursor(*this);

    // Set appropriate states. If you only draw ImGui, it is sufficient to
    //   just enable blending and scissor test in the constructor.
    GL::Renderer::enable(GL::Renderer::Feature::Blending);
    GL::Renderer::enable(GL::Renderer::Feature::ScissorTest);
    GL::Renderer::disable(GL::Renderer::Feature::FaceCulling);
    GL::Renderer::disable(GL::Renderer::Feature::DepthTest);

    _imgui.drawFrame();

    // Reset state. Only needed if you want to draw something else with
    //  different state after.
    GL::Renderer::enable(GL::Renderer::Feature::DepthTest);
    GL::Renderer::enable(GL::Renderer::Feature::FaceCulling);
    GL::Renderer::disable(GL::Renderer::Feature::ScissorTest);
    GL::Renderer::disable(GL::Renderer::Feature::Blending);


    swapBuffers();
    redraw();
}

void ImGuiExample::viewportEvent(ViewportEvent& event) {
    GL::defaultFramebuffer.setViewport({ {}, event.framebufferSize() });


    template_viewer.resize(event.framebufferSize());


    _imgui.relayout(Vector2{ event.windowSize() } / event.dpiScaling(),
        event.windowSize(), event.framebufferSize());
}

void ImGuiExample::keyPressEvent(KeyEvent& event) {
    if (_imgui.handleKeyPressEvent(event)) return;
}

void ImGuiExample::keyReleaseEvent(KeyEvent& event) {
    if (_imgui.handleKeyReleaseEvent(event)) return;
}

void ImGuiExample::mousePressEvent(MouseEvent& event) {
    if (_imgui.handleMousePressEvent(event)) return;

    SDL_CaptureMouse(SDL_TRUE);
    arc_ball->initTransformation(event.position());

    event.setAccepted();

}

void ImGuiExample::mouseReleaseEvent(MouseEvent& event) {
    if (_imgui.handleMouseReleaseEvent(event)) return;

    SDL_CaptureMouse(SDL_FALSE);


    const Vector2i position = event.position() * Vector2 {framebufferSize()} / Vector2{ windowSize() };
    const Vector2i fbPosition{ position.x(), GL::defaultFramebuffer.viewport().sizeY() - position.y() - 1 };
    template_viewer.sensor_pick(fbPosition);

}

void ImGuiExample::mouseMoveEvent(MouseMoveEvent& event) {
    if (_imgui.handleMouseMoveEvent(event)) return;

    if (!event.buttons()) return;

    if (event.modifiers() & MouseMoveEvent::Modifier::Shift)
        arc_ball->translate(event.position());
    else arc_ball->rotate(event.position());

    event.setAccepted();

}

void ImGuiExample::mouseScrollEvent(MouseScrollEvent& event) {
    if (_imgui.handleMouseScrollEvent(event)) {
        /* Prevent scrolling the page */
        event.setAccepted();
        return;
    }

    const Float delta = event.offset().y();
    if (Math::abs(delta) < 1.0e-2f) return;

    arc_ball->zoom(delta);

    event.setAccepted();

}

void ImGuiExample::textInputEvent(TextInputEvent& event) {
    if (_imgui.handleTextInputEvent(event)) return;
}


ImGuiExample::~ImGuiExample()
{

    render_data.join_threads();

    video_annotator.join_threads();

    model_trainer.join_threads();

}


MAGNUM_APPLICATION_MAIN(ImGuiExample)


