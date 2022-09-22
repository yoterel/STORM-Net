
#include <Magnum/PixelFormat.h>

#include "template_viewer.h"


using namespace std;

using namespace Magnum;
using namespace Math::Literals;



TemplateViewer::TemplateViewer(Range2Di& viewport) :

    framebuffer{ GL::defaultFramebuffer.viewport() }
{

}

TemplateViewer::~TemplateViewer()
{

}

void TemplateViewer::draw_gui(bool* show_template_viewer)
{
    ImGui::Begin("Template Viewer", show_template_viewer, ImGuiWindowFlags_MenuBar);

    if (ImGui::BeginMenuBar())
    {
        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::MenuItem("Open")) {

                static string template_path;
                nfdfilteritem_t filterItem[] = { { "Template file", "txt" } };

                if (getPath(filterItem, template_path)) {
                    tmpl = Template::read(template_path);
                }

            }
            ImGui::EndMenu();
        }
        ImGui::EndMenuBar();
    }

    if (tmpl.isEmpty()) {
        ImGui::Text("No template loaded.");
    }
    else {

        ImGui::Text("Num sessions: %d", tmpl.data.size());

        static ImGuiTableFlags table_flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;

        ImGui::Text("Means");
        if (ImGui::BeginTable("means", 3, table_flags)) {
            for (int i = 0; i < tmpl.data_mean.size(); ++i) {
                for (int j = 0; j < 3; ++j) {
                    ImGui::TableNextColumn();
                    ImGui::Text("%g", tmpl.data_mean[i].index({ j }).item<float>());
                }
            }

            ImGui::EndTable();
        }


        if (selected != -1) {
            ImGui::Text("Selected: %s", tmpl.names[0][selected].c_str());
        }


        if (ImGui::CollapsingHeader("Data")) {
            if (ImGui::BeginTable("data", 4, table_flags)) {
                for (int i = 0; i < tmpl.data[0].sizes()[0]; ++i) {
                    ImGui::TableNextColumn();
                    ImGui::Text("%s", tmpl.names[0][i].c_str());
                    for (int j = 0; j < 3; ++j) {
                        ImGui::TableNextColumn();
                        ImGui::Text("%g", tmpl.data[0].index({ i, j }).item<float>());
                    }
                }

                ImGui::EndTable();
            }
        }

    }

    ImGui::End();
}

void TemplateViewer::setup_geometry()
{

    sphere = MeshTools::compile(Primitives::uvSphereSolid(16, 16));
    grid = MeshTools::compile(Primitives::grid3DWireframe({ 15, 15 }));


}

void TemplateViewer::draw_offscreen(Matrix4& projectionMatrix, Matrix4& viewMatrix)
{

    framebuffer
        .clearColor(0, Color3{ 0.125f })
        .clearColor(1, Vector4ui{})
        .clearDepth(1.0f)
        .bind();


    if (!tmpl.isEmpty()) {

        shader.setProjectionMatrix(projectionMatrix)
            .setLightPositions({ {0.0f, 0.0f, 1.0f, 0.0f} });


        

        auto& data = tmpl.data;

        const Vector3 centerMean(
            tmpl.data_mean[0].index({ 0 }).item<float>(),
            tmpl.data_mean[0].index({ 1 }).item<float>(),
            tmpl.data_mean[0].index({ 2 }).item<float>());
        float minz = 666.0;
        for (int i = 0; i < data[0].sizes()[0]; ++i) {
            const Vector3 sensorCenter(
                data[0].index({ i,0 }).item<float>(),
                data[0].index({ i,1 }).item<float>(),
                data[0].index({ i,2 }).item<float>());
            if (sensorCenter[2] < minz)
            {
                minz = sensorCenter[2];
            }
            const Matrix4 transformation = viewMatrix * Matrix4::rotationY(180.0_degf) * Matrix4::rotationX(-90.0_degf) * Matrix4::translation(sensorCenter - centerMean) * Matrix4::scaling(Vector3(0.5f));


            Color3 c;
            if (i == selected) {
                c = Color3{ 1.0f, 0.0f, 0.0f };
            }
            else {
                c = Color3{ 1.0f, 1.0f, 1.0f };
            }


            shader.setDiffuseColor(c)
                .setTransformationMatrix(transformation)
                .setNormalMatrix(transformation.normalMatrix())
                .setObjectId(i)
                .draw(sphere);

        }
        const Vector3 grid_center(
            centerMean[0],
            centerMean[1],
            minz);
        const Matrix4 grid_transform = viewMatrix * Matrix4::rotationY(180.0_degf) * Matrix4::rotationX(-90.0_degf) * Matrix4::translation(grid_center - centerMean) * Matrix4::scaling(Vector3(10.0f));
        shader.setDiffuseColor(Color3{ 1.0f, 1.0f, 1.0f })
            .setTransformationMatrix(grid_transform)
            .setNormalMatrix(grid_transform.normalMatrix())
            .setObjectId(data[0].sizes()[0])
            .draw(grid);

    }

}


void TemplateViewer::setup_shader()
{

    shader.setSpecularColor(Color3(1.0f))
        .setShininess(20);

}

void TemplateViewer::setup_framebuffer()
{

    color.setStorage(GL::RenderbufferFormat::RGBA8, GL::defaultFramebuffer.viewport().size());
    objectId.setStorage(GL::RenderbufferFormat::R32UI, GL::defaultFramebuffer.viewport().size());
    depth.setStorage(GL::RenderbufferFormat::DepthComponent24, GL::defaultFramebuffer.viewport().size());

    framebuffer.attachRenderbuffer(GL::Framebuffer::ColorAttachment{ 0 }, color)
        .attachRenderbuffer(GL::Framebuffer::ColorAttachment{ 1 }, objectId)
        .attachRenderbuffer(GL::Framebuffer::BufferAttachment::Depth, depth)
        .mapForDraw({ {Shaders::PhongGL::ColorOutput, GL::Framebuffer::ColorAttachment{0}},
            {Shaders::PhongGL::ObjectIdOutput, GL::Framebuffer::ColorAttachment{1}} });
    CORRADE_INTERNAL_ASSERT(framebuffer.checkStatus(GL::FramebufferTarget::Draw) == GL::Framebuffer::Status::Complete);

}

void TemplateViewer::blit()
{

    GL::AbstractFramebuffer::blit(framebuffer, GL::defaultFramebuffer,
        framebuffer.viewport(), GL::FramebufferBlit::Color);

}

void TemplateViewer::sensor_pick(Vector2i fbPosition)
{


    // Read object ID at given click position, and then switch to the color
    //   attachment again so drawEvent() blits correct buffer
    framebuffer.mapForRead(GL::Framebuffer::ColorAttachment{ 1 });
    Image2D data = framebuffer.read(
        Range2Di::fromSize(fbPosition, { 1, 1 }),
        { PixelFormat::R32UI });
    framebuffer.mapForRead(GL::Framebuffer::ColorAttachment{ 0 });

    UnsignedInt id = data.pixels<UnsignedInt>()[0][0];
    if (id > 0) {
        selected = id;
    }

}

void TemplateViewer::resize(Vector2i framebufferSize)
{

    color.setStorage(GL::RenderbufferFormat::RGBA8, framebufferSize);
    objectId.setStorage(GL::RenderbufferFormat::R32UI, framebufferSize);
    depth.setStorage(GL::RenderbufferFormat::DepthComponent24, framebufferSize);

    framebuffer.setViewport({ {}, framebufferSize });

}


