#pragma once

#include <Magnum/ImGuiIntegration/Context.hpp>
#include <Magnum/Platform/Sdl2Application.h>

#include "template.h"
#include "file_dialog.h"

#include <cmath>

#include <Magnum/Math/Matrix4.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/Primitives/UVSphere.h>
#include <Magnum/Primitives/Grid.h>
#include <Magnum/Shaders/PhongGL.h>
#include <Magnum/Trade/MeshData.h>

#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Framebuffer.h>
#include <Magnum/GL/Renderbuffer.h>
#include <Magnum/GL/RenderbufferFormat.h>
#include <Magnum/GL/Version.h>
#include <nfd.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Image.h>
#include <Magnum/Trade/ImageData.h>


using namespace std;

using namespace Magnum;
using namespace Math::Literals;


class TemplateViewer
{
private:

    Template tmpl;

    GL::Mesh sphere{ NoCreate }, grid{ NoCreate };

    Shaders::PhongGL shader{ Shaders::PhongGL::Flag::ObjectId };

    GL::Framebuffer framebuffer;
    GL::Renderbuffer color, objectId, depth;

    int selected = -1;


public:

    TemplateViewer(Range2Di& viewport);

    ~TemplateViewer();

    void draw_gui(bool* show_template_viewer);

    void setup_geometry();

    void draw_offscreen(Matrix4& projectionMatrix, Matrix4& viewMatrix);

    void setup_shader();

    void setup_framebuffer();

    void blit();

    void sensor_pick(Vector2i fbPosition);

    void resize(Vector2i framebufferSize);

};


