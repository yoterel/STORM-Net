#pragma once

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



// This could be a singleton eventually
class Settings
{
public:
  
  int starting_frame = 0;
  int local_env_size = 1;

  int stormnet_batchsize = 16;

  float stormnet_lr = 1e-4f;
  float stormnet_beta1 = 0.9f;

  int stormnet_num_epochs = 50;

  torch::DeviceType device = torch::kCPU; // torch::kCUDA;

  int steps_per_datapoint = 10;
public:
  
  Settings();


  ~Settings();

  void draw_gui(bool* show_settings);

};

