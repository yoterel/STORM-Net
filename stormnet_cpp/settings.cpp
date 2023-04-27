
#include "settings.h"




Settings::Settings(){}
Settings::~Settings(){}
void Settings::draw_gui(bool* show_settings)
{
    ImGui::Begin("Settings", show_settings);
    if (ImGui::CollapsingHeader("Video loading")) {
        ImGui::PushItemWidth(40);
        ImGui::DragInt("Starting frame", &starting_frame);
        ImGui::DragInt("Local env size", &local_env_size);

        ImGui::PopItemWidth();
    }

    if (ImGui::CollapsingHeader("STORM net training")) {
        ImGui::PushItemWidth(80);
        ImGui::InputInt("batch size", &stormnet_batchsize);
        ImGui::InputFloat("learning rate", &stormnet_lr);
        ImGui::InputFloat("beta1", &stormnet_beta1);
        ImGui::InputInt("Num epochs", &stormnet_num_epochs);
        {
            static int device_id = 0;
            const char* items[] = { "CPU", "CUDA" };
            if (ImGui::BeginCombo("Device", items[device_id])) {
                if (ImGui::Selectable(items[0], device_id == 0)) {
                    device = torch::kCPU;
                    device_id = 0;
                }
                if (ImGui::Selectable(items[1], device_id == 1)) {
                    device = torch::kCUDA;
                    device_id = 1;
                }
                ImGui::EndCombo();
            }
        }
        ImGui::PopItemWidth();
    }
    ImGui::End();
}


