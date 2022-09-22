
#include "file_dialog.h"

#include "utils.h"

#include <spdlog/spdlog.h>

#include <fstream>

#include "render_data.h"


using namespace std;

namespace fs = std::filesystem;

using namespace std::chrono_literals;


RenderData::RenderData()
{

    render_data_model = "";
    render_data_model_short = "";

    render_log_file = "";
    render_log_file_short = "";


    render_synth_output = "";
    render_synth_output_short = "";

    render_executable = "";
    render_executable_short = "";

}

RenderData::~RenderData()
{

}

void RenderData::draw_gui()
{

    bool start_render = false;

    ImGui::Text("Template:");
    ImGui::SameLine();
    if (render_data_model == "") {
        ImGui::Text("None");
    }
    else {
        ImGui::Text("%s", render_data_model_short.c_str());
    }

    ImGui::SameLine();

    if (ImGui::Button("Load")) {
        nfdfilteritem_t filterItem[] = { { "Template file", "txt" } };
        getPath(filterItem, render_data_model);
        if (render_data_model != "") {
            render_data_model_short = render_data_model.substr(render_data_model.find_last_of("/\\") + 1);
        }
    }

    ImGui::Text("Renderer Log File:");
    ImGui::SameLine();
    if (render_log_file == "") {
        ImGui::Text("None");
    }
    else {
        ImGui::Text("%s", render_log_file_short.c_str());
    }

    ImGui::SameLine();

    ImGui::PushID("Load Log File");
    if (ImGui::Button("Load")) {
        nfdfilteritem_t filterItem[] = { { "Log file", "txt" } };
        getPath(filterItem, render_log_file);
        if (render_log_file != "") {
            render_log_file_short = render_log_file.substr(render_log_file.find_last_of("/\\") + 1);
        }
    }
    ImGui::PopID();

    ImGui::Text("Synth output directory:");
    ImGui::SameLine();
    if (render_synth_output == "") {
        ImGui::Text("None");
    }
    else {
        ImGui::Text("%s", render_synth_output_short.c_str());
    }

    ImGui::SameLine();

    ImGui::PushID("Load Synth Output");
    if (ImGui::Button("Load")) {
        getDirPath(render_synth_output);
        if (render_synth_output != "") {
            render_synth_output_short = render_synth_output.substr(render_synth_output.find_last_of("/\\") + 1);
        }
    }
    ImGui::PopID();

    ImGui::Text("Renderer Executable:");
    ImGui::SameLine();
    if (render_executable == "") {
        ImGui::Text("None");
    }
    else {
        ImGui::Text("%s", render_executable_short.c_str());
    }

    ImGui::SameLine();


    ImGui::PushID("Load Render Executable");
    if (ImGui::Button("Load")) {
        nfdfilteritem_t filterItem[] = { { "Application", "exe" } };
        getPath(filterItem, render_executable);
        if (render_executable != "") {
            render_executable_short = render_executable.substr(render_executable.find_last_of("/\\") + 1);
        }
    }
    ImGui::PopID();

    ImGui::Text("Iterations:");
    ImGui::SameLine();

    ImGui::PushID("Render Iterations");
    ImGui::PushItemWidth(120);
    ImGui::InputInt("", &render_iterations);
    ImGui::PopItemWidth();
    ImGui::PopID();


    if (ImGui::Button("Render")) {

        if (render_data_model == "") {
            ImGui::OpenPopup("Render No Template");
        }

        else if (render_log_file == "") {
            ImGui::OpenPopup("Render No Template");
        }

        else if (render_synth_output == "") {
            ImGui::OpenPopup("Render No Synth Output");
        }

        else if (render_executable == "") {
            ImGui::OpenPopup("Render No Executable");
        }

        else if (render_iterations < 1) {
            ImGui::OpenPopup("Render No Iteration");
        }


        else {
            start_render = true;

            // Create parent directory and ignores if exist already
            fs::create_directories(render_synth_output);


            bool dir_empty = true;
            for (auto const& dir_entry : std::filesystem::directory_iterator(render_synth_output)) {
                dir_empty = false;
                break;
            }

            if (!dir_empty) {
                ImGui::OpenPopup("Dir Empty");
                start_render = false;
            }


            bool is_running = isProcessActive(render_executable_short);
            if (is_running) {
                ImGui::OpenPopup("Open Process Running");
                start_render = false;
            }

        }
    }


    ImGui::SameLine();
    if (ImGui::Button("Set Default")) {
        render_log_file = ".\\cache\\render_log.txt";
        render_log_file_short = render_log_file.substr(render_log_file.find_last_of("/\\") + 1);

        render_synth_output = ".\\cache\\synth_data";
        render_synth_output_short = render_synth_output.substr(render_synth_output.find_last_of("/\\") + 1);

        render_executable = ".\\..\\DataSynth\\windows_build\\DataSynth.exe";
        render_executable_short = render_executable.substr(render_executable.find_last_of("/\\") + 1);

        render_data_model = ".\\..\\example_models\\example_model.txt";
        render_data_model_short = render_data_model.substr(render_data_model.find_last_of("/\\") + 1);


        render_iterations = 10000;

    }



    if (ImGui::BeginPopup("Render No Template")) {
        ImGui::Text("Missing template model file.");
        ImGui::EndPopup();
    }

    if (ImGui::BeginPopup("Render No Log")) {
        ImGui::Text("Missing log file path.");
        ImGui::EndPopup();
    }

    if (ImGui::BeginPopup("Render No Synth Output")) {
        ImGui::Text("Missing synthetic data output folder");
        ImGui::EndPopup();
    }

    if (ImGui::BeginPopup("Render No Executable")) {
        ImGui::Text("Renderer executable already running.");
        ImGui::EndPopup();
    }

    if (ImGui::BeginPopup("Render No Iteration")) {
        ImGui::Text("Number of iterations invalid.");
        ImGui::EndPopup();
    }

    if (ImGui::BeginPopup("Open Process Running")) {
        ImGui::Text("Renderer executable already running.");
        ImGui::EndPopup();
    }



    if (ImGui::BeginPopup("Dir Empty")) {
        ImGui::Text("The synthetic data folder you provided as output\nis not empty, this action will delete any content\nin the folder. Proceed?");
        if (ImGui::Button("Yes")) {

            for (auto const& dir_entry : std::filesystem::directory_iterator(render_synth_output)) {
                fs::remove_all(dir_entry.path());
            }

            ImGui::CloseCurrentPopup();
            start_render = true;
        }

        ImGui::SameLine();
        if (ImGui::Button("No")) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }



    if (!stop_log_logthread) {
        ImGui::SameLine();
        if (ImGui::Button("Stop")) {
            stop_log_logthread = true;
        }
    }


    if (start_render) {

        Template render_tmpl = Template::read(render_data_model);


        render_tmpl = render_tmpl.toStandardCoordinateSystem();
        render_tmpl = render_tmpl.fixYaw();


        fs::path cache = "cache";
        fs::path temporary_path = cache / "template_transformed.txt";
        fs::create_directories(cache);
        render_tmpl.writeTemporaryFile(temporary_path.string());


        {
            std::ofstream log_out(render_log_file.c_str());
            log_out.close();
        }


        ostringstream oss;

        oss << "\"" << render_executable << "\""
            << " -logFile " << render_log_file
            << " -iterations " << render_iterations
            << " -input_file " << temporary_path.string()
            << " -output_folder " << render_synth_output
            << " -batchmode";

        spdlog::info("cmd.exe : {}", oss.str());
        executeAsyncStart(oss.str(), &render_data_pipe);


        if (log_thread.joinable()) {
            stop_log_logthread = true;
            log_thread.join();
        }
        log_lines.clear();

        stop_log_logthread = false;

        log_thread = thread([&]() {

            string line;
            int num_lines = 0;
            while (!stop_log_logthread) {
                std::ifstream log(render_log_file);

                int i = 0;
                while (std::getline(log, line)) {
                    if (line.find("Done") != string::npos) {
                        stop_log_logthread = true;
                        break;
                    }

                    if (i > num_lines) {
                        log_mutex.lock();
                        log_lines.push_back(move(line));
                        log_mutex.unlock();
                        num_lines++;
                    }

                    i++;
                }

                if (!stop_log_logthread) {
                    this_thread::sleep_for(500ms);
                }
            }


            if (render_data_pipe) {
                executeAsyncStop(render_data_pipe);
                render_data_pipe = nullptr;
            }
            });

    }


    ImGui::BeginChild("log_viewer", ImVec2(0, ImGui::GetFontSize() * 10.0f), true);

    log_mutex.lock();
    for (const auto& line : log_lines) {
        ImGui::Text("%s", line.c_str());
    }
    log_mutex.unlock();


    // Auto-scroll
    if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY()) {
        ImGui::SetScrollHereY(1.0f);
    }

    ImGui::EndChild();

}

void RenderData::join_threads()
{

    if (log_thread.joinable()) {
        stop_log_logthread = true;
        log_thread.join();
    }

    if (render_data_pipe) {
        executeAsyncStop(render_data_pipe);
        render_data_pipe = nullptr;
    }

}

string RenderData::getExecutablePathShort()
{
    return render_executable_short;
}

