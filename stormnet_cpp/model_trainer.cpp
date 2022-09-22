
#include "torch_data.h"

#include "model_trainer.h"


class MyDataset : public torch::data::datasets::Dataset<MyDataset> {
  using Example = torch::data::Example<>;

  
  torch::Tensor data, labels;


public:
  MyDataset(fs::path data_path, bool train=true) 
  {
    
    torch::Tensor X_train_tensor;
    torch::Tensor X_val_tensor;
    torch::Tensor Y_train_tensor;
    torch::Tensor Y_val_tensor;

    auto raw_data_file = data_path / "data_split.dat";

    
    fs::path raw_data_path = raw_data_file.c_str();
    if(!fs::exists(raw_data_path)) {
      spdlog::info("loading raw data. this might take a while.");

      
      vector<vector<vector<Vector2>>> X;
      vector<Vector3> Y;

      int timesteps_per_sample = 0;

      int number_of_features = 0;


      vector<string> json_files;
      for (auto const& dir_entry : fs::directory_iterator(data_path)) {
        auto file_path = dir_entry.path().string();
        if(file_path.size() > 5 && file_path.substr(file_path.size()-5) == ".json") {
          json_files.push_back(file_path);
        }
      }

      std::sort(json_files.begin(), json_files.end());

      for(string filename : json_files) {
        
        vector<vector<Vector2>> x_session;

        int sticker_count = 0;

        timesteps_per_sample = 0;

        Vector3 cap_rotation;

        ifstream in(filename);

        string line;
        int line_idx = 0;
        while(getline(in, line)) {
          auto my_dict = json::parse(line);
          
          
          timesteps_per_sample ++;


          
          auto sticker_3d_locs = my_dict.at("stickers_locs");
          auto valid_stickers = my_dict.at("valid_stickers");
          if(line_idx == 0) {
            cap_rotation[0] = my_dict.at("cap_rot")["x"].get<float>();
            cap_rotation[1] = my_dict.at("cap_rot")["y"].get<float>();
            cap_rotation[2] = my_dict.at("cap_rot")["z"].get<float>();
          }

          
          double rescalex = 1.0/960.0;
          double rescaley = 1.0/540.0;

          vector<Vector2> sticker_2d_locs;
          for(int i=0; i<valid_stickers.size(); ++i) {
            Vector2 loc;
            loc[0] = sticker_3d_locs[i].at("x").get<float>() * rescalex;
            loc[1] = sticker_3d_locs[i].at("y").get<float>() * rescaley;

            sticker_2d_locs.push_back(loc);
          }

          
          if(line_idx == 0) {
            number_of_features = sticker_2d_locs.size() * 2;
          }

          for(int i=0; i<valid_stickers.size(); ++i) {
            if(!valid_stickers[i].get<bool>()) {
              sticker_2d_locs[i][0] = 0;
              sticker_2d_locs[i][1] = 0;
            } else {
              sticker_count++;
            }
          }

          auto& loc0 = sticker_2d_locs[0];
          auto& loc1 = sticker_2d_locs[1];
          auto& loc2 = sticker_2d_locs[2];

          if((loc0.x() == 0 && loc0.y() == 0) || 
              (loc1.x() == 0 && loc1.y() == 0) || 
              (loc2.x() == 0 && loc2.y() == 0) || line_idx >= 5) {
            loc0 = Vector2 {};
            loc1 = Vector2 {};
            loc2 = Vector2 {};
          }

          
          if(line_idx == 0) {
            if(cap_rotation[0] > 180) {
              cap_rotation[0] -= 360;
            }

            if(cap_rotation[1] > 180) {
              cap_rotation[1] -= 360;
            }

            if(cap_rotation[2] > 180) {
              cap_rotation[2] -= 360;
            }
          }

          

          
          x_session.push_back(move(sticker_2d_locs));

          line_idx++;
        }

        
        X.push_back(move(x_session));
        Y.push_back(move(cap_rotation));

      }


      
      int num_total = X.size();
      int num_valid = (int)(num_total*0.2f);
      int num_train = num_total - num_valid;
      vector<int> indices(num_total);
      std::iota(indices.begin(), indices.end(), 0);

      std::mt19937 g(1);
      std::shuffle(indices.begin(), indices.end(), g);

      vector<vector<vector<Vector2>>> X_train, X_val;
      vector<Vector3> Y_train, Y_val;

      for(int i=0; i<num_valid; ++i) {
        X_val.push_back(X[indices[i]]);
        Y_val.push_back(Y[indices[i]]);
      }

      for(int i=num_valid; i<num_total; ++i) {
        X_train.push_back(X[indices[i]]);
        Y_train.push_back(Y[indices[i]]);
      }

      
      // TODO: Do less copies
      std::vector<float> X_train_vec;
      for(int i=0; i<X_train.size(); ++i) {
        for(int ii=0; ii<X_train[0].size(); ++ii) {
          for(int iii=0; iii<X_train[0][0].size(); ++iii) {
            X_train_vec.push_back(X_train[i][ii][iii][0]);
            X_train_vec.push_back(X_train[i][ii][iii][1]);
          }
        }
      }

      std::vector<float> X_val_vec;
      for(int i=0; i<X_val.size(); ++i) {
        for(int ii=0; ii<X_val[0].size(); ++ii) {
          for(int iii=0; iii<X_val[0][0].size(); ++iii) {
            X_val_vec.push_back(X_val[i][ii][iii][0]);
            X_val_vec.push_back(X_val[i][ii][iii][1]);
          }
        }
      }

      X_train_tensor = 
        torch::from_blob(X_train_vec.data(), {(int)X_train.size(), timesteps_per_sample, number_of_features}, torch::kFloat).clone();
      X_val_tensor = 
        torch::from_blob(X_val_vec.data(), {(int)X_val.size(), timesteps_per_sample, number_of_features}, torch::kFloat).clone();
      Y_train_tensor = 
        torch::from_blob(Y_train.data(), {(int)Y_train.size(), 3}, torch::kFloat).clone();
      Y_val_tensor = 
        torch::from_blob(Y_val.data(), {(int)Y_val.size(), 3}, torch::kFloat).clone();

      
      torch::save({ X_train_tensor, X_val_tensor, Y_train_tensor, Y_val_tensor}, raw_data_path.string());

    }

    
    else {
      spdlog::info("(train data) loading train-validation split from: {}", raw_data_path.string());

      std::vector<torch::Tensor> splits;

      torch::load(splits, raw_data_path.string());

      X_train_tensor = splits[0];
      X_val_tensor = splits[1];
      Y_train_tensor = splits[2];
      Y_val_tensor = splits[3];
    }

    
    if(train) {
      data = X_train_tensor;
      labels = Y_train_tensor;
    } else {
      data = X_val_tensor;
      labels = Y_val_tensor;
    }


    // self.labels[:, 1:3] *= -1
    labels.index({Slice(), Slice(1,3)}) *= -1.f;

  }

  Example get(size_t index) 
  {
    
    auto x = data.index({(int)index}).clone();
    center_data(x);

    
    x.index({Slice(), Slice(0, None, 2)}) *= 256.f;
    x.index({Slice(), Slice(1, None, 2)}) *= 256.f;

    torch::Tensor heatmaps = torch::empty({10, 256, 256});

    for(int i=0; i<10; ++i) {
      torch::Tensor heatmap;
      drawGMM(x[i].reshape({-1, 2}), heatmap, 256, 256);
      heatmaps.index_put_({i}, heatmap);
    }

    return {heatmaps, labels[index]};
  }

  

  torch::optional<size_t> size() const {
    return data.sizes()[0];
  }
};



ModelTrainer::ModelTrainer(Settings& settings) :
  settings(settings)
{
  
  train_synth_output = "";
  train_synth_output_short = "";

  train_log_file = "";
  train_log_file_short = "";

}

ModelTrainer::~ModelTrainer()
{
  
}

void ModelTrainer::draw_gui()
{
  
  ImGui::Text("Model Name: ");

  static char model_name[512];
  ImGui::PushID("Model Name");
  ImGui::InputText("", model_name, IM_ARRAYSIZE(model_name));
  ImGui::PopID();

  ImGui::Text("Synth output directory:");
  ImGui::SameLine();
  if(train_synth_output == "") {
    ImGui::Text("None");
  } else {
    ImGui::Text("%s", train_synth_output_short.c_str());
  }

  ImGui::SameLine();

  ImGui::PushID("Train Load Synth Output");
  if(ImGui::Button("Load")) {
    getDirPath(train_synth_output );
    if(train_synth_output != "") {
      train_synth_output_short = train_synth_output.substr(train_synth_output.find_last_of("/\\") + 1);
    }
  }
  ImGui::PopID();

  ImGui::Text("Training Log File:");
  ImGui::SameLine();
  if(train_log_file == "") {
    ImGui::Text("None");
  } else {
    ImGui::Text("%s", train_log_file_short.c_str());
  }

  ImGui::SameLine();

  ImGui::PushID("Load Train Log File");
  if(ImGui::Button("Load")) {
    nfdfilteritem_t filterItem[] = { { "Log file", "txt" } };
    getPath(filterItem, train_log_file );
    if(train_log_file != "") {
      train_log_file_short = train_log_file.substr(train_log_file.find_last_of("/\\") + 1);
    }
  }
  ImGui::PopID();

  
  if(ImGui::Button("Default Settings")) {
    train_log_file = "./cache/training_log.txt";
    train_log_file_short = train_log_file.substr(train_log_file.find_last_of("/\\") + 1);

    train_synth_output  = "./cache/synth_data";
    train_synth_output_short = train_synth_output.substr(train_synth_output.find_last_of("/\\") + 1);

    strcpy(model_name, "my_new_model");
  }

  ImGui::SameLine();

  bool start_train = false;
  if(ImGui::Button("Train")) {
    
    if(model_name[0] == '\0') {
      ImGui::OpenPopup("Train No Model Name");
    }

    else if(train_log_file == "") {
      ImGui::OpenPopup("Train No Log");
    }

    else if(train_synth_output == "") {
      ImGui::OpenPopup("Train No Synth");
    }

    else if(train_synth_output == "") {
      ImGui::OpenPopup("Train No Synth");
    }

    else {
      start_train = true;
      
      // string render_executable_short = render_data.getExecutablePathShort();
      // if(render_executable_short != "") {
        // bool is_running = isProcessActive(render_executable_short);
        // if(is_running) {
          // ImGui::OpenPopup("Open Process Running Train");
          // start_train = false;
        // }
      // }

      
    }
  }

  if(start_train) {
    if(train_stormnet.joinable()) {
      stop_stormnet_training = true;
      train_stormnet.join();
    }

    stop_stormnet_training = false;
    train_stormnet = std::thread([&]() {
      torch::manual_seed(1);

      
      train_lines.clear();
      spdlog::info("Training Started!");

      
      Convd2d convd2d(10, 3);
      convd2d->to(settings.device);


      
      auto train_set = MyDataset(train_synth_output, true).map(torch::data::transforms::Stack<>());
      // auto train_set = MyDataset(train_synth_output, true);
      auto train_size = train_set.size().value();

      auto val_set = MyDataset(train_synth_output, false).map(torch::data::transforms::Stack<>());
      auto val_size = val_set.size().value();


      
      auto train_loader =
        torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(train_set), settings.stormnet_batchsize);

      auto val_loader =
        torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(val_set), settings.stormnet_batchsize);
        

      
      torch::optim::Adam optimizer(
          convd2d->parameters(), torch::optim::AdamOptions(settings.stormnet_lr)
            .betas(std::make_tuple(settings.stormnet_beta1, 0.999))
            .weight_decay(1e-5f));


      
      for(int epoch=0; epoch<settings.stormnet_num_epochs; ++epoch) {
        if(stop_stormnet_training) {
          break;
        }

        convd2d->train();

        float train_loss_total = 0.f;

        int batch_idx = 0;
        for (auto& batch : *train_loader) {
          if(stop_stormnet_training) {
            break;
          }

          auto data = batch.data.to(settings.device);
          auto targets = batch.target.to(settings.device);

          auto output = convd2d->forward(data);

          // TODO: Make this into a custom loss function
          auto loss = torch::mse_loss(output, targets);
          assert(!std::isnan(loss.item<float>()));

          optimizer.zero_grad();
          loss.backward();
          optimizer.step();

          ostringstream ss;
          ss << "train: epoch: " << epoch << ", batch " << batch_idx << " / " << train_size/settings.stormnet_batchsize << ", loss: " << loss.item<float>();
          train_log_mutex.lock();
          train_lines.push_back(ss.str());
          train_log_mutex.unlock();

          train_loss_total += loss.item<float>();
          batch_idx++;
        }

        train_loss_total /= (train_size/settings.stormnet_batchsize);

        ostringstream ss;
        ss << "train: epoch: " << epoch << ", loss " << train_loss_total; 
        train_log_mutex.lock();
        train_lines.push_back(ss.str());
        train_log_mutex.unlock();

        // Test model
        convd2d->eval();
        {
          torch::NoGradGuard no_grad;
          
          float val_loss_total = 0.f;
          for (auto& batch : *val_loader) {
            auto data = batch.data.to(settings.device);
            auto targets = batch.target.to(settings.device);

            auto output = convd2d->forward(data);

            // TODO: Make this into a custom loss function
            auto loss = torch::mse_loss(output, targets);
            assert(!std::isnan(loss.item<float>()));

            val_loss_total += loss.item<float>();
          }

          val_loss_total /= (val_size/settings.stormnet_batchsize);

          ostringstream ss;
          ss << "validation: epoch: " << epoch << ", loss " << val_loss_total; 
          train_log_mutex.lock();
          train_lines.push_back(ss.str());
          train_log_mutex.unlock();

        }

        
        fs::path path = "models";
        ostringstream oss;
        oss << model_name << ".pth";
        path = path / oss.str();
        torch::save(convd2d, path.string());

      }

      if(stop_stormnet_training) {
        return;
      }

    });
  }

  if(!stop_stormnet_training) {
    ImGui::SameLine();
    if(ImGui::Button("Stop")) {
      stop_stormnet_training = true;
    }
  }

  
  if(ImGui::BeginPopup("Train No Model Name")) {
    ImGui::Text("Missing new model name.");
    ImGui::EndPopup();
  }

  if(ImGui::BeginPopup("Train No Log")) {
    ImGui::Text("Missing log file path.");
    ImGui::EndPopup();
  }

  if(ImGui::BeginPopup("Train No Synth")) {
    ImGui::Text("Missing synthetic data output folder");
    ImGui::EndPopup();
  }

  if(ImGui::BeginPopup("Train No Synth")) {
    ImGui::Text("Missing synthetic data output folder");
    ImGui::EndPopup();
  }

  if(ImGui::BeginPopup("Open Process Running Train")) {
    ImGui::Text("Cannot fine-tune while renderer executable is running.");
    ImGui::EndPopup();
  }

  
  ImGui::BeginChild("Train Output", ImVec2(0, ImGui::GetFontSize() * 10.0f), true);
  train_log_mutex.lock();
  for(const auto& line : train_lines) {
    ImGui::Text("%s", line.c_str());
  }
  train_log_mutex.unlock();

  // Auto-scroll
  if(ImGui::GetScrollY() >= ImGui::GetScrollMaxY()) {
    ImGui::SetScrollHereY(1.0f);
  }

  ImGui::EndChild();

}

void ModelTrainer::join_threads()
{
  
  if(train_stormnet.joinable()) {
    stop_stormnet_training = true;
    train_stormnet.join();
  }

}


