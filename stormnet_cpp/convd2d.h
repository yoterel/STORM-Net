#pragma once

#include <torch/torch.h>



struct Convd2dImpl : torch::nn::SequentialImpl {
  Convd2dImpl(int input_size, int output_size) {
    using namespace torch::nn;

    push_back(Conv2d(torch::nn::Conv2dOptions(input_size, 64, 3).padding(1)));
    push_back(BatchNorm2d(64));
    push_back(MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
    push_back(Conv2d(torch::nn::Conv2dOptions(64, 128, 3).padding(1)));
    push_back(BatchNorm2d(128));
    push_back(ReLU());
    push_back(MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
    push_back(Conv2d(torch::nn::Conv2dOptions(128, 256, 3).padding(1)));
    push_back(BatchNorm2d(256));
    push_back(ReLU());
    push_back(MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
    push_back(Conv2d(torch::nn::Conv2dOptions(256, 512, 3).padding(1)));
    push_back(BatchNorm2d(512));
    push_back(ReLU());
    push_back(MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
    push_back(Flatten());
    push_back(Linear(131072, 16));
    push_back(ReLU());
    push_back(Linear(16, output_size));
  }
};
TORCH_MODULE(Convd2d);

