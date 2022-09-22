#pragma once

#include <torch/torch.h>

#include <vector>


using namespace torch::indexing;



// TODO: Transform this into a nn::Module
bool drawGMM(const torch::Tensor& single_markers, torch::Tensor& result, int width, int height);

void center_data(torch::Tensor& x);


