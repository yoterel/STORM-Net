
#include "utils.h"
#include <spdlog/spdlog.h>

#define _USE_MATH_DEFINES
#include <cmath>

#include "torch_data.h"


using namespace torch::indexing;



bool drawGMM(const torch::Tensor& single_markers, torch::Tensor& heatmap, int width, int height)
{

    STORM_ASSERT(single_markers.sizes()[0] == 7);
    STORM_ASSERT(single_markers.sizes()[1] == 2);


    float sigma = 5.f;
    float coeff = 1.f / (sigma * sqrt(2 * M_PI));

    heatmap = torch::zeros({ 256, 256 });

    for (int i = 0; i < 7; ++i) {
        auto vec = single_markers.index({ i });
        float px = vec.index({ 0 }).item<float>();
        float py = vec.index({ 1 }).item<float>();
        if (px != 0 || py != 0) {

            auto x_axis = torch::arange(0, 256);
            auto y_axis = torch::arange(0, 256);
            float x_mean = px * 256.f / width;
            float y_mean = py * 256.f / height;
            auto x_gaussian = coeff * torch::exp(-0.5 * torch::pow((x_axis - x_mean) / sigma, 2));
            auto y_gaussian = coeff * torch::exp(-0.5 * torch::pow((y_axis - y_mean) / sigma, 2));

            x_gaussian = x_gaussian.repeat({ 256, 1 });
            y_gaussian = y_gaussian.unsqueeze(1).repeat({ 1, 256 });

            heatmap += x_gaussian * y_gaussian;

        }
    }

    float heatmap_max = torch::max(heatmap).item<float>();
    float heatmap_min = torch::min(heatmap).item<float>();
    if (heatmap_max > heatmap_min) {
        heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min);
    }

    return true;
}

void center_data(torch::Tensor& x)
{

    auto zero_indices = x == 0;
    auto xvec_cent = torch::true_divide(x.index({ Slice(), Slice(None, None, 2) }).sum(1), (x.index({ Slice(), Slice(None, None, 2) }) != 0).sum(1));
    xvec_cent = torch::nan_to_num(xvec_cent);

    auto yvec_cent = torch::true_divide(x.index({ Slice(), Slice(1, None, 2) }).sum(1), (x.index({ Slice(), Slice(1, None, 2) }) != 0).sum(1));
    yvec_cent = torch::nan_to_num(yvec_cent);


    x.index({ Slice(), Slice(None, None, 2) }) += (0.5 - xvec_cent).unsqueeze(1);
    x.index({ Slice(), Slice(1, None, 2) }) += (0.5 - yvec_cent).unsqueeze(1);

    x.index({ zero_indices }) = 0;
}


