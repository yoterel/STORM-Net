#pragma once

#include <string>

#include <regex>
#include <sstream>

#include <torch/torch.h>

#include <algorithm>

#include <fstream>

#include <vector>



using namespace std;

using namespace torch::indexing;


class Template
{
public:

    bool empty = true;

    vector<torch::Tensor> data;

    vector<vector<string>> names;

    vector<torch::Tensor> data_mean;


    Template();

    bool isEmpty() const { return empty; }

    Template toStandardCoordinateSystem();

    Template fixYaw();

    void writeTemporaryFile(std::string file_path);

    Template applyRigidTransform(vector<torch::Tensor>& rs, vector<torch::Tensor>& sc);


    static Template read(string filename, string input_file_format = "");

private:

};


