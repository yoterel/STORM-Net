
#include <fstream>
#include <vector>
#include <spdlog/spdlog.h>

#include "string_utils.h"

#include <cctype>

#include <algorithm>

#include "utils.h"

#include "template.h"


using namespace std;

using namespace torch::indexing;


Template::Template()
{
}
//    reads a template file in format ("name x y z")
//    multiple sessions in same file are assumed to be delimited by a line "*" (and first session starts with it)
//    :return: positions is a list of np array per session, names is a list of lists of names per session.
//             note: if two sensors exists, they are stacked in a nx2x3 array, else nx3 for positions.
Template Template::read(string filename)
{
    vector<string> tokens;
    vector<vector<string>> names{ {} };
    Template tmpl;



    fstream in(filename);
    if (!in.is_open()) {
        spdlog::error("Could not load file: \"{}\"", filename);
        return Template();
    }


    vector<string> lines;
    string line;
    while (getline(in, line)) {
        if (line.size() > 0) {
            lines.push_back(line);
        }
    }

    torch::Tensor sensor_data = torch::empty({ (int)lines.size(), 3 });
    for (int i = 0; i < lines.size(); ++i) {
        auto& line = lines[i];

        split_words(line, tokens);
        if (tokens.size() != 4) {
            split_words(line, tokens, ',');
        }

        sensor_data.index_put_({ i, 0 }, str2float(tokens[1]));
        sensor_data.index_put_({ i, 1 }, str2float(tokens[2]));
        sensor_data.index_put_({ i, 2 }, str2float(tokens[3]));


        string name;
        if (tokens[0].size() > 0 && !isdigit(tokens[0][0])) {
            name = tolower(tokens[0]);
        }
        else {
            name = tokens[0];
        }
        names[0].push_back(name);

    }

    auto it0 = std::find(names[0].begin(), names[0].end(), "0");
    auto it1 = std::find(names[0].begin(), names[0].end(), "1");

    if (it0 == names[0].end() && it1 != names[0].end()) {
        int end = str2int(names[0].back());
        names[0].erase(it1, names[0].end());

        for (int i = 0; i < end; ++i) {
            names[0].push_back(to_string(i));
        }
    }


    torch::Tensor mean = torch::mean(sensor_data, 0);
    tmpl.data_mean.push_back(mean);


    tmpl.data.push_back(sensor_data);


    tmpl.names = move(names);

    tmpl.empty = false;

    return tmpl;
}

// given certain sticker names, converts the nx3 data to the standard coordinate system where:
// x is from left to right ear
// y is from back to front of head
// z is from bottom to top of head
// origin is defined by (x,y,z) = ((lefteye.x+righteye.x) / 2, cz.y, (lefteye.z+righteye.z) / 2)
// scale is cm. if cz is too close to origin in terms of cm, this function scales it to cm (assuming it is inch)
// note: only performs swaps, reflections, translation and possibly scale (no rotation is performed).
// :param names:
// :param data:
// :return: returns the data in the standard coordinate system
Template Template::toStandardCoordinateSystem()
{
    Template new_template(*this);

    auto& template_names = new_template.names[0];
    auto& d0 = new_template.data[0];

    int left_eye = getIndex(template_names, "lefteye");
    int right_eye = getIndex(template_names, "righteye");

    int cz = getIndex(template_names, "cz");

    int left_triangle, right_triangle;
    left_triangle = getIndex(template_names, "left_triangle");
    right_triangle = getIndex(template_names, "right_triangle");

    if (left_triangle == -1 && right_triangle == -1) {
        left_triangle = getIndex(template_names, "fp1");
        right_triangle = getIndex(template_names, "fp2");
    }

    // swap x axis with best candidate
    int x_axis = torch::argmax(
        torch::abs(d0.index({ right_eye }) -
            d0.index({ left_eye }))).item<int>();

    // data[:, [0, x_axis]] = data[:, [x_axis, 0]]
    d0.index_put_({ Slice(), torch::tensor({0,x_axis}) }, d0.index({ Slice(), torch::tensor({x_axis, 0}) }));

    auto eyes_midpoint = (d0.index({ left_eye }) + d0.index({ right_eye })) / 2;

    auto fp1fp2_midpoint = (d0.index({ left_triangle }) + d0.index({ right_triangle })) / 2;

    int z_axis = torch::argmax(
        torch::abs(eyes_midpoint -
            fp1fp2_midpoint)).item<int>();

    if (z_axis != 0) {
        // data[:, [2, z_axis]] = data[:, [z_axis, 2]]
        d0.index_put_({ Slice(), torch::tensor({2,z_axis}) }, d0.index({ Slice(), torch::tensor({z_axis, 2}) }));
    }

    // find reflections
    float xdir = (d0.index({ right_eye, 0 }) - d0.index({ left_eye, 0 })).item<float>();
    float ydir = (d0.index({ left_eye, 1 }) - d0.index({ cz, 1 })).item<float>();
    float zdir = (d0.index({ cz, 2 }) - d0.index({ left_eye, 2 })).item<float>();

    float i = (xdir > 0) * 2 - 1;
    float j = (ydir > 0) * 2 - 1;
    float k = (zdir > 0) * 2 - 1;

    d0.index_put_({ Slice(), 0 }, d0.index({ Slice(), 0 }) * i);
    d0.index_put_({ Slice(), 1 }, d0.index({ Slice(), 1 }) * j);
    d0.index_put_({ Slice(), 2 }, d0.index({ Slice(), 2 }) * k);

    eyes_midpoint = (d0.index({ left_eye }) + d0.index({ right_eye })) / 2;
    auto origin = torch::tensor({ eyes_midpoint[0].item<float>(), d0.index({cz, 1}).item<float>(), eyes_midpoint[2].item<float>() });

    d0 = d0 - origin;

    if (d0.index({ cz, 2 }).item<float>() < 7) {
        d0 *= 2.54f;
    }

    return new_template;
}

// given sticker names and data (nx3),
// rotates data such that x axis is along the vector going from left to right (using 6 fiducials),
// and z is pointing upwards.
// :param names:
// :param data:
// :return:
Template Template::fixYaw()
{
    Template new_template(*this);

    auto& template_names = new_template.names[0];
    auto& d0 = new_template.data[0];

    int left_eye = getIndex(template_names, "lefteye");
    int right_eye = getIndex(template_names, "righteye");
    int left_ear = getIndex(template_names, "leftear");
    int right_ear = getIndex(template_names, "rightear");
    int right_triangle = getIndex(template_names, "right_triangle");
    int left_triangle = getIndex(template_names, "left_triangle");

    auto yaw_vec_1 = (d0.index({ right_eye }) - d0.index({ left_eye })) * torch::tensor({ 1, 1, 0 });
    auto yaw_vec_2 = (d0.index({ right_ear }) - d0.index({ left_ear })) * torch::tensor({ 1, 1, 0 });
    auto yaw_vec_3 = (d0.index({ right_triangle }) - d0.index({ left_triangle })) * torch::tensor({ 1, 1, 0 });

    yaw_vec_1 /= torch::linalg_norm(yaw_vec_1);
    yaw_vec_2 /= torch::linalg_norm(yaw_vec_2);
    yaw_vec_3 /= torch::linalg_norm(yaw_vec_3);

    auto avg = torch::mean(torch::vstack({ yaw_vec_1, yaw_vec_2, yaw_vec_3 }), 0);

    avg /= torch::linalg_norm(avg);
    auto u = avg;
    auto v = torch::tensor({ 0.f, 0.f, 1.f });
    auto w = torch::cross(v, u);

    auto transform = torch::vstack({ u, w, v });

    // torch.dot does not behave like np.dot
    // instead torch.mm has to be used for matrix multiplication
    auto new_data = torch::mm(transform, torch::transpose(d0, 0, 1));
    d0 = new_data.transpose(0, 1);

    return new_template;
}

void Template::writeTemporaryFile(std::string file_path)
{
    std::ofstream out(file_path);

    auto& template_names = names[0];
    auto& d0 = data[0];

    for (int i = 0; i < (int)template_names.size(); ++i) {
        out << template_names[i] << " "
            << d0.index({ i, 0 }).item<float>()
            << d0.index({ i, 1 }).item<float>()
            << d0.index({ i, 2 }).item<float>() << std::endl;
    }
}

Template Template::applyRigidTransform(vector<torch::Tensor>& rs, vector<torch::Tensor>& sc)
{
    auto& template_names = names[0];
    auto& d0 = data[0];

    auto it = std::find(template_names.begin(), template_names.end(), "0");

    torch::Tensor data_origin, data_optodes;
    if (it != template_names.end()) {
        int start = std::distance(template_names.begin(), it);
        data_origin = d0.index({ Slice(None, start) });
        data_optodes = d0.index({ Slice(start,None) });
    }
    else {
        data_origin = torch::empty({ 0, 3 });
        data_optodes = d0;
    }

    // Using a template because it has the same structure,
    // that may not be semantically correct
    Template vid_est;

    if (rs.size() != 1) {
        spdlog::error("rs.size() = {} in applyRigidTransform (should be 1)", rs.size());
        return vid_est;
    }

    auto& rot_mat = rs[0];
    auto& scale_mat = sc[0];

    auto transformed_data_sim = rot_mat.mm(scale_mat.mm(torch::transpose(data_optodes, 0, 1)));
    data_optodes = torch::transpose(transformed_data_sim, 0, 1);

    vid_est.names.push_back(template_names);
    vid_est.data.push_back(torch::vstack({ data_origin, data_optodes }));
    return vid_est;
}

