
#include <algorithm>

#include "utils.h"


int getIndex(const std::vector<std::string>& vec, const std::string& str)
{
    auto it = std::find(vec.begin(), vec.end(), str);
    if (it == vec.end()) {
        return -1;
    }
    return std::distance(vec.begin(), it);
}

std::string executeSync(const std::string& cmd)
{

    FILE* pipe;
#ifdef _WIN32
    pipe = _popen(cmd.c_str(), "r");
#else
    pipe = popen(cmd.c_str(), "r");
#endif


    std::array<char, 128> buffer;
    std::string result;

    while (!feof(pipe)) {
        if (fgets(buffer.data(), 128, pipe) != nullptr) {
            result += buffer.data();
        }
    }


#ifdef _WIN32
    _pclose(pipe);
#else
    pclose(pipe);
#endif

    return result;
}

bool isProcessActive(const std::string& name)
{
    std::ostringstream oss;
    oss << "TASKLIST /FI \"imagename eq " << name << "\"";
    std::string out = executeSync(oss.str());
    return out.find(name) != std::string::npos;
}

void executeAsyncStart(const std::string& cmd, FILE** p_pipe)
{

    FILE* pipe;
#ifdef _WIN32
    pipe = _popen(cmd.c_str(), "r");
#else
    pipe = popen(cmd.c_str(), "r");
#endif

    * p_pipe = pipe;
}

void executeAsyncStop(FILE* pipe)
{

#ifdef _WIN32
    _pclose(pipe);
#else
    pclose(pipe);
#endif

}

torch::Tensor CSV_to_tensor(std::ifstream& file, bool hasHeader, torch::Device device) {
    std::vector<std::vector<float>> features;
    const std::regex comma(",");
    std::vector<std::vector<std::string>> csv_data;
    std::string line{};
    // throw first row if header
    if (hasHeader)
    {
        getline(file, line);
    }
    while (file && getline(file, line)) {
        // Tokenize the line and store result in vector. Use range constructor of std::vector
        std::vector<std::string> row{ std::sregex_token_iterator(line.begin(), line.end(), comma,-1), std::sregex_token_iterator() };
        csv_data.push_back(row);
    }
    auto tensor = torch::from_blob(csv_data.data(), { (int)csv_data.size(), (int)csv_data[0].size() }, device);
    return tensor;
}