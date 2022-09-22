
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


