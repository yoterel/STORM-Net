#pragma once

#include <vector>
#include <string>

#include <cstdio>
#include <array>

#include <sstream>
#include <iostream>

#include <sstream>
#include <iostream>



int getIndex(const std::vector<std::string>& vec, const std::string& str);

std::string executeSync(const std::string& cmd);

bool isProcessActive(const std::string& name);

void executeAsyncStart(const std::string& cmd, FILE** p_pipe);

void executeAsyncStop(FILE* pipe);


#define STORM_ASSERT(cond) if(!(cond)) { spdlog::error(#cond); return false; }


