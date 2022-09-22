#pragma once

#include <string>
#include <vector>

#include <sstream>


using namespace std;



void split_words(const string& str, vector<string>& words, char sep=' ');

float str2float(const string& str);

string trim(const string& str);

string tolower(const string& str);

int str2int(const string& str);


