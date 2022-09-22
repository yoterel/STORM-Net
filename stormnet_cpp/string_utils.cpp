
#include <cctype>

#include "string_utils.h"


using namespace std;



void split_words(const string& str, vector<string>& words, char sep)
{
    words.clear();

    string word;
    for (char c : str) {
        if (c == sep) {
            if (word != "") {
                words.push_back(word);
                word = "";
            }
        }
        else {
            word += c;
        }
    }

    if (word != "") {
        words.push_back(word);
    }
}

float str2float(const string& str)
{
    float value;
    istringstream iss(str);
    iss >> value;
    return value;
}

string trim(const string& str)
{
    int s = 0, e = str.size() - 1;
    while (isspace(str[s]) && s < str.size()) { ++s; }
    while (isspace(str[e]) && e >= 0) { --e; }

    if (s > e) { return ""; }
    return str.substr(s, e - s + 1);
}

string tolower(const string& str)
{
    string result;
    for (char c : str) {
        result += tolower(c);
    }
    return result;
}

int str2int(const string& str)
{
    int value;
    istringstream iss(str);
    iss >> value;
    return value;
}


