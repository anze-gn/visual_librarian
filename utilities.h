#ifndef __UTILITIES_H
#define __UTILITIES_H

#include <string>
#include <vector>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

string join(const string& root, const string& path);

vector<string> &split(const string &s, char delimiters, vector<string> &elements);
vector<string> split(const string &s, char delimiters);

void read(const FileNode& node, vector<string>& vector);

#endif
