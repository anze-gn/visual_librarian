#include <sstream>
#include <iostream>
#include "utilities.h"

#ifdef _WIN32

string join(const string& root, const string& path) {
	if (root[root.size() - 1] == '\\')
		return root + path;
	else if (path[0] == '\\')
		return path;
	else return root + "\\" + path;
}

#else

string join(const string& root, const string& path) {
	if (path[0] == '/')
		return path;
	else if (root[root.size() - 1] == '/')
		return root + path;
	else return root + "/" + path;
}

#endif

vector<string> &split(const string &s, char delimiter, vector<string> &elements) {
	stringstream ss(s);
	string item;
	while (getline(ss, item, delimiter))
		elements.push_back(item);
	return elements;
}

vector<string> split(const string &s, char delimiter) {
	vector<string> elements;
	split(s, delimiter, elements);
	return elements;
}

void read(const FileNode& node, vector<string>& out) {
	if (node.isSeq()) {
		for (FileNodeIterator it = node.begin(); it != node.end(); it++) {
			out.push_back(string(*it));
		}
	}
}