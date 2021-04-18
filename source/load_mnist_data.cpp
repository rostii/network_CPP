#include "load_mnist_data.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <iterator>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

std::vector<u_char> read_from_file(std::string file_path);

void load_mnist_data(string file_path)
{
    vector<u_char> data = read_from_file(file_path);
    
    int magic_number_first_second = (data[0] << 8) | data[1];
    if (magic_number_first_second != 0) {
        cerr << "[ERROR] corrupt data.";
        throw;
    }
    
    u_char type_of_data = data[2];
    if (type_of_data != 0x08) {
        cerr << "[ERROR] Wrong data type. Expected unsigned byte.\n";
        throw;
    }
    
    int numb_dims = data[3];
    vector<int> dims;
    for (int i = 0; i < numb_dims; i++) {
        int value = (data[i * 4 + 4] << 24) | (data[i * 4 + 5] << 16) | (data[i * 4 + 6] << 8) | data[i * 4 + 7];
        dims.push_back(value);
    }
    
    Mat img = Mat(28, 28, CV_8UC1, &data[16]);
    namedWindow("test", 0);
    imshow("test", img);
    waitKey();
    destroyAllWindows();
    
}

vector<u_char> read_from_file(string file_path)
{
    ifstream ifile(file_path, ios::binary);
    
    if (not ifile.is_open()) {
        cerr << "[ERROR] Could not open file for reading.\n";
        throw;
    }
    
    vector<u_char> data(istreambuf_iterator<char>(ifile), {});
    
    return data;
}
