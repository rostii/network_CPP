#include "load_mnist_data.hpp"

#include <fstream>
#include <iterator>

using namespace std;

vector<u_char> read_from_file(std::string file_path);

vector<vector<u_char>> load_mnist_data(string file_path)
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
    if (numb_dims != 1 && numb_dims != 3) {
        cerr << "[ERROR] Wrong data dimension. Expected 3 dimensions for images and 1 dimension for labels.\n";
        throw;
    }
    
    vector<int> dims;
    for (int i = 0; i < numb_dims; i++) {
        int value = (data[i * 4 + 4] << 24) | (data[i * 4 + 5] << 16) | (data[i * 4 + 6] << 8) | data[i * 4 + 7];
        dims.push_back(value);
    }
    
    vector<vector<u_char>> loaded_data;
    if (numb_dims == 1) {
        int numb_labels = dims[0];
        
        if (data.size() < 8 + numb_labels) {
            cerr << "[ERROR] Data do not contain all stated labels.\n";
            throw;
        }
        
        for (int i = 0; i < numb_labels; ++i) {
            vector<u_char> temp_labels(10, 0);
            u_char temp_label = data[8 + i];
            temp_labels[temp_label] = 1;
            loaded_data.push_back(temp_labels);
        }
    }
    
    if (numb_dims == 3) {
        int numb_images = dims[0];
        int image_size = dims[1] * dims[2];
        
        if (data.size() < 16 + numb_images * image_size) {
            cerr << "[ERROR] Data do not contain all stated images.\n";
            throw;
        }
        
        for (int i = 0; i < numb_images; ++i) {
            vector<u_char> temp_img;
            for (int j = 0; j < image_size; ++j) {
                temp_img.push_back(data[16 + image_size * i + j]);
            }
            loaded_data.push_back(temp_img);
        }
    }
    
    return loaded_data;
}

vector<u_char> read_from_file(string file_path)
{
    ifstream ifile(file_path, ios::binary);
    
    if (not ifile.is_open()) {
        cerr << "[ERROR] Could not open file for reading.\n";
        throw;
    }
    
    vector<u_char> data(istreambuf_iterator<char>(ifile), {});
    
    if (data.size() < 16) {
        cerr << "[ERROR] Found not enough data in file.\n";
        throw;
    }
    
    return data;
}
