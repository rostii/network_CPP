#include "load_mnist_data.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <fstream>
#include <iterator>

using namespace std;
using namespace cv;

dataset_t load_data(const string& src_path, const string& images_path, const string& labels_path)
{
    bool examine_loaded_data = false;
    
    vector<dataimage_t> images = load_image_data(src_path + images_path);
    vector<datalabel_t> labels = load_label_data(src_path + labels_path);
    
    dataset_t data;
    for (size_t i = 0; i < images.size(); ++i) {
        data.push_back(make_tuple(images[i], labels[i]));
    }
    
    if (examine_loaded_data) {
        int idx = 1061;
        for (int i = 0; i < 10; ++i) {
            if (labels[idx][i] == 1) {
                cout << "[INFO] Image label is: " << i << "\n";
            }
        }

        Mat img = Mat(28, 28, CV_8UC1, &images[idx][0]);
        namedWindow("Image", 0);
        imshow("Image", img);
        waitKey();
        destroyAllWindows();
    }
    
    return data;
}

vector<dataimage_t> load_image_data(const string& file_path)
{
    datastream_t data = read_from_file(file_path);
    
    int magic_number_first_second = (data[0] << 8) | data[1];
    if (magic_number_first_second != 0) {
        cerr << "[ERROR] corrupt data.";
        throw;
    }
    
    unsigned char type_of_data = data[2];
    if (type_of_data != 0x08) {
        cerr << "[ERROR] Wrong data type. Expected unsigned byte.\n";
        throw;
    }

    int numb_dims = data[3];
    if (numb_dims != 3) {
        cerr << "[ERROR] Wrong data dimension. Expected 3 dimensions for image data.\n";
        throw;
    }

    vector<int> dims;
    for (int i = 0; i < numb_dims; i++) {
        int value = (data[i * 4 + 4] << 24) | (data[i * 4 + 5] << 16) | (data[i * 4 + 6] << 8) | data[i * 4 + 7];
        dims.push_back(value);
    }

    vector<dataimage_t> loaded_image_data;
    int numb_images = dims[0];
    int image_size = dims[1] * dims[2];

    if (data.size() < 16 + numb_images * image_size) {
        cerr << "[ERROR] Data do not contain all stated images.\n";
        throw;
    }

    for (int i = 0; i < numb_images; ++i) {
        dataimage_t temp_img;
        for (int j = 0; j < image_size; ++j) {
            temp_img.push_back(static_cast<double>(data[16 + image_size * i + j]) / 255.);
        }
        loaded_image_data.push_back(temp_img);
    }

    return loaded_image_data;
}

vector<datalabel_t> load_label_data(const string& file_path)
{
    datastream_t data = read_from_file(file_path);

    int magic_number_first_second = (data[0] << 8) | data[1];
    if (magic_number_first_second != 0) {
        cerr << "[ERROR] corrupt data.";
        throw;
    }

    unsigned char type_of_data = data[2];
    if (type_of_data != 0x08) {
        cerr << "[ERROR] Wrong data type. Expected unsigned byte.\n";
        throw;
    }

    int numb_dims = data[3];
    if (numb_dims != 1) {
        cerr << "[ERROR] Wrong data dimension. Expected 1 dimension for label data.\n";
        throw;
    }

    int numb_labels = (data[4] << 24) | (data[5] << 16) | (data[6] << 8) | data[7];

    vector<datalabel_t> loaded_label_data;

    if (data.size() < 8 + numb_labels) {
        cerr << "[ERROR] Data do not contain all stated labels.\n";
        throw;
    }

    for (int i = 0; i < numb_labels; ++i) {
        datalabel_t temp_labels(10, 0);
        unsigned char temp_label = data[8 + i];
        temp_labels[temp_label] = 1;
        loaded_label_data.push_back(temp_labels);
    }

    return loaded_label_data;
}

datastream_t read_from_file(const string& file_path)
{
    ifstream ifile(file_path, ios::binary);
    
    if (not ifile.is_open()) {
        cerr << "[ERROR] Could not open file for reading.\n";
        throw;
    }
    
    datastream_t data(istreambuf_iterator<char>(ifile), {});
    
    if (data.size() < 16) {
        cerr << "[ERROR] Found not enough data in file.\n";
        throw;
    }
    
    return data;
}
