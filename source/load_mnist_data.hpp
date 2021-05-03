#ifndef load_mnist_data_hpp
#define load_mnist_data_hpp

#include <iostream>
#include <string>
#include <vector>
#include <tuple>

using datastream_t = std::vector<unsigned char>;
using dataimage_t = std::vector<double>;
using datalabel_t = std::vector<unsigned char>;
using datapoint_t = std::tuple<dataimage_t, datalabel_t>;
using dataset_t = std::vector<datapoint_t>;

dataset_t load_data(const std::string& src_path, const std::string& images_path, const std::string& labels_path);
std::vector<dataimage_t> load_image_data(const std::string& file_path);
std::vector<datalabel_t> load_label_data(const std::string& file_path);
datastream_t read_from_file(const std::string& file_path);

#endif /* load_mnist_data_hpp */
