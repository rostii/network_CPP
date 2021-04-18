#include "load_mnist_data.hpp"

#include <iostream>
#include <string>
#include <fstream>

using namespace std;

int main()
{
    string source_path = "../assets/";
    string file_path = "train-images.idx3-ubyte";
    
    load_mnist_data(source_path + file_path);
    
    return EXIT_SUCCESS;
}
