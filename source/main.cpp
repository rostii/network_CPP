//
// MNIST labeled handwritten numbers loaded from: http://yann.lecun.com/exdb/mnist/
//
#include "load_mnist_data.hpp"
#include "network.hpp"

using namespace std;

int main()
{
    bool load = true;
    
    dataset_t train_data, test_data;
    if (load) {
        string source_path       = "/Users/rosti/Projects/XCode/DNN_CPP/assets/";
        string train_images_path = "train-images.idx3-ubyte";
        string train_labels_path = "train-labels.idx1-ubyte";
        string test_images_path  = "t10k-images.idx3-ubyte";
        string test_labels_path  = "t10k-labels.idx1-ubyte";

        train_data = load_data(source_path, train_images_path, train_labels_path);
        test_data = load_data(source_path, test_images_path, test_labels_path);
    }
    
    vector<int> nodes{784, 30, 10};
    Network ai(nodes);
    
    ai.train(train_data, 10, 30, 3., test_data);
    
    
    return EXIT_SUCCESS;
}
