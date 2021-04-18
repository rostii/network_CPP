//
// MNIST labeled handwritten numbers loaded from: http://yann.lecun.com/exdb/mnist/
//

#include "load_mnist_data.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main()
{
    bool control_loaded_data = true;
    string source_path = "../assets/";
    string train_images_file_path = "train-images.idx3-ubyte";
    string train_labels_file_path = "train-labels.idx1-ubyte";
    string test_images_file_path  = "t10k-images.idx3-ubyte";
    string test_labels_file_path  = "t10k-labels.idx1-ubyte";

    
    vector<vector<u_char>> train_images = load_mnist_data(source_path + train_images_file_path);
    vector<vector<u_char>> train_labels = load_mnist_data(source_path + train_labels_file_path);
    vector<vector<u_char>> test_images  = load_mnist_data(source_path + test_images_file_path);
    vector<vector<u_char>> test_labels  = load_mnist_data(source_path + test_labels_file_path);
    
    if (control_loaded_data) {
        // Control laoded images
        int train_idx = 1061;
        int test_idx = 88;
        for (int i = 0; i < 10; ++i) {
            if (train_labels[train_idx][i] == 1) {
                cout << "Train image label is: " << i << "\n";
            }
            if (test_labels[test_idx][i] == 1) {
                cout << "Test image label is: " << i << "\n";
            }
        }
        
        Mat train_img = Mat(28, 28, CV_8UC1, &train_images[train_idx][0]);
        Mat test_img  = Mat(28, 28, CV_8UC1,   &test_images[test_idx][0]);
        namedWindow("Train image", 0);
        namedWindow("Test image",  0);
        imshow("Train image", train_img);
        imshow("Test image",   test_img);
        waitKey();
        destroyAllWindows();
    }
    
    return EXIT_SUCCESS;
}
