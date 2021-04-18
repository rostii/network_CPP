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
    string test_images_file_path = "t10k-images.idx3-ubyte";
    string test_labels_file_path = "t10k-labels.idx1-ubyte";

    
    vector<vector<u_char>> train_images = load_mnist_data(source_path + train_images_file_path);
    vector<vector<u_char>> train_labels = load_mnist_data(source_path + train_labels_file_path);

    
    if (control_loaded_data) {
        // Control laoded images
        int idx = 1061;
        for (int i = 0; i < 10; ++i) {
            if (train_labels[idx][i] == 1) {
                cout << "Train image label is: " << i << "\n";
            }
        }
        
        Mat img = Mat(28, 28, CV_8UC1, &train_images[idx][0]);
        namedWindow("test", 0);
        imshow("test", img);
        waitKey();
        destroyAllWindows();
    }
    
    
    return EXIT_SUCCESS;
}
