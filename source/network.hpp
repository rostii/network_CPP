#ifndef network_hpp
#define network_hpp

#include "load_mnist_data.hpp"

#include <vector>

using bias_t = double;
using weight_t = double;
using weights_t = std::vector<weight_t>;
using node_t = std::tuple<weights_t, bias_t>;
using layer_t = std::vector<node_t>;
using network_t = std::vector<layer_t>;

class Network
{
public:
    Network(const std::vector<int>& nodes);

    network_t network_nodes;
    
    void train(dataset_t train_data, size_t batch_size, int epochs, double learn_rate, const dataset_t& test_data);
    
private:
    void forwardpropagate(std::vector<std::vector<double>>& weighted_inputs,
                          std::vector<std::vector<double>>& activated_outputs,
                          const dataimage_t& image);
    void backpropagate(network_t& nabla,
                       std::vector<std::vector<double>>& weighted_inputs,
                       std::vector<std::vector<double>>& activated_outputs,
                       const datalabel_t& label);
    void update_parameter(const network_t& nabla, double learn_rate, size_t batch_size);
    double activate(double value);
    double activate_derivative(double value);
    double cost_derivativ(unsigned char label, double activation);
    int evaluate(const dataset_t& data);
};

#endif /* network_hpp */
