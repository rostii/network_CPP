#include "network.hpp"

#include <random>
#include <algorithm>
#include <chrono>

using namespace std;
using namespace std::chrono;

Network::Network(const vector<int>& nodes)
{
    random_device rd;
    default_random_engine generator(rd());
    normal_distribution<double> distribution;
    
    for (size_t i = 1; i < nodes.size(); ++i) {
        layer_t layer;
        for (int j = 0; j < nodes[i]; ++j) {
            bias_t bias = distribution(generator);
            
            weights_t weights;
            for (int k = 0; k < nodes[i - 1]; ++k) {
                double weight = distribution(generator);
                weights.push_back(weight);
            }
            
            node_t node = make_tuple(weights, bias);
            layer.push_back(node);
        }
        
        network_nodes.push_back(layer);
    }
}

void Network::train(dataset_t train_data, size_t batch_size, int epochs, double learn_rate, const dataset_t& test_data)
{
    if (!train_data.size()) {
        cout << "[ERROR] No images in dataset.\n";
        throw;
    }

    if (train_data.size() < batch_size) {
        cout << "[INFO] Size of batch bigger than images. Batch size is adjusted to number of images.\n";
        batch_size = train_data.size();
    }
    
    steady_clock::time_point begin_train = steady_clock::now();

    random_device rd;
    default_random_engine generator(rd());
    for (int i = 0; i < epochs; ++i) {
        cout << "Epoch " << i + 1 << ": ";
        steady_clock::time_point begin_epoch = steady_clock::now();
        
        shuffle(train_data.begin(), train_data.end(), generator);
        
        for (dataset_t::iterator data_it = train_data.begin(); data_it < train_data.end(); data_it += batch_size) {
            dataset_t batch(data_it, min(data_it + batch_size, train_data.end()));
            
            for (datapoint_t data_pt: batch) {
                vector<vector<double>> weighted_inputs;
                vector<vector<double>> activated_outputs;
                forwardpropagate(weighted_inputs, activated_outputs, get<0>(data_pt));
                
                network_t nabla;
                backpropagate(nabla, weighted_inputs, activated_outputs, get<1>(data_pt));
                
                update_parameter(nabla, learn_rate, batch.size());
            }
        }
        
        int correct_classified = evaluate(test_data);
        
        steady_clock::time_point end_epoch = steady_clock::now();
        cout << correct_classified << "/" << test_data.size()
             << " (" << duration_cast<seconds>(end_epoch - begin_epoch).count() << " seconds)\n";
    }
    
    steady_clock::time_point end_train = steady_clock::now();
    cout << "Total training time: " << duration_cast<minutes>(end_train - begin_train).count() << " minutes\n";
}

void Network::forwardpropagate(vector<vector<double>>& weighted_inputs,
                               vector<vector<double>>& activated_outputs,
                               const dataimage_t& image)
{
    vector<double> activations(image.begin(), image.end());
    activated_outputs.push_back(activations);
    
    for (layer_t layer : network_nodes) {
        vector<double> weighted_inputs_layer;
        vector<double> activated_outputs_layer;
        
        for (node_t node: layer) {
            weights_t& weights = get<0>(node);
            bias_t& bias = get<1>(node);
            
            double weighted_input = 0;
            vector<double>::iterator node_activations_it = activations.begin();
            for (weight_t weight: weights) {
                weighted_input += *node_activations_it * weight;
                ++node_activations_it;
            }
            weighted_input += bias;
            
            weighted_inputs_layer.push_back(weighted_input);
            activated_outputs_layer.push_back(activate(weighted_input));
        }
        
        weighted_inputs.push_back(weighted_inputs_layer);
        activated_outputs.push_back(activated_outputs_layer);
        
        activations = activated_outputs_layer;
    }
}

void Network::backpropagate(network_t& nabla,
                            vector<vector<double>>& weighted_inputs,
                            vector<vector<double>>& activated_outputs,
                            const datalabel_t& label)
{
    network_t::reverse_iterator network_r_it = network_nodes.rbegin();
    vector<vector<double>>::reverse_iterator activations_r_it = activated_outputs.rbegin();
    vector<vector<double>>::reverse_iterator w_inputs_r_it = weighted_inputs.rbegin();
    
    layer_t& last_layer_nodes = *network_r_it;
    vector<double>& second_last_layer_activations = *(activations_r_it + 1);
    datalabel_t::const_iterator label_it = label.begin();
    vector<double>::iterator last_layer_activations_it = (*activations_r_it).begin();
    vector<double>::iterator last_layer_w_inputs_it = (*w_inputs_r_it).begin();
    
    vector<double> deltas;
    layer_t last_layer_nabla;
    for (node_t node: last_layer_nodes) {
        bias_t nabla_bias;
        weights_t nabla_weights;
        
        double delta = cost_derivativ(*label_it, *last_layer_activations_it) * activate_derivative(*last_layer_w_inputs_it);
        nabla_bias = delta;
        for (double sll_activation: second_last_layer_activations)
            nabla_weights.push_back(delta * sll_activation);
        
        deltas.push_back(delta);
        last_layer_nabla.push_back(make_tuple(nabla_weights, nabla_bias));
        
        ++label_it; ++last_layer_activations_it; ++last_layer_w_inputs_it;
    }
    
    nabla.push_back(last_layer_nabla);
    
    ++network_r_it; ++activations_r_it; ++w_inputs_r_it;
    
    for (; network_r_it != network_nodes.rend(); ++network_r_it, ++activations_r_it, ++w_inputs_r_it) {
        layer_t& layer_nodes = *network_r_it;
        layer_t& next_layer_nodes = *(network_r_it - 1);
        vector<double>& prev_layer_activations = *(activations_r_it + 1);
        vector<double>::iterator layer_w_inputs_it = (*w_inputs_r_it).begin();
        
        int node_idx = 0;
        vector<double> layer_deltas;
        layer_t layer_nabla;
        for (node_t node: layer_nodes) {
            bias_t nabla_bias;
            weights_t nabla_weights;
            
            vector<double>::iterator deltas_it = deltas.begin();
            double delta = 0;
            for (node_t next_layer_node: next_layer_nodes) {
                weights_t& next_layer_node_weights = get<0>(next_layer_node);
                weight_t weight = *(next_layer_node_weights.data() + node_idx);
                delta += weight * *deltas_it;
                ++deltas_it;
            }
            delta *= activate_derivative(*layer_w_inputs_it);
            
            nabla_bias = delta;
            for (double pl_activation: prev_layer_activations)
                nabla_weights.push_back(delta * pl_activation);
            
            layer_deltas.push_back(delta);
            layer_nabla.push_back(make_tuple(nabla_weights, nabla_bias));
            
            ++layer_w_inputs_it; ++node_idx;
        }
        
        deltas = layer_deltas;
        nabla.insert(nabla.begin(), layer_nabla);
    }
}

void Network::update_parameter(const network_t &nabla, double learn_rate, size_t batch_size)
{
    network_t::iterator network_it = network_nodes.begin();
    
    for (layer_t layer_nabla : nabla) {
        layer_t::iterator layer_it = (*network_it).begin();
        
        for (node_t node_nabla : layer_nabla) {
            node_t& node = (*layer_it);
            weights_t& weights = get<0>(node);
            weights_t nabla_weights = get<0>(node_nabla);
            bias_t& bias = get<1>(node);
            bias_t nabla_bias = get<1>(node_nabla);
            
            bias -= learn_rate * nabla_bias / batch_size;
            
            weights_t::iterator weights_it = weights.begin();
            for (weight_t nabla_weight: nabla_weights) {
                weight_t& weight = *weights_it;
                weight -= learn_rate * nabla_weight / batch_size;
                
                ++weights_it;
            }
            
            ++layer_it;
        }
        
        ++network_it;
    }
}

double Network::activate(double value)
{
    return 1. / (1. + exp(-value));
}

double Network::activate_derivative(double value)
{
    return activate(value) * (1 - activate(value));
}

double Network::cost_derivativ(u_char label, double activation)
{
    return activation - label;
}

int Network::evaluate(const dataset_t& data)
{
    int sum_correct_classified = 0;
    
    for (datapoint_t data_pt: data) {
        dataimage_t& image = get<0>(data_pt);
        datalabel_t& label = get<1>(data_pt);
        
        vector<double> activations = image;
        for (layer_t layer: network_nodes) {
            vector<double> layer_activations;
            
            for (node_t node: layer) {
                weights_t& weights = get<0>(node);
                bias_t& bias = get<1>(node);
                
                double weighted_input = 0;
                vector<double>::iterator activations_it = activations.begin();
                for (weight_t weight: weights) {
                    weighted_input += *activations_it * weight;
                    ++activations_it;
                }
                weighted_input += bias;
                
                layer_activations.push_back(activate(weighted_input));
            }
            
            activations = layer_activations;
        }
        
        size_t max_value_idx = max_element(activations.begin(), activations.end()) - activations.begin();
        if (label[max_value_idx] == 1) ++sum_correct_classified;
    }
    
    return sum_correct_classified;
}
