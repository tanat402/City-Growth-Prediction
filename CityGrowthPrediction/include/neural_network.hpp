#pragma once

#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

namespace city_ai {

// Activation functions
enum class ActivationType {
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    LeakyReLU
};

// Dense layer for neural network
class DenseLayer {
private:
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    ActivationType activation;
    std::vector<double> last_input;
    std::vector<double> last_output;
    
    double leaky_relu_slope = 0.01;
    
public:
    DenseLayer(int input_size, int output_size, ActivationType act = ActivationType::ReLU);
    
    // Forward pass
    std::vector<double> forward(const std::vector<double>& input);
    
    // Get output
    std::vector<double> get_output() const { return last_output; }
    
    // Apply activation function
    static double activate(double x, ActivationType type);
    static double activate_derivative(double x, ActivationType type);
    
    // Get layer parameters
    int get_input_size() const { return weights.empty() ? 0 : weights[0].size(); }
    int get_output_size() const { return weights.size(); }
};

// Neural Network model
class NeuralNetwork {
private:
    std::vector<DenseLayer> layers;
    double learning_rate;
    
    // Training data
    std::vector<std::vector<double>> training_inputs;
    std::vector<std::vector<double>> training_outputs;
    
    // Normalization parameters
    std::vector<double> input_min, input_max;
    std::vector<double> output_min, output_max;
    
public:
    NeuralNetwork(double lr = 0.001);
    
    // Add layers
    void add_layer(int size, ActivationType activation = ActivationType::ReLU);
    
    // Forward pass through entire network
    std::vector<double> predict(const std::vector<double>& input);
    
    // Train the network
    void train(const std::vector<std::vector<double>>& inputs,
               const std::vector<std::vector<double>>& outputs,
               int epochs,
               double learning_rate);
    
    // Normalize input data
    void normalize_input_data();
    
    // Denormalize output data
    std::vector<double> denormalize_output(const std::vector<double>& normalized);
    
    // Set normalization parameters
    void set_normalization_params(const std::vector<double>& in_min,
                                   const std::vector<double>& in_max,
                                   const std::vector<double>& out_min,
                                   const std::vector<double>& out_max);
};

} // namespace city_ai
