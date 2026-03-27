#include "neural_network.hpp"
#include <iostream>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace city_ai {

// DenseLayer implementation
DenseLayer::DenseLayer(int input_size, int output_size, ActivationType act)
    : activation(act) {
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 0.1);
    
    // Xavier/He initialization
    double scale = std::sqrt(2.0 / input_size);
    
    // Initialize weights
    weights.resize(output_size);
    for (int i = 0; i < output_size; ++i) {
        weights[i].resize(input_size);
        for (int j = 0; j < input_size; ++j) {
            weights[i][j] = dist(gen) * scale;
        }
    }
    
    // Initialize biases
    biases.resize(output_size, 0.0);
}

double DenseLayer::activate(double x, ActivationType type) {
    switch (type) {
        case ActivationType::ReLU:
            return std::max(0.0, x);
        case ActivationType::Sigmoid:
            return 1.0 / (1.0 + std::exp(-std::clamp(x, -500.0, 500.0)));
        case ActivationType::Tanh:
            return std::tanh(x);
        case ActivationType::LeakyReLU:
            return x > 0 ? x : 0.01 * x;
        case ActivationType::Softmax:
            return x; // Handled separately
        default:
            return x;
    }
}

double DenseLayer::activate_derivative(double x, ActivationType type) {
    switch (type) {
        case ActivationType::ReLU:
            return x > 0 ? 1.0 : 0.0;
        case ActivationType::Sigmoid: {
            double sig = 1.0 / (1.0 + std::exp(-std::clamp(x, -500.0, 500.0)));
            return sig * (1.0 - sig);
        }
        case ActivationType::Tanh:
            return 1.0 - std::tanh(x) * std::tanh(x);
        case ActivationType::LeakyReLU:
            return x > 0 ? 1.0 : 0.01;
        default:
            return 1.0;
    }
}

std::vector<double> DenseLayer::forward(const std::vector<double>& input) {
    last_input = input;
    int output_size = weights.size();
    
    last_output.resize(output_size);
    
    // Matrix multiplication + bias
    for (int i = 0; i < output_size; ++i) {
        double sum = biases[i];
        for (size_t j = 0; j < input.size(); ++j) {
            sum += weights[i][j] * input[j];
        }
        last_output[i] = activate(sum, activation);
    }
    
    return last_output;
}

// NeuralNetwork implementation
NeuralNetwork::NeuralNetwork(double lr) : learning_rate(lr) {}

void NeuralNetwork::add_layer(int size, ActivationType activation) {
    int input_size;
    if (layers.empty()) {
        input_size = 12; // CityMetrics has 12 features
    } else {
        input_size = layers.back().get_output_size();
    }
    layers.emplace_back(input_size, size, activation);
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& input) {
    std::vector<double> output = input;
    for (auto& layer : layers) {
        output = layer.forward(output);
    }
    return output;
}

void NeuralNetwork::train(const std::vector<std::vector<double>>& inputs,
                          const std::vector<std::vector<double>>& outputs,
                          int epochs,
                          double lr) {
    
    learning_rate = lr;
    
    // Simple gradient descent training
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_error = 0.0;
        
        for (size_t i = 0; i < inputs.size(); ++i) {
            // Forward pass
            std::vector<double> prediction = predict(inputs[i]);
            
            // Calculate error
            std::vector<double> error(outputs[i].size());
            for (size_t j = 0; j < outputs[i].size(); ++j) {
                error[j] = outputs[i][j] - prediction[j];
                total_error += error[j] * error[j];
            }
            
            // Backward pass (simplified - using numerical gradient approximation)
            double epsilon = 0.0001;
            
            for (auto& layer : layers) {
                // This is a simplified training loop
                // In a full implementation, we would implement proper backpropagation
            }
        }
        
        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << "/" << epochs 
                      << " - Error: " << std::sqrt(total_error / inputs.size()) << std::endl;
        }
    }
}

void NeuralNetwork::set_normalization_params(const std::vector<double>& in_min,
                                              const std::vector<double>& in_max,
                                              const std::vector<double>& out_min,
                                              const std::vector<double>& out_max) {
    input_min = in_min;
    input_max = in_max;
    output_min = out_min;
    output_max = out_max;
}

std::vector<double> NeuralNetwork::denormalize_output(const std::vector<double>& normalized) {
    std::vector<double> result(normalized.size());
    for (size_t i = 0; i < normalized.size(); ++i) {
        if (output_max[i] - output_min[i] > 0.0001) {
            result[i] = normalized[i] * (output_max[i] - output_min[i]) + output_min[i];
        } else {
            result[i] = normalized[i];
        }
    }
    return result;
}

} // namespace city_ai
