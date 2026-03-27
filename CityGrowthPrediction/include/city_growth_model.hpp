#pragma once

#include "city_data.hpp"
#include "neural_network.hpp"
#include <vector>
#include <string>

namespace city_ai {

// City growth prediction model
class CityGrowthModel {
private:
    NeuralNetwork network;
    bool is_trained;
    
    // Historical data for training
    std::vector<CityMetrics> historical_data;
    std::vector<GrowthPrediction> growth_targets;
    
    // Train the model with historical data
    void train_model(int epochs = 1000);
    
    // Generate synthetic training data if no historical data available
    void generate_synthetic_data();
    
public:
    CityGrowthModel();
    
    // Train the model with historical city data
    void train(const std::vector<CityMetrics>& historical_metrics,
               const std::vector<GrowthPrediction>& growth_predictions,
               int epochs = 1000);
    
    // Predict future city growth
    GrowthPrediction predict(const CityMetrics& current_metrics, int years_ahead = 5);
    
    // Load model from file
    bool load(const std::string& filename);
    
    // Save model to file
    bool save(const std::string& filename) const;
    
    // Check if model is trained
    bool is_trained_model() const { return is_trained; }
    
    // Print model architecture
    void print_architecture() const;
};

} // namespace city_ai
