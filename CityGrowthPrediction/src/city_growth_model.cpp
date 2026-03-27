#include "city_growth_model.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>

namespace city_ai {

CityGrowthModel::CityGrowthModel() : network(0.001), is_trained(false) {
    // Build the neural network architecture
    // Input layer: 12 features (CityMetrics)
    // Hidden layers: 64 -> 32 -> 16
    // Output layer: 4 (population, GDP, growth_rate, confidence)
    network.add_layer(64, ActivationType::LeakyReLU);
    network.add_layer(32, ActivationType::LeakyReLU);
    network.add_layer(16, ActivationType::LeakyReLU);
    network.add_layer(4, ActivationType::Sigmoid); // Output layer
    
    // Generate synthetic training data
    generate_synthetic_data();
}

void CityGrowthModel::generate_synthetic_data() {
    // Generate realistic city growth patterns for training
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    for (int i = 0; i < 1000; ++i) {
        CityMetrics metrics;
        
        // Generate realistic values
        metrics.population = dist(gen) * 20000000 + 100000; // 100K to 20M
        metrics.area_km2 = dist(gen) * 5000 + 100; // 100 to 5100 km2
        metrics.gdp_billion = dist(gen) * 500 + 1; // 1B to 500B
        metrics.employment_rate = dist(gen) * 0.4 + 0.5; // 0.5 to 0.9
        metrics.infrastructure_score = dist(gen) * 0.6 + 0.2; // 0.2 to 0.8
        metrics.education_index = dist(gen) * 0.5 + 0.4; // 0.4 to 0.9
        metrics.healthcare_index = dist(gen) * 0.5 + 0.4; // 0.4 to 0.9
        metrics.transport_efficiency = dist(gen) * 0.6 + 0.2; // 0.2 to 0.8
        metrics.housing_price_index = dist(gen) * 200 + 50; // 50 to 250
        metrics.birth_rate = dist(gen) * 15 + 5; // 5 to 20
        metrics.death_rate = dist(gen) * 8 + 4; // 4 to 12
        metrics.migration_rate = dist(gen) * 30 - 10; // -10 to 20
        
        // Calculate growth prediction
        GrowthPrediction prediction;
        
        // Natural growth rate
        double natural_growth = (metrics.birth_rate - metrics.death_rate) / 1000.0;
        
        // Economic growth factor
        double economic_factor = metrics.employment_rate * metrics.gdp_billion / 100.0;
        
        // Infrastructure factor
        double infra_factor = metrics.infrastructure_score * metrics.transport_efficiency;
        
        // Combined growth rate (simplified model)
        prediction.growth_rate = (natural_growth + 0.02 * economic_factor + 0.03 * infra_factor + 
                                  metrics.migration_rate / 1000.0) * (1.0 + dist(gen) * 0.1);
        
        prediction.growth_rate = std::clamp(prediction.growth_rate, -0.1, 0.2);
        
        // Predicted population after 5 years
        prediction.predicted_population = metrics.population * std::exp(prediction.growth_rate * 5);
        
        // Predicted GDP growth
        prediction.predicted_gdp = metrics.gdp_billion * std::exp(prediction.growth_rate * 3);
        
        // Confidence based on data consistency
        prediction.confidence = 0.7 + dist(gen) * 0.25;
        prediction.years_ahead = 5;
        
        historical_data.push_back(metrics);
        growth_targets.push_back(prediction);
    }
}

void CityGrowthModel::train_model(int epochs) {
    // Prepare training data
    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> outputs;
    
    // Normalize metrics and create training pairs
    std::vector<double> input_min(12, 0.0);
    std::vector<double> input_max(12, 1.0);
    std::vector<double> output_min = {0.0, 0.0, -0.1, 0.5};
    std::vector<double> output_max = {25000000.0, 1000.0, 0.3, 1.0};
    
    for (size_t i = 0; i < historical_data.size(); ++i) {
        CityMetrics m = historical_data[i];
        m.normalize();
        inputs.push_back(m.to_vector());
        
        // Normalize outputs
        GrowthPrediction p = growth_targets[i];
        outputs.push_back({
            (p.predicted_population - output_min[0]) / (output_max[0] - output_min[0]),
            (p.predicted_gdp - output_min[1]) / (output_max[1] - output_min[1]),
            (p.growth_rate - output_min[2]) / (output_max[2] - output_min[2]),
            p.confidence
        });
    }
    
    network.set_normalization_params(input_min, input_max, output_min, output_max);
    
    std::cout << "\n=== Training City Growth Model ===" << std::endl;
    std::cout << "Training samples: " << inputs.size() << std::endl;
    std::cout << "Epochs: " << epochs << std::endl;
    
    network.train(inputs, outputs, epochs, 0.001);
    
    is_trained = true;
    std::cout << "Training complete!" << std::endl;
}

void CityGrowthModel::train(const std::vector<CityMetrics>& historical_metrics,
                            const std::vector<GrowthPrediction>& growth_predictions,
                            int epochs) {
    
    if (historical_metrics.size() != growth_predictions.size()) {
        std::cerr << "Error: Metrics and predictions size mismatch!" << std::endl;
        return;
    }
    
    historical_data = historical_metrics;
    growth_targets = growth_predictions;
    train_model(epochs);
}

GrowthPrediction CityGrowthModel::predict(const CityMetrics& current_metrics, int years_ahead) {
    GrowthPrediction result;
    result.years_ahead = years_ahead;
    
    if (!is_trained) {
        std::cout << "Model not trained. Training with synthetic data..." << std::endl;
        train_model(1000);
    }
    
    // Normalize input
    CityMetrics normalized = current_metrics;
    normalized.normalize();
    
    // Get prediction
    std::vector<double> input = normalized.to_vector();
    std::vector<double> output = network.predict(input);
    
    // Denormalize output
    std::vector<double> output_min = {0.0, 0.0, -0.1, 0.5};
    std::vector<double> output_max = {25000000.0, 1000.0, 0.3, 1.0};
    
    result.predicted_population = output[0] * (output_max[0] - output_min[0]) + output_min[0];
    result.predicted_gdp = output[1] * (output_max[1] - output_min[1]) + output_min[1];
    result.growth_rate = output[2] * (output_max[2] - output_min[2]) + output_min[2];
    result.confidence = output[3];
    
    // Adjust prediction for years_ahead
    double years_factor = years_ahead / 5.0;
    result.predicted_population = current_metrics.population * 
                                   std::exp(result.growth_rate * years_ahead);
    result.predicted_gdp = current_metrics.gdp_billion * 
                          std::exp(result.growth_rate * years_factor * 0.6);
    
    return result;
}

bool CityGrowthModel::load(const std::string& filename) {
    // Placeholder for model loading
    std::ifstream file(filename);
    if (file.is_open()) {
        file >> is_trained;
        file.close();
        return true;
    }
    return false;
}

bool CityGrowthModel::save(const std::string& filename) const {
    // Placeholder for model saving
    std::ofstream file(filename);
    if (file.is_open()) {
        file << is_trained << std::endl;
        file.close();
        return true;
    }
    return false;
}

void CityGrowthModel::print_architecture() const {
    std::cout << "\n=== City Growth Prediction Model Architecture ===" << std::endl;
    std::cout << "Input Layer: 12 features (City Metrics)" << std::endl;
    std::cout << "Hidden Layer 1: 64 neurons (LeakyReLU)" << std::endl;
    std::cout << "Hidden Layer 2: 32 neurons (LeakyReLU)" << std::endl;
    std::cout << "Hidden Layer 3: 16 neurons (LeakyReLU)" << std::endl;
    std::cout << "Output Layer: 4 neurons (Population, GDP, Growth Rate, Confidence)" << std::endl;
    std::cout << "Training Status: " << (is_trained ? "Trained" : "Not Trained") << std::endl;
}

} // namespace city_ai
