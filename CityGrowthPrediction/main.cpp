#include "city_growth_model.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <limits>

using namespace city_ai;

// Display menu
void print_menu() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "    AI City Growth Prediction Model    " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "1. Predict city growth" << std::endl;
    std::cout << "2. Show model architecture" << std::endl;
    std::cout << "3. Train with custom data" << std::endl;
    std::cout << "4. Run demo prediction" << std::endl;
    std::cout << "5. Exit" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Enter choice: ";
}

// Get city metrics from user
CityMetrics get_city_metrics_from_user() {
    CityMetrics metrics;
    
    std::cout << "\nEnter city metrics:" << std::endl;
    std::cout << "Population: ";
    std::cin >> metrics.population;
    
    std::cout << "Area (km2): ";
    std::cin >> metrics.area_km2;
    
    std::cout << "GDP (billion): ";
    std::cin >> metrics.gdp_billion;
    
    std::cout << "Employment rate (0-1): ";
    std::cin >> metrics.employment_rate;
    
    std::cout << "Infrastructure score (0-1): ";
    std::cin >> metrics.infrastructure_score;
    
    std::cout << "Education index (0-1): ";
    std::cin >> metrics.education_index;
    
    std::cout << "Healthcare index (0-1): ";
    std::cin >> metrics.healthcare_index;
    
    std::cout << "Transport efficiency (0-1): ";
    std::cin >> metrics.transport_efficiency;
    
    std::cout << "Housing price index: ";
    std::cin >> metrics.housing_price_index;
    
    std::cout << "Birth rate (per 1000): ";
    std::cin >> metrics.birth_rate;
    
    std::cout << "Death rate (per 1000): ";
    std::cin >> metrics.death_rate;
    
    std::cout << "Migration rate (per 1000): ";
    std::cin >> metrics.migration_rate;
    
    return metrics;
}

int main() {
    std::cout << "AI City Growth Prediction Model v1.0" << std::endl;
    
    CityGrowthModel model;
    
    int choice;
    while (true) {
        print_menu();
        std::cin >> choice;
        
        if (std::cin.fail()) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Invalid input. Please enter a number." << std::endl;
            continue;
        }
        
        switch (choice) {
            case 1: {
                // Predict city growth
                CityMetrics metrics = get_city_metrics_from_user();
                
                int years;
                std::cout << "Years to predict ahead: ";
                std::cin >> years;
                
                GrowthPrediction prediction = model.predict(metrics, years);
                prediction.print();
                break;
            }
            
            case 2:
                // Show model architecture
                model.print_architecture();
                break;
            
            case 3: {
                // Train with custom data
                std::cout << "\nCustom training feature - coming soon!" << std::endl;
                std::cout << "Using pre-trained model with synthetic data." << std::endl;
                break;
            }
            
            case 4: {
                // Run demo prediction with sample city data
                std::cout << "\n=== Running Demo Prediction ===" << std::endl;
                
                // Sample cities
                std::vector<std::string> city_names = {
                    "New York", "Tokyo", "London", "Mumbai", "Shanghai"
                };
                
                std::vector<CityMetrics> sample_cities = {
                    // New York
                    [](){
                        CityMetrics m;
                        m.population = 8336817;
                        m.area_km2 = 783.8;
                        m.gdp_billion = 800.0;
                        m.employment_rate = 0.92;
                        m.infrastructure_score = 0.85;
                        m.education_index = 0.88;
                        m.healthcare_index = 0.90;
                        m.transport_efficiency = 0.80;
                        m.housing_price_index = 250;
                        m.birth_rate = 12.5;
                        m.death_rate = 6.8;
                        m.migration_rate = 5.2;
                        return m;
                    }(),
                    
                    // Tokyo
                    [](){
                        CityMetrics m;
                        m.population = 13960256;
                        m.area_km2 = 2194.0;
                        m.gdp_billion = 950.0;
                        m.employment_rate = 0.94;
                        m.infrastructure_score = 0.92;
                        m.education_index = 0.91;
                        m.healthcare_index = 0.93;
                        m.transport_efficiency = 0.88;
                        m.housing_price_index = 180;
                        m.birth_rate = 7.5;
                        m.death_rate = 9.0;
                        m.migration_rate = 3.5;
                        return m;
                    }(),
                    
                    // London
                    [](){
                        CityMetrics m;
                        m.population = 8982000;
                        m.area_km2 = 1572.0;
                        m.gdp_billion = 600.0;
                        m.employment_rate = 0.90;
                        m.infrastructure_score = 0.82;
                        m.education_index = 0.89;
                        m.healthcare_index = 0.87;
                        m.transport_efficiency = 0.78;
                        m.housing_price_index = 320;
                        m.birth_rate = 13.0;
                        m.death_rate = 7.5;
                        m.migration_rate = 8.0;
                        return m;
                    }(),
                    
                    // Mumbai
                    [](){
                        CityMetrics m;
                        m.population = 12478447;
                        m.area_km2 = 603.0;
                        m.gdp_billion = 310.0;
                        m.employment_rate = 0.85;
                        m.infrastructure_score = 0.65;
                        m.education_index = 0.72;
                        m.healthcare_index = 0.68;
                        m.transport_efficiency = 0.55;
                        m.housing_price_index = 280;
                        m.birth_rate = 18.5;
                        m.death_rate = 5.8;
                        m.migration_rate = 15.0;
                        return m;
                    }(),
                    
                    // Shanghai
                    [](){
                        CityMetrics m;
                        m.population = 24280000;
                        m.area_km2 = 6340.0;
                        m.gdp_billion = 650.0;
                        m.employment_rate = 0.93;
                        m.infrastructure_score = 0.88;
                        m.education_index = 0.82;
                        m.healthcare_index = 0.85;
                        m.transport_efficiency = 0.75;
                        m.housing_price_index = 200;
                        m.birth_rate = 8.5;
                        m.death_rate = 5.5;
                        m.migration_rate = 12.0;
                        return m;
                    }()
                };
                
                int years;
                std::cout << "Enter prediction years (1-20): ";
                std::cin >> years;
                
                for (size_t i = 0; i < sample_cities.size(); ++i) {
                    std::cout << "\n----------------------------------------" << std::endl;
                    std::cout << "City: " << city_names[i] << std::endl;
                    GrowthPrediction prediction = model.predict(sample_cities[i], years);
                    prediction.print();
                }
                break;
            }
            
            case 5:
                std::cout << "Exiting..." << std::endl;
                return 0;
            
            default:
                std::cout << "Invalid choice. Please try again." << std::endl;
        }
    }
    
    return 0;
}
