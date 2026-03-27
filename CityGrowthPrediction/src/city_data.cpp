#include "city_data.hpp"
#include <algorithm>
#include <numeric>
#include <iostream>

namespace city_ai {

CityMetrics::CityMetrics() {
    population = 0.0;
    area_km2 = 0.0;
    gdp_billion = 0.0;
    employment_rate = 0.0;
    infrastructure_score = 0.0;
    education_index = 0.0;
    healthcare_index = 0.0;
    transport_efficiency = 0.0;
    housing_price_index = 0.0;
    birth_rate = 0.0;
    death_rate = 0.0;
    migration_rate = 0.0;
}

void CityMetrics::normalize() {
    // These are typical world city ranges for normalization
    population = std::clamp(population / 30000000.0, 0.0, 1.0); // Max 30M
    area_km2 = std::clamp(area_km2 / 20000.0, 0.0, 1.0); // Max 20,000 km2
    gdp_billion = std::clamp(gdp_billion / 1000.0, 0.0, 1.0); // Max 1000B
    employment_rate = std::clamp(employment_rate, 0.0, 1.0);
    infrastructure_score = std::clamp(infrastructure_score, 0.0, 1.0);
    education_index = std::clamp(education_index, 0.0, 1.0);
    healthcare_index = std::clamp(healthcare_index, 0.0, 1.0);
    transport_efficiency = std::clamp(transport_efficiency, 0.0, 1.0);
    housing_price_index = std::clamp(housing_price_index / 500.0, 0.0, 1.0); // Max index 500
    birth_rate = std::clamp(birth_rate / 50.0, 0.0, 1.0); // Max 50 per 1000
    death_rate = std::clamp(death_rate / 30.0, 0.0, 1.0); // Max 30 per 1000
    migration_rate = std::clamp(migration_rate / 100.0, 0.0, 1.0); // Max 100 per 1000
}

std::vector<double> CityMetrics::to_vector() const {
    return {
        population,
        area_km2,
        gdp_billion,
        employment_rate,
        infrastructure_score,
        education_index,
        healthcare_index,
        transport_efficiency,
        housing_price_index,
        birth_rate,
        death_rate,
        migration_rate
    };
}

void GrowthPrediction::print() const {
    std::cout << "\n=== City Growth Prediction ===" << std::endl;
    std::cout << "Years ahead: " << years_ahead << std::endl;
    std::cout << "Predicted Population: " << predicted_population << std::endl;
    std::cout << "Predicted GDP (Billion): " << predicted_gdp << std::endl;
    std::cout << "Growth Rate: " << (growth_rate * 100.0) << "%" << std::endl;
    std::cout << "Confidence: " << (confidence * 100.0) << "%" << std::endl;
}

} // namespace city_ai
