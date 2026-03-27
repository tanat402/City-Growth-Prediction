#pragma once

#include <vector>
#include <string>

namespace city_ai {

// City metrics that will be used for prediction
struct CityMetrics {
    double population;           // Current population
    double area_km2;             // City area in square kilometers
    double gdp_billion;          // GDP in billions
    double employment_rate;       // Employment rate (0-1)
    double infrastructure_score; // Infrastructure development score (0-1)
    double education_index;      // Education index (0-1)
    double healthcare_index;      // Healthcare index (0-1)
    double transport_efficiency;  // Transportation efficiency (0-1)
    double housing_price_index;   // Housing price index
    double birth_rate;            // Birth rate per 1000
    double death_rate;            // Death rate per 1000
    double migration_rate;        // Net migration rate per 1000
    
    // Constructor with default values
    CityMetrics();
    
    // Normalize all metrics to [0, 1] range
    void normalize();
    
    // Convert to vector for neural network input
    std::vector<double> to_vector() const;
};

// Growth prediction result
struct GrowthPrediction {
    double predicted_population;
    double predicted_gdp;
    double growth_rate;
    double confidence;
    int years_ahead;
    
    void print() const;
};

} // namespace city_ai
