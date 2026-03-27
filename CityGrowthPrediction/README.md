# AI City Growth Prediction Model

An artificial intelligence model for predicting city population and economic growth based on various urban metrics.

## Project Overview

This project implements a neural network-based model that predicts future city growth metrics including:
- Population growth
- GDP growth
- Overall growth rate
- Prediction confidence

## Features

- **Neural Network Architecture**: Custom-built dense layers with multiple activation functions
- **City Metrics Analysis**: 12 key urban indicators for prediction
- **Training System**: Automated synthetic data generation and model training
- **Prediction Engine**: Fast and accurate growth predictions
- **Interactive CLI**: User-friendly command-line interface

## Architecture

```
Input Layer: 12 features
├── Population
├── Area (km²)
├── GDP (billion)
├── Employment Rate
├── Infrastructure Score
├── Education Index
├── Healthcare Index
├── Transport Efficiency
├── Housing Price Index
├── Birth Rate
├── Death Rate
└── Migration Rate

Hidden Layers:
├── Dense Layer 1: 64 neurons (LeakyReLU)
├── Dense Layer 2: 32 neurons (LeakyReLU)
└── Dense Layer 3: 16 neurons (LeakyReLU)

Output Layer: 4 predictions
├── Predicted Population
├── Predicted GDP
├── Growth Rate
└── Confidence Score
```

## Building the Project

### Prerequisites
- C++ compiler with C++17 support (GCC, Clang, MSVC)
- CMake 3.10 or higher

### Build Instructions

```bash
# Navigate to project directory
cd CityGrowthPrediction

# Create build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build the project
cmake --build . --config Release
```

### Running the Application

```bash
# Run from build directory
./bin/city_growth_prediction

# Or from project root
./build/bin/city_growth_prediction
```

## Usage

### Interactive Mode

1. **Predict City Growth**: Enter custom city metrics and predict future growth
2. **View Architecture**: Display the neural network structure
3. **Run Demo**: Test with sample data from major cities (New York, Tokyo, London, Mumbai, Shanghai)
4. **Exit**: Close the application

### Programmatic Usage

```cpp
#include "city_growth_model.hpp"

using namespace city_ai;

// Create model
CityGrowthModel model;

// Set up city metrics
CityMetrics metrics;
metrics.population = 1000000;
metrics.area_km2 = 500;
metrics.gdp_billion = 50;
// ... set other metrics

// Predict growth for 5 years
GrowthPrediction prediction = model.predict(metrics, 5);
prediction.print();
```

## City Metrics Explained

| Metric | Description | Range |
|--------|-------------|-------|
| Population | Total city residents | 0 - 30M |
| Area | City geographic area | 0 - 20,000 km² |
| GDP | Gross Domestic Product | 0 - 1,000B |
| Employment Rate | Working population ratio | 0.0 - 1.0 |
| Infrastructure Score | Development level | 0.0 - 1.0 |
| Education Index | Education quality | 0.0 - 1.0 |
| Healthcare Index | Healthcare quality | 0.0 - 1.0 |
| Transport Efficiency | Transportation quality | 0.0 - 1.0 |
| Housing Price Index | Real estate costs | 0 - 500 |
| Birth Rate | Births per 1000 | 0 - 50 |
| Death Rate | Deaths per 1000 | 0 - 30 |
| Migration Rate | Net migration per 1000 | -100 - 100 |

## Model Training

The model uses synthetic training data based on real-world urban growth patterns:

- **Training Samples**: 1,000 synthetic city profiles
- **Training Algorithm**: Gradient descent with backpropagation
- **Normalization**: Min-max scaling for all inputs/outputs

## Future Enhancements

- [ ] Time-series LSTM implementation
- [ ] Model persistence (save/load weights)
- [ ] Data import from CSV/JSON files
- [ ] Visualization dashboard
- [ ] Multi-year sequential prediction
- [ ] Regional comparison analysis
- [ ] API server for web integration

## Technical Details

- **Language**: C++17
- **Build System**: CMake
- **Dependencies**: Standard Library only (no external ML libraries)
- **License**: MIT License

## Contributing

Contributions are welcome! Please read the contribution guidelines before submitting pull requests.

## Authors

AI City Growth Prediction Model - Version 1.0

## License

MIT License - See LICENSE file for details
