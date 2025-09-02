#!/usr/bin/env python3
"""
Simple test to verify SOM functionality works
"""

import sys
import os

# Add self_organizing_maps to path
sys.path.insert(0, 'self_organizing_maps')

try:
    from self_organizing_map import SelfOrganizingMap, SOMNeuron
    print("✅ Import successful!")
    
    # Create a simple SOM
    som = SelfOrganizingMap(map_size=(3, 3), input_dim=2)
    print(f"✅ SOM created: {som.map_height}x{som.map_width}, input_dim={som.input_dim}")
    
    # Create simple test data
    import numpy as np
    data = np.random.randn(10, 2)
    print(f"✅ Test data created: {data.shape}")
    
    # Train the SOM
    som.train(data, n_iterations=50, verbose=False)
    print("✅ Training completed!")
    
    # Test BMU finding (map_input finds the best matching unit)
    test_input = np.random.randn(2)
    bmu = som.map_input(test_input)
    print(f"✅ BMU found: {bmu}")
    
    # Test prediction (scikit-learn style interface)
    predictions = som.predict(data)
    print(f"✅ Predictions: {predictions.shape}")
    
    print("\n🎉 All basic functionality working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()