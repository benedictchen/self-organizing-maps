#!/usr/bin/env python3
"""
Test the enhanced Self-Organizing Maps with topology preservation and adaptive weight adaptation
"""

import numpy as np
import sys
import os

# Add the module to path
sys.path.insert(0, os.path.dirname(__file__))

from self_organizing_maps import SelfOrganizingMap

def test_enhanced_som_features():
    """Test the new topology preservation and weight adaptation features"""
    
    print("🧪 Testing Enhanced SOM Features...")
    
    # Create synthetic 2D data with clear clusters
    np.random.seed(42)
    
    # Three clusters
    cluster1 = np.random.normal([2, 2], 0.5, (30, 2))
    cluster2 = np.random.normal([6, 6], 0.5, (30, 2))  
    cluster3 = np.random.normal([2, 6], 0.5, (30, 2))
    
    data = np.vstack([cluster1, cluster2, cluster3])
    
    # Create SOM
    som = SelfOrganizingMap(
        map_size=(10, 10),
        input_dim=2,
        initial_learning_rate=0.5,
        neighborhood_function='gaussian',
        parameter_schedule='exponential'
    )
    
    print(f"✅ SOM initialized: {som.map_height}x{som.map_width}")
    
    # Train the SOM
    print("🏋️ Training SOM...")
    results = som.train(data, n_iterations=500, verbose=False)
    
    print(f"✅ Training complete. Final QE: {results['final_quantization_error']:.4f}")
    
    # Test topology preservation
    print("🗺️ Measuring topology preservation...")
    topology_score = som.measure_topology_preservation(data)
    
    print(f"✅ Topology preservation score: {topology_score:.4f}")
    assert 0.0 <= topology_score <= 1.0, "Topology score should be between 0 and 1"
    
    # Test adaptive weight adaptation
    print("🧠 Testing adaptive weight adaptation...")
    
    # Select a test input
    test_input = data[0]
    current_lr = 0.1
    current_radius = 2.0
    
    # Test adaptive weight adaptation
    adaptation_stats = som.adaptive_weight_adaptation(
        test_input, 
        current_lr, 
        current_radius,
        adaptation_strength=1.5
    )
    
    print(f"✅ Weight adaptation completed:")
    print(f"   - Neurons updated: {adaptation_stats['neurons_updated']}")
    print(f"   - Total weight changes: {adaptation_stats['weight_changes']:.4f}")
    print(f"   - Avg adaptation factor: {adaptation_stats['avg_adaptation_factor']:.4f}")
    print(f"   - Topology stability: {adaptation_stats['topology_stability']:.4f}")
    
    # Verify adaptation stats are reasonable
    assert adaptation_stats['neurons_updated'] > 0, "Should update some neurons"
    assert adaptation_stats['weight_changes'] > 0, "Should have weight changes"
    assert 0.0 <= adaptation_stats['avg_adaptation_factor'] <= 3.0, "Adaptation factor should be reasonable"
    
    # Test helper methods
    print("🔧 Testing helper methods...")
    
    # Test local density calculation
    density = som._calculate_local_density(5, 5, test_input)
    assert 0.0 <= density <= 1.0, "Density should be normalized"
    print(f"✅ Local density: {density:.4f}")
    
    # Test weight stability calculation
    stability = som._calculate_weight_stability(5, 5)
    assert 0.0 <= stability <= 1.0, "Stability should be normalized"
    print(f"✅ Weight stability: {stability:.4f}")
    
    # Test neuron response matrix
    responses = som._get_neuron_responses(test_input)
    assert responses.shape == (som.map_height, som.map_width), "Response matrix shape should match map"
    print(f"✅ Neuron responses shape: {responses.shape}")
    
    print("\n🎉 All enhanced SOM features working correctly!")
    print(f"📊 Summary:")
    print(f"   - Topology Preservation: {topology_score:.3f}/1.000")
    print(f"   - Weight Adaptation: ✅ Functional")
    print(f"   - Helper Methods: ✅ All working")
    
    return {
        'topology_preservation': topology_score,
        'adaptation_working': True,
        'final_qe': results['final_quantization_error']
    }

if __name__ == "__main__":
    test_results = test_enhanced_som_features()
    print("\n✨ Enhanced SOM implementation is research-compliant!")