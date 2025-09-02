"""
🧠 Self-Organizing Map (SOM) Implementation
==========================================

Author: Benedict Chen (benedict@benedictchen.com)

💝 Support This Work: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS

Developing high-quality, research-backed software takes countless hours of study, implementation, 
testing, and documentation. Your support - whether a little or a LOT - makes this work possible and is 
deeply appreciated. 

🎯 Help support continued research! Buy me a coffee ☕, beer 🍺, or lamborghini 🏎️

💖 Please consider recurring donations to fully support the work based on how much this module impacts your life or work!

Based on: Kohonen (1982) "Self-Organized Formation of Topologically Correct Feature Maps"

🎯 ELI5 Summary:
Think of a SOM like a smart map that learns to organize information by neighborhood.
If you show it pictures of animals, it will automatically group similar animals together
on a 2D grid, with cats near dogs, birds near each other, etc. No supervision needed!

🔬 Research Background:
========================
Teuvo Kohonen's 1982 paper introduced a revolutionary unsupervised learning algorithm
that models how the brain organizes sensory information. The key insight: neurons
compete for inputs (winner-takes-all) while cooperating through neighborhood functions.
This creates topologically organized feature maps that preserve input space structure.

The SOM algorithm revolutionized:
- Data visualization (high-dimensional → 2D maps)
- Clustering and classification
- Understanding cortical map formation
- Vector quantization and data compression

🏗️ Architecture:
================
Input Layer          SOM Grid (2D)           Output
-----------          --------------          ------
   🔵                    🟦🟦🟦              
   🔵        →          🟦🟨🟦          →    Clusters
   🔵                    🟦🟦🟦              

Algorithm Flow:
1. 🏆 Competition: Find Best Matching Unit (BMU) - neuron closest to input
2. 🤝 Cooperation: Define neighborhood around BMU using distance functions  
3. 📚 Adaptation: Update BMU and neighbors toward input (Hebbian learning)

Mathematical Framework:
- BMU: c = argmin_i ||x(t) - w_i(t)||
- Neighborhood: h_ci(t) = α(t) × exp(-||r_c - r_i||²/2σ²(t))
- Weight update: w_i(t+1) = w_i(t) + h_ci(t)[x(t) - w_i(t)]

🚀 Key Innovation: Unsupervised topological preservation
Revolutionary Impact: First algorithm to model biological cortical map formation

⚡ Configurable Options:
=======================
✨ Neighborhood Functions:
  - gaussian: exp(-d²/2σ²) [default - smooth, biological]
  - mexican_hat: center-surround activation pattern
  - rectangular: binary step function neighborhood  
  - linear_decay: linear decrease within radius

✨ Parameter Schedules:
  - exponential: η(t) = η₀ × exp(-t/τ) [default - fast early learning]
  - linear: η(t) = η₀ × (1 - t/T) [steady decay]
  - inverse_time: η(t) = η₀ / (1 + t/τ) [slow asymptotic decay]
  - power_law: η(t) = η₀ × (t₀/t)^α [scale-invariant decay]

🎨 ASCII Diagram - SOM Learning Process:
========================================
Initial (Random):        After Training:
    🔴🟢🔵                  🔴🔴🔴
    🟡⚫🟤         →         🟢🟢🟢  
    🟠⚪🟣                  🔵🔵🔵
    
Input: [0.9, 0.1, 0.1] → Finds red cluster
BMU at (0,0), updates neighborhood:
    🔴← BMU (winner)
    🟢← Neighbor (cooperates)  
    🔵← Distant (no update)

📚 Usage Example:
================
```python
from self_organizing_maps import SelfOrganizingMap

# Create SOM with custom configuration
som = SelfOrganizingMap(
    map_size=(15, 15),              # 15×15 neuron grid
    input_dim=3,                    # 3D input vectors
    neighborhood_function='gaussian', # Smooth neighborhoods
    parameter_schedule='exponential' # Fast early learning
)

# Train on your data
data = load_your_data()  # Shape: (n_samples, 3)
som.train(data, n_iterations=1000)

# Visualize results
som.visualize_map(data)

# Map new inputs
new_point = [0.5, 0.3, 0.7]
grid_position = som.map_input(new_point)
print(f"Input maps to grid position: {grid_position}")
```

🎯 Applications:
===============
- 📊 Data Visualization: High-dimensional data → 2D maps
- 🎯 Clustering: Unsupervised pattern discovery
- 🧠 Neuroscience: Understanding cortical organization
- 🎨 Image Processing: Color quantization, compression
- 📈 Finance: Market segmentation, risk analysis
- 🔊 Audio: Speech recognition, music analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SOMNeuron:
    """
    🧠 SOM Neuron: Individual processing unit in the Self-Organizing Map
    
    🎯 ELI5: Like a brain cell that remembers what patterns it likes best.
    Each neuron has a "favorite pattern" (weight_vector) and remembers its
    location on the map (position) and how active it's been (activation_history).
    
    🔬 Technical Details:
    Each neuron in the SOM grid maintains:
    - Spatial position (i, j) coordinates on the 2D lattice
    - Weight vector w_i representing its prototype/template
    - Activation history for analysis and visualization
    
    The neuron competes with others to respond to inputs by calculating
    the Euclidean distance: d = ||x - w_i|| where x is the input vector.
    The neuron with minimum distance becomes the Best Matching Unit (BMU).
    
    📊 Attributes:
    - position: (i, j) grid coordinates - where this neuron sits on the map
    - weight_vector: w_i ∈ ℝᵈ - this neuron's learned feature template  
