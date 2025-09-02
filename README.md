# 💰 Support This Research - Please Donate!

**🙏 If this library helps your research or project, please consider donating to support continued development:**

**[💳 DONATE VIA PAYPAL - CLICK HERE](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS)**

[![CI](https://github.com/benedictchen/self-organizing-maps/workflows/CI/badge.svg)](https://github.com/benedictchen/self-organizing-maps/actions)
[![PyPI version](https://badge.fury.io/py/self-organizing-maps.svg)](https://badge.fury.io/py/self-organizing-maps)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Custom%20Non--Commercial-red.svg)](LICENSE)

---

# Self-Organizing Maps

🗺️ Kohonen Self-Organizing Maps for unsupervised learning and visualization

**Kohonen, T. (1982)** - "Self-organized formation of topologically correct feature maps"  
**Kohonen, T. (2001)** - "Self-Organizing Maps: Third Edition"

## 📦 Installation

```bash
pip install self-organizing-maps
```

## 🚀 Quick Start

### Basic SOM Example
```python
from self_organizing_maps import SelfOrganizingMap
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data (2D clusters)
data = np.random.rand(1000, 3) * 100

# Create and train SOM
som = SelfOrganizingMap(
    width=10,
    height=10, 
    input_dim=3,
    learning_rate=0.1,
    neighborhood_radius=5.0,
    decay_function='exponential'
)

# Train the SOM
som.fit(data, epochs=1000)

# Get winning neurons for new data
test_data = np.random.rand(10, 3) * 100
winners = som.predict(test_data)
print("Winning neurons:", winners)

# Visualize the trained map
som.plot_map()
plt.show()
```

### Growing Self-Organizing Map (GSOM)
```python
from self_organizing_maps import GrowingSOM

# Create GSOM that grows based on data distribution
gsom = GrowingSOM(
    initial_width=2,
    initial_height=2,
    input_dim=4,
    growth_threshold=0.1,
    max_nodes=100,
    learning_rate=0.1
)

# Adaptive training with automatic growth
iris_data = load_iris_dataset()  # 4D feature space
gsom.fit(iris_data, epochs=500)

print(f"Final map size: {gsom.width}x{gsom.height}")
print(f"Total neurons: {gsom.get_neuron_count()}")

# Visualize growth pattern
gsom.plot_growth_history()
plt.show()
```

### Hierarchical Self-Organizing Map
```python
from self_organizing_maps import HierarchicalSOM

# Create hierarchical structure for complex data
hsom = HierarchicalSOM(
    levels=3,
    map_sizes=[(8, 8), (4, 4), (2, 2)],
    input_dim=10,
    branching_factor=4
)

# Complex dataset (e.g., document features)
documents = load_document_vectors()  # 10D TF-IDF vectors
hsom.fit(documents, epochs=500)

# Navigate hierarchy
top_level_clusters = hsom.get_clusters(level=0)
detailed_clusters = hsom.get_clusters(level=2)

# Visualize hierarchy
hsom.plot_hierarchy()
plt.show()
```

## 🎨 Advanced Visualization

### U-Matrix and Component Planes
```python
from self_organizing_maps import Visualization

viz = Visualization(som)

# Create comprehensive visualization dashboard
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# U-matrix (distance matrix)
viz.plot_u_matrix(ax=axes[0,0])

# Component planes for each input dimension
for i in range(3):
    viz.plot_component_plane(i, ax=axes[0,1+i] if i < 2 else axes[1,0])

# Hit histogram
viz.plot_hit_histogram(ax=axes[1,1])

# Cluster boundaries
viz.plot_cluster_boundaries(ax=axes[1,2])

plt.tight_layout()
plt.show()
```

### Interactive Exploration
```python
# Create interactive map exploration
viz.create_interactive_map(
    data=training_data,
    labels=data_labels,
    feature_names=['feature1', 'feature2', 'feature3']
)
```

## 🧬 Key Algorithmic Features

### Classical Kohonen SOM
- **Competitive Learning**: Winner-take-all neuron selection
- **Neighborhood Function**: Gaussian or Mexican-hat topological preservation
- **Learning Rate Decay**: Exponential or linear decay schedules
- **Distance Metrics**: Euclidean, Manhattan, or custom distance functions

### Growing Self-Organizing Maps (GSOM)
- **Dynamic Growth**: Automatic map expansion based on data requirements
- **Spread Factor**: Controls growth sensitivity and final map size
- **Boundary Neurons**: Special handling of edge neurons in growth process
- **Error-Based Growth**: Growth triggered by quantization error thresholds

### Hierarchical SOMs
- **Multi-Level Organization**: Coarse-to-fine data organization
- **Recursive Partitioning**: Each neuron can spawn sub-maps
- **Level-Specific Parameters**: Different learning rates and neighborhoods per level
- **Cross-Level Navigation**: Seamless traversal between hierarchy levels

## 📊 Implementation Highlights

- **Research Accuracy**: Faithful implementation of original Kohonen algorithms
- **Performance Optimized**: Vectorized operations with NumPy for efficiency
- **Flexible Architecture**: Modular design for easy experimentation
- **Rich Visualization**: Comprehensive plotting capabilities
- **Educational Value**: Clear code structure for learning SOM concepts

## 🔬 Applications Supported

### Data Analysis
- **Dimensionality Reduction**: High-dimensional data visualization
- **Cluster Discovery**: Unsupervised pattern recognition
- **Anomaly Detection**: Identification of outliers and rare patterns
- **Data Exploration**: Interactive analysis of complex datasets

### Specific Domains
- **Image Processing**: Color quantization and feature extraction
- **Text Mining**: Document clustering and topic discovery
- **Financial Analysis**: Market segmentation and risk analysis
- **Bioinformatics**: Gene expression analysis and protein classification

## 🎓 About the Implementation

Implemented by **Benedict Chen** - bringing foundational AI research to modern Python.

📧 Contact: benedict@benedictchen.com

---

## 💰 Support This Work - Donation Appreciated!

**This implementation represents hundreds of hours of research and development. If you find it valuable, please consider donating:**

**[💳 DONATE VIA PAYPAL - CLICK HERE](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS)**

**Your support helps maintain and expand these research implementations! 🙏**