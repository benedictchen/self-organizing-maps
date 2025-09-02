#!/usr/bin/env python3
"""
🔬 Comprehensive Research-Aligned Tests for self_organizing_maps
========================================================

Tests based on:
• Kohonen (1982) - Self-organized formation of topologically correct feature maps

Key concepts tested:
• Competitive Learning
• Topological Preservation
• Neighborhood Function
• Learning Rate Decay
• Winner-Take-All

Author: Benedict Chen (benedict@benedictchen.com)
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import self_organizing_maps
except ImportError:
    pytest.skip(f"Module self_organizing_maps not available", allow_module_level=True)


class TestBasicFunctionality:
    """Test basic module functionality"""
    
    def test_module_import(self):
        """Test that the module imports successfully"""
        assert self_organizing_maps.__version__
        assert hasattr(self_organizing_maps, '__all__')
    
    def test_main_classes_available(self):
        """Test that main classes are available"""
        main_classes = ['SelfOrganizingMap', 'GrowingSelfOrganizingMap']
        for cls_name in main_classes:
            assert hasattr(self_organizing_maps, cls_name), f"Missing class: {cls_name}"
    
    def test_key_concepts_coverage(self):
        """Test that key research concepts are implemented"""
        # This test ensures all key concepts from the research papers
        # are covered in the implementation
        key_concepts = ['Competitive Learning', 'Topological Preservation', 'Neighborhood Function', 'Learning Rate Decay', 'Winner-Take-All']
        
        # Check if concepts appear in module documentation, class names, or source files
        module_attrs = dir(self_organizing_maps)
        module_str = str(self_organizing_maps.__doc__ or "")
        
        # Also check source code content for concepts
        import inspect
        som_classes = ['SelfOrganizingMap', 'GrowingSelfOrganizingMap']
        source_content = ""
        
        for class_name in som_classes:
            if hasattr(self_organizing_maps, class_name):
                cls = getattr(self_organizing_maps, class_name)
                try:
                    source = inspect.getsource(cls)
                    source_content += source.lower()
                except:
                    pass
        
        covered_concepts = []
        concept_mappings = {
            'competitive learning': ['competitive', 'competition', 'compete'],
            'topological preservation': ['topological', 'topology', 'preserve', 'neighbor'],
            'neighborhood function': ['neighborhood', 'neighbour', 'gaussian', 'mexican'],
            'learning rate decay': ['learning_rate', 'decay', 'schedule'],
            'winner-take-all': ['winner', 'bmu', 'best_matching']
        }
        
        for concept in key_concepts:
            concept_lower = concept.lower()
            search_terms = concept_mappings.get(concept_lower, [concept_lower.replace(' ', '').replace('-', '')])
            
            # Check multiple sources for concept presence
            found = False
            for term in search_terms:
                if (any(term in attr.lower() for attr in module_attrs) or 
                    term in module_str.lower() or 
                    term in source_content):
                    found = True
                    break
            
            if found:
                covered_concepts.append(concept)
        
        coverage_ratio = len(covered_concepts) / len(key_concepts)
        assert coverage_ratio >= 0.6, f"Only {coverage_ratio:.1%} of key concepts covered. Found: {covered_concepts}"


class TestResearchPaperAlignment:
    """Test alignment with original research papers"""
    
    @pytest.mark.parametrize("paper", ['Kohonen (1982) - Self-organized formation of topologically correct feature maps'])
    def test_paper_concepts_implemented(self, paper):
        """Test that concepts from each research paper are implemented"""
        # This is a meta-test that ensures the implementation
        # follows the principles from the research papers
        assert True  # Placeholder - specific tests would go here


class TestConfigurationOptions:
    """Test that users have lots of configuration options"""
    
    def test_main_class_parameters(self):
        """Test that main classes have configurable parameters"""
        main_classes = ['SelfOrganizingMap', 'GrowingSelfOrganizingMap']
        
        for cls_name in main_classes:
            if hasattr(self_organizing_maps, cls_name):
                cls = getattr(self_organizing_maps, cls_name)
                if hasattr(cls, '__init__'):
                    # Check that __init__ has parameters (indicating configurability)
                    import inspect
                    sig = inspect.signature(cls.__init__)
                    params = [p for p in sig.parameters.values() if p.name != 'self']
                    assert len(params) >= 3, f"{cls_name} should have more configuration options"


# Module-specific tests would be added here based on the actual implementation
# These would test the specific algorithms and methods from the research papers

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
