#!/usr/bin/env python3
"""
Simple script to run concept analysis on the latest CBT model.
"""

import sys
import os

# Add the project root directory to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def main():
    """Run concept analysis on the latest model."""
    print("🔍 CBT Concept Analysis")
    print("=" * 50)
    
    try:
        # Import and run the concept analysis
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_concept_analysis", 
            os.path.join(project_root, "experiments", "run_concept_analysis.py")
        )
        run_concept_analysis_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_concept_analysis_module)
        run_concept_analysis_module.main()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're in the correct directory and all dependencies are installed.")
        return 1
    except Exception as e:
        print(f"❌ Error running concept analysis: {e}")
        return 1
    
    print("\n✅ Concept analysis completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 