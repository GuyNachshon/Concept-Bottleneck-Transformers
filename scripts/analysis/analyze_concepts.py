#!/usr/bin/env python3
"""
Simple script to run concept analysis on the latest CBT model.
"""

import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Run concept analysis on the latest model."""
    print("🔍 CBT Concept Analysis")
    print("=" * 50)
    
    try:
        # Import and run the concept analysis
        from experiments.run_concept_analysis import main as run_concept_analysis
        run_concept_analysis()
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