#!/usr/bin/env python3
"""
Entry point for running CBT experiments.
Run this from the project root directory.
"""

import sys
import os

# Add the current directory to Python path (this is the project root)
if __name__ == "__main__":
    # This script should be run from the project root
    sys.path.insert(0, os.getcwd())
    
    # Import and run the experiments
    from experiments.run_trained_experiments import main
    main() 