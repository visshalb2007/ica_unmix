"""CLI entry point for ica-unmix."""

import os
import sys

# Add the parent directory to sys.path so we can import scripts
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_ica import main

if __name__ == "__main__":
    main()
