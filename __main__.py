"""Main entry point for the Semantic SEO Analyzer application.

This module allows the package to be executed directly with:
python -m semantic_seo_analyzer
"""

import os
import sys
import streamlit.web.cli as stcli

def main():
    """Launch the Streamlit application."""
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the app.py file
    app_path = os.path.join(current_dir, "app.py")
    
    # Check if app.py exists
    if not os.path.exists(app_path):
        print(f"Error: Could not find app.py at {app_path}")
        sys.exit(1)
    
    # Launch Streamlit app
    sys.argv = ["streamlit", "run", app_path]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()
