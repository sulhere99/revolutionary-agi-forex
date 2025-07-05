#!/usr/bin/env python3
"""
Load environment variables from .env file
"""

import os
from pathlib import Path
from dotenv import load_dotenv

def load_environment_variables():
    """Load environment variables from .env file"""
    # Get the directory containing this script
    script_dir = Path(__file__).parent.absolute()
    
    # Path to .env file
    env_path = script_dir / '.env'
    
    # Load environment variables from .env file
    load_dotenv(env_path)
    
    print("✅ Environment variables loaded successfully")
    
    # Return True if .env file exists, False otherwise
    return env_path.exists()

if __name__ == "__main__":
    # Load environment variables
    env_loaded = load_environment_variables()
    
    if env_loaded:
        print("Environment variables loaded from .env file")
    else:
        print("Warning: .env file not found. Using system environment variables.")