#!/usr/bin/env python3
"""
API Server untuk Revolutionary AGI Forex Trading System
"""

import os
import sys
from pathlib import Path
import uvicorn
import argparse

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Load environment variables from .env file
from load_env import load_environment_variables
load_environment_variables()

def run_api_server(host="0.0.0.0", port=8000, reload=False):
    """Run the API server"""
    print(f"\n🚀 STARTING REVOLUTIONARY AGI FOREX TRADING API SERVER 🚀")
    print(f"🌐 API server running at http://{host}:{port}")
    print(f"📚 API documentation available at http://{host}:{port}/docs")
    print("Press Ctrl+C to stop the server...\n")
    
    # Import the API app
    from api.app import app
    
    # Run the API server
    uvicorn.run(
        "api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🚀 Revolutionary AGI Forex Trading API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host to bind the server to (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to bind the server to (default: 8000)'
    )
    
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Enable auto-reload on code changes (development mode)'
    )
    
    args = parser.parse_args()
    
    try:
        run_api_server(args.host, args.port, args.reload)
    except KeyboardInterrupt:
        print("\n🛑 API server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting API server: {e}")
        sys.exit(1)