#!/usr/bin/env python3
"""
Demo mode untuk Revolutionary AGI Forex Trading System
"""

import os
import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Load environment variables from .env file
from load_env import load_environment_variables
load_environment_variables()

# Set demo mode environment variable
os.environ["REVOLUTIONARY_DEMO_MODE"] = "true"

# Import run_revolutionary_system from run_revolutionary.py
from run_revolutionary import run_revolutionary_system, print_revolutionary_banner, print_technology_status

async def run_demo():
    """Run the system in demo mode"""
    print("\n🚀 RUNNING REVOLUTIONARY AGI FOREX TRADING SYSTEM IN DEMO MODE 🚀\n")
    print("⚠️ Demo mode: No real trades will be executed")
    print("⚠️ Demo mode: Using simulated market data")
    print("⚠️ Demo mode: API keys not required\n")
    
    # Print revolutionary banner
    print_revolutionary_banner()
    
    # Print technology status
    print_technology_status()
    
    print("\n🎮 DEMO MODE ACTIVATED 🎮")
    print("Press Ctrl+C to stop the demo...\n")
    
    try:
        # Run the revolutionary system with demo config
        await run_revolutionary_system("config/config.yaml")
    except KeyboardInterrupt:
        print("\n🛑 Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Error in demo mode: {e}")
    finally:
        print("\n✅ Demo completed successfully!")

if __name__ == "__main__":
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        print("\n🛑 Demo terminated by user")
    except Exception as e:
        print(f"\n❌ Fatal error in demo: {e}")
        sys.exit(1)