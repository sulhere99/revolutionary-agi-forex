#!/usr/bin/env python3
"""
Real Trading Mode untuk Revolutionary AGI Forex Trading System
"""

import os
import sys
import asyncio
from pathlib import Path
import argparse
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Load environment variables from .env file
from load_env import load_environment_variables
load_environment_variables()

# Import run_revolutionary_system from run_revolutionary.py
from run_revolutionary import run_revolutionary_system, print_revolutionary_banner, print_technology_status

async def run_real_trading(config_path="config/config.yaml"):
    """Run the system in real trading mode"""
    print("\n🚀 RUNNING REVOLUTIONARY AGI FOREX TRADING SYSTEM IN REAL TRADING MODE 🚀\n")
    print("⚠️ ATTENTION: This is REAL TRADING mode. Actual trades will be executed!")
    print("⚠️ ATTENTION: Real money will be at risk!")
    print("⚠️ ATTENTION: Make sure you have configured your API keys correctly!\n")
    
    # Set environment variable for real trading mode
    os.environ["REVOLUTIONARY_REAL_TRADING_MODE"] = "true"
    
    # Print revolutionary banner
    print_revolutionary_banner()
    
    # Print technology status
    print_technology_status()
    
    print("\n💰 REAL TRADING MODE ACTIVATED 💰")
    print("Press Ctrl+C to stop real trading...\n")
    
    try:
        # Run the revolutionary system with real trading config
        await run_revolutionary_system(config_path)
    except KeyboardInterrupt:
        print("\n🛑 Real trading stopped by user")
    except Exception as e:
        print(f"\n❌ Error in real trading mode: {e}")
    finally:
        print("\n✅ Real trading session completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🚀 Revolutionary AGI Forex Trading System - Real Trading Mode",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config', '-c',
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Confirm real trading mode
    print("\n⚠️ WARNING: You are about to start REAL TRADING mode ⚠️")
    print("Real money will be at risk. Are you sure you want to continue?")
    confirmation = input("Type 'yes' to confirm: ")
    
    if confirmation.lower() == 'yes':
        try:
            asyncio.run(run_real_trading(args.config))
        except KeyboardInterrupt:
            print("\n🛑 Real trading terminated by user")
        except Exception as e:
            print(f"\n❌ Fatal error in real trading: {e}")
            sys.exit(1)
    else:
        print("Real trading mode cancelled. Exiting...")
        sys.exit(0)