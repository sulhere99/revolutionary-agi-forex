#!/usr/bin/env python3
"""
AGI Forex Trading System - Startup Script
=========================================

Script untuk menjalankan sistem AGI Forex Trading dengan berbagai mode:
- Development mode
- Production mode
- Testing mode
- Backtesting mode
"""

import asyncio
import argparse
import sys
import os
import signal
import logging
from pathlib import Path
from typing import Optional
import uvicorn
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import main system
from main import AGIForexTradingSystem
from utils.logger import setup_logging

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="AGI Forex Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --mode development --debug
  python run.py --mode production --config config/production.yaml
  python run.py --mode backtest --start-date 2023-01-01 --end-date 2023-12-31
  python run.py --mode paper-trading --pairs EURUSD,GBPUSD
        """
    )
    
    # Mode selection
    parser.add_argument(
        '--mode', '-m',
        choices=['development', 'production', 'testing', 'backtest', 'paper-trading'],
        default='development',
        help='Running mode (default: development)'
    )
    
    # Configuration
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/config.yaml',
        help='Configuration file path (default: config/config.yaml)'
    )
    
    # Debug mode
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug mode'
    )
    
    # Log level
    parser.add_argument(
        '--log-level', '-l',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Log level (default: INFO)'
    )
    
    # API server options
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='API server host (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=8000,
        help='API server port (default: 8000)'
    )
    
    # Trading options
    parser.add_argument(
        '--pairs',
        type=str,
        help='Currency pairs to trade (comma-separated, e.g., EURUSD,GBPUSD)'
    )
    
    parser.add_argument(
        '--timeframes',
        type=str,
        help='Timeframes to analyze (comma-separated, e.g., M15,H1,H4)'
    )
    
    # Backtesting options
    parser.add_argument(
        '--start-date',
        type=str,
        help='Backtest start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='Backtest end date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--initial-balance',
        type=float,
        default=10000.0,
        help='Initial balance for backtesting (default: 10000)'
    )
    
    # Feature flags
    parser.add_argument(
        '--enable-telegram',
        action='store_true',
        help='Enable Telegram bot'
    )
    
    parser.add_argument(
        '--enable-trading',
        action='store_true',
        help='Enable live trading (use with caution!)'
    )
    
    parser.add_argument(
        '--enable-paper-trading',
        action='store_true',
        default=True,
        help='Enable paper trading (default: True)'
    )
    
    parser.add_argument(
        '--disable-ai',
        action='store_true',
        help='Disable AI components (manual trading only)'
    )
    
    # Performance options
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of worker processes (default: 1)'
    )
    
    parser.add_argument(
        '--max-memory',
        type=str,
        help='Maximum memory usage (e.g., 4G, 8G)'
    )
    
    # Monitoring options
    parser.add_argument(
        '--enable-metrics',
        action='store_true',
        default=True,
        help='Enable metrics collection (default: True)'
    )
    
    parser.add_argument(
        '--metrics-port',
        type=int,
        default=9090,
        help='Metrics server port (default: 9090)'
    )
    
    # Development options
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Enable auto-reload for development'
    )
    
    parser.add_argument(
        '--profile',
        action='store_true',
        help='Enable performance profiling'
    )
    
    # Utility commands
    parser.add_argument(
        '--check-config',
        action='store_true',
        help='Check configuration and exit'
    )
    
    parser.add_argument(
        '--check-dependencies',
        action='store_true',
        help='Check dependencies and exit'
    )
    
    parser.add_argument(
        '--version',
        action='store_true',
        help='Show version and exit'
    )
    
    return parser.parse_args()

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'tensorflow', 'torch',
        'fastapi', 'uvicorn', 'redis', 'sqlalchemy', 'psycopg2',
        'python-telegram-bot', 'ccxt', 'yfinance', 'ta-lib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All required dependencies are installed")
    return True

def check_environment():
    """Check environment variables and configuration"""
    required_env_vars = [
        'OANDA_API_KEY',
        'TELEGRAM_BOT_TOKEN',
        'DB_PASSWORD',
        'REDIS_PASSWORD'
    ]
    
    missing_vars = []
    
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("⚠️  Missing environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nSet these variables in your .env file or environment")
        return False
    
    print("✅ All required environment variables are set")
    return True

def setup_signal_handlers(system: AGIForexTradingSystem):
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        asyncio.create_task(system.stop_system())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

async def run_development_mode(args):
    """Run in development mode"""
    print("🔧 Starting in DEVELOPMENT mode")
    
    # Setup logging with debug level
    logger = setup_logging()
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    
    # Create system with development configuration
    system = AGIForexTradingSystem(args.config)
    
    # Override configuration for development
    config = system.config_manager.get_config()
    config['system']['environment'] = 'development'
    config['system']['debug'] = args.debug
    config['api']['debug'] = args.debug
    config['api']['reload'] = args.reload
    
    # Setup signal handlers
    setup_signal_handlers(system)
    
    try:
        # Start system
        await system.start_system()
        
        # Run API server
        uvicorn_config = uvicorn.Config(
            system.app,
            host=args.host,
            port=args.port,
            debug=args.debug,
            reload=args.reload,
            log_level=args.log_level.lower()
        )
        
        server = uvicorn.Server(uvicorn_config)
        await server.serve()
        
    except KeyboardInterrupt:
        print("\n🛑 Received keyboard interrupt")
    except Exception as e:
        logger.error(f"❌ Error in development mode: {e}")
        raise
    finally:
        await system.stop_system()

async def run_production_mode(args):
    """Run in production mode"""
    print("🚀 Starting in PRODUCTION mode")
    
    # Setup logging
    logger = setup_logging()
    logger.setLevel(getattr(logging, args.log_level))
    
    # Create system
    system = AGIForexTradingSystem(args.config)
    
    # Override configuration for production
    config = system.config_manager.get_config()
    config['system']['environment'] = 'production'
    config['system']['debug'] = False
    config['api']['debug'] = False
    config['api']['reload'] = False
    
    # Setup signal handlers
    setup_signal_handlers(system)
    
    try:
        # Start system
        await system.start_system()
        
        # Run API server with multiple workers
        uvicorn_config = uvicorn.Config(
            system.app,
            host=args.host,
            port=args.port,
            workers=args.workers,
            log_level=args.log_level.lower(),
            access_log=True
        )
        
        server = uvicorn.Server(uvicorn_config)
        await server.serve()
        
    except Exception as e:
        logger.error(f"❌ Error in production mode: {e}")
        raise
    finally:
        await system.stop_system()

async def run_backtest_mode(args):
    """Run in backtest mode"""
    print("📊 Starting BACKTEST mode")
    
    if not args.start_date or not args.end_date:
        print("❌ Backtest mode requires --start-date and --end-date")
        sys.exit(1)
    
    # Setup logging
    logger = setup_logging()
    
    # Import backtesting module
    try:
        from utils.backtester import Backtester
    except ImportError:
        print("❌ Backtesting module not available")
        sys.exit(1)
    
    # Create backtester
    backtester = Backtester(
        config_path=args.config,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_balance=args.initial_balance
    )
    
    # Set currency pairs
    if args.pairs:
        pairs = [pair.strip() for pair in args.pairs.split(',')]
        backtester.set_currency_pairs(pairs)
    
    # Run backtest
    try:
        results = await backtester.run_backtest()
        
        # Display results
        print("\n📈 BACKTEST RESULTS")
        print("=" * 50)
        print(f"Period: {args.start_date} to {args.end_date}")
        print(f"Initial Balance: ${args.initial_balance:,.2f}")
        print(f"Final Balance: ${results['final_balance']:,.2f}")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.2%}")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        
        # Save detailed results
        results_file = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        backtester.save_results(results_file)
        print(f"\n💾 Detailed results saved to: {results_file}")
        
    except Exception as e:
        logger.error(f"❌ Error in backtest: {e}")
        raise

async def run_paper_trading_mode(args):
    """Run in paper trading mode"""
    print("📝 Starting PAPER TRADING mode")
    
    # Setup logging
    logger = setup_logging()
    
    # Create system
    system = AGIForexTradingSystem(args.config)
    
    # Override configuration for paper trading
    config = system.config_manager.get_config()
    config['system']['environment'] = 'paper_trading'
    config['trading']['paper_trading'] = True
    config['trading']['live_trading'] = False
    
    # Set currency pairs
    if args.pairs:
        pairs = [pair.strip() for pair in args.pairs.split(',')]
        config['data_collection']['currency_pairs']['major'] = pairs
    
    # Setup signal handlers
    setup_signal_handlers(system)
    
    try:
        # Start system
        await system.start_system()
        
        print("📝 Paper trading started. Press Ctrl+C to stop.")
        
        # Keep running
        while system.is_running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Paper trading stopped")
    except Exception as e:
        logger.error(f"❌ Error in paper trading: {e}")
        raise
    finally:
        await system.stop_system()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Handle utility commands
    if args.version:
        print("AGI Forex Trading System v2.0.0")
        sys.exit(0)
    
    if args.check_dependencies:
        if check_dependencies():
            sys.exit(0)
        else:
            sys.exit(1)
    
    if args.check_config:
        try:
            from utils.config_manager import ConfigManager
            config_manager = ConfigManager(args.config)
            config = config_manager.get_config()
            print("✅ Configuration is valid")
            
            # Check environment variables
            if check_environment():
                print("✅ Environment is properly configured")
                sys.exit(0)
            else:
                sys.exit(1)
                
        except Exception as e:
            print(f"❌ Configuration error: {e}")
            sys.exit(1)
    
    # Check basic requirements
    if not check_dependencies():
        sys.exit(1)
    
    # Set environment variables from args
    if args.debug:
        os.environ['DEBUG'] = 'true'
    
    # Run based on mode
    try:
        if args.mode == 'development':
            asyncio.run(run_development_mode(args))
        elif args.mode == 'production':
            asyncio.run(run_production_mode(args))
        elif args.mode == 'backtest':
            asyncio.run(run_backtest_mode(args))
        elif args.mode == 'paper-trading':
            asyncio.run(run_paper_trading_mode(args))
        elif args.mode == 'testing':
            # Run tests
            import subprocess
            result = subprocess.run(['python', '-m', 'pytest', 'tests/', '-v'])
            sys.exit(result.returncode)
        else:
            print(f"❌ Unknown mode: {args.mode}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()