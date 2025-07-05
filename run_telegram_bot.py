#!/usr/bin/env python3
"""
Telegram Bot untuk Revolutionary AGI Forex Trading System
"""

import os
import sys
from pathlib import Path
import argparse
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Load environment variables from .env file
from load_env import load_environment_variables
load_environment_variables()

# Import logger
from utils.logger import setup_logging

def run_telegram_bot(webhook_mode=False):
    """Run the Telegram bot"""
    # Setup logging
    logger = setup_logging()
    
    # Import the bot
    from telegram_bot.advanced_bot import AdvancedTelegramBot
    from utils.config_manager import ConfigManager
    
    # Load configuration
    config = ConfigManager("config/config.yaml").get_config()
    
    print("\n🚀 STARTING REVOLUTIONARY AGI FOREX TRADING TELEGRAM BOT 🚀")
    
    # Check if bot token is available
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN") or config.get("telegram", {}).get("bot_token")
    if not bot_token or bot_token.startswith("${"):
        logger.error("❌ Telegram bot token not found in environment variables or config")
        print("❌ Error: Telegram bot token not found. Please set TELEGRAM_BOT_TOKEN environment variable.")
        return
    
    # Create and start the bot
    try:
        bot = AdvancedTelegramBot(config)
        
        if webhook_mode:
            # Get webhook URL
            webhook_url = os.environ.get("TELEGRAM_WEBHOOK_URL") or config.get("telegram", {}).get("webhook_url")
            if not webhook_url or webhook_url.startswith("${"):
                logger.error("❌ Webhook URL not found in environment variables or config")
                print("❌ Error: Webhook URL not found. Please set TELEGRAM_WEBHOOK_URL environment variable.")
                return
            
            print(f"🌐 Starting bot in webhook mode with URL: {webhook_url}")
            bot.start_webhook(webhook_url)
        else:
            print("🔄 Starting bot in polling mode")
            bot.start_polling()
            
        print("✅ Telegram bot started successfully")
        print("Press Ctrl+C to stop the bot...\n")
        
        # Keep the bot running
        bot.idle()
        
    except KeyboardInterrupt:
        logger.info("🛑 Telegram bot stopped by user")
        print("\n🛑 Telegram bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Error starting Telegram bot: {e}")
        print(f"\n❌ Error starting Telegram bot: {e}")
    finally:
        if 'bot' in locals():
            bot.stop()
            print("✅ Telegram bot stopped successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🚀 Revolutionary AGI Forex Trading Telegram Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--webhook',
        action='store_true',
        help='Run bot in webhook mode instead of polling mode'
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
    
    try:
        run_telegram_bot(args.webhook)
    except KeyboardInterrupt:
        print("\n🛑 Telegram bot terminated by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)