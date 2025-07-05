"""
Advanced Logging System
======================

Comprehensive logging system with multiple handlers, formatters,
and advanced features for the AGI Forex Trading System.
"""

import logging
import logging.handlers
import sys
import os
from pathlib import Path
from datetime import datetime
import json
import traceback
from typing import Dict, Any, Optional
import threading
import queue
import time

class ColoredFormatter(logging.Formatter):
    """Colored console formatter"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'thread_name': record.threadName,
            'process': record.process
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry)

class TelegramHandler(logging.Handler):
    """Custom handler to send critical logs to Telegram"""
    
    def __init__(self, bot_token: str, chat_id: str, level=logging.ERROR):
        super().__init__(level)
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.session = None
        
        # Rate limiting
        self.last_sent = {}
        self.rate_limit_seconds = 300  # 5 minutes
        
    def emit(self, record):
        try:
            # Rate limiting - don't spam Telegram
            message_key = f"{record.levelname}:{record.module}:{record.funcName}"
            current_time = time.time()
            
            if (message_key in self.last_sent and 
                current_time - self.last_sent[message_key] < self.rate_limit_seconds):
                return
            
            self.last_sent[message_key] = current_time
            
            # Format message for Telegram
            message = self._format_telegram_message(record)
            
            # Send to Telegram (simplified - in production, use proper async HTTP client)
            self._send_to_telegram(message)
            
        except Exception:
            # Don't let logging errors crash the application
            pass
    
    def _format_telegram_message(self, record):
        """Format log record for Telegram"""
        emoji_map = {
            'ERROR': '🔴',
            'CRITICAL': '💀',
            'WARNING': '⚠️'
        }
        
        emoji = emoji_map.get(record.levelname, '📝')
        
        message = f"{emoji} *{record.levelname}*\n\n"
        message += f"📍 *Module:* {record.module}\n"
        message += f"🔧 *Function:* {record.funcName}\n"
        message += f"📝 *Message:* {record.getMessage()}\n"
        message += f"⏰ *Time:* {datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        if record.exc_info:
            message += f"\n💥 *Exception:* {record.exc_info[1]}"
        
        return message
    
    def _send_to_telegram(self, message):
        """Send message to Telegram (placeholder implementation)"""
        # In production, implement proper async HTTP client
        pass

class DatabaseHandler(logging.Handler):
    """Custom handler to store logs in database"""
    
    def __init__(self, db_connection, level=logging.INFO):
        super().__init__(level)
        self.db_connection = db_connection
        self.buffer = queue.Queue(maxsize=1000)
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
    
    def emit(self, record):
        try:
            # Add to buffer for batch processing
            log_data = {
                'timestamp': datetime.fromtimestamp(record.created),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno,
                'thread': record.thread,
                'process': record.process
            }
            
            if record.exc_info:
                log_data['exception'] = str(record.exc_info[1])
                log_data['traceback'] = ''.join(traceback.format_exception(*record.exc_info))
            
            self.buffer.put_nowait(log_data)
            
        except queue.Full:
            # Buffer is full, drop the log entry
            pass
        except Exception:
            # Don't let logging errors crash the application
            pass
    
    def _worker(self):
        """Background worker to process log entries"""
        batch = []
        batch_size = 10
        timeout = 5  # seconds
        
        while True:
            try:
                # Collect batch of log entries
                try:
                    log_data = self.buffer.get(timeout=timeout)
                    batch.append(log_data)
                    
                    # Process batch when it's full or timeout occurs
                    while len(batch) < batch_size:
                        try:
                            log_data = self.buffer.get_nowait()
                            batch.append(log_data)
                        except queue.Empty:
                            break
                    
                    if batch:
                        self._store_batch(batch)
                        batch = []
                        
                except queue.Empty:
                    # Timeout occurred, process any pending entries
                    if batch:
                        self._store_batch(batch)
                        batch = []
                        
            except Exception as e:
                # Log error to console (avoid infinite recursion)
                print(f"Error in database logging worker: {e}")
                time.sleep(1)
    
    def _store_batch(self, batch):
        """Store batch of log entries in database"""
        try:
            # Implement database storage logic
            # This is a placeholder - implement based on your database schema
            pass
        except Exception as e:
            print(f"Error storing log batch to database: {e}")

class PerformanceLogger:
    """Performance logging utility"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.timers = {}
    
    def start_timer(self, name: str):
        """Start a performance timer"""
        self.timers[name] = time.time()
    
    def end_timer(self, name: str, log_level: int = logging.INFO):
        """End a performance timer and log the duration"""
        if name in self.timers:
            duration = time.time() - self.timers[name]
            self.logger.log(log_level, f"Performance: {name} took {duration:.4f} seconds", 
                          extra={'performance_metric': name, 'duration': duration})
            del self.timers[name]
            return duration
        return None
    
    def log_memory_usage(self):
        """Log current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            self.logger.info(f"Memory usage: RSS={memory_info.rss / 1024 / 1024:.2f}MB, "
                           f"VMS={memory_info.vms / 1024 / 1024:.2f}MB",
                           extra={'memory_rss': memory_info.rss, 'memory_vms': memory_info.vms})
        except ImportError:
            self.logger.warning("psutil not available for memory monitoring")

class LoggingManager:
    """Central logging manager"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.loggers = {}
        self.handlers = {}
        self.performance_logger = None
        
    def setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging system"""
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter(
            '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        self.handlers['console'] = console_handler
        
        # File handler for all logs
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "agi_forex.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=10
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-20s | %(module)s:%(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        self.handlers['file'] = file_handler
        
        # JSON file handler for structured logs
        json_handler = logging.handlers.RotatingFileHandler(
            log_dir / "agi_forex.json",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        json_handler.setLevel(logging.INFO)
        json_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(json_handler)
        self.handlers['json'] = json_handler
        
        # Error file handler
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / "errors.log",
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_handler)
        self.handlers['error'] = error_handler
        
        # Performance log handler
        performance_handler = logging.handlers.RotatingFileHandler(
            log_dir / "performance.log",
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3
        )
        performance_handler.setLevel(logging.INFO)
        performance_handler.addFilter(lambda record: hasattr(record, 'performance_metric'))
        performance_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(performance_handler)
        self.handlers['performance'] = performance_handler
        
        # Telegram handler for critical errors (if configured)
        telegram_config = self.config.get('telegram', {})
        if telegram_config.get('bot_token') and telegram_config.get('chats', {}).get('admin', {}).get('chat_id'):
            telegram_handler = TelegramHandler(
                telegram_config['bot_token'],
                telegram_config['chats']['admin']['chat_id'],
                level=logging.CRITICAL
            )
            root_logger.addHandler(telegram_handler)
            self.handlers['telegram'] = telegram_handler
        
        # Setup performance logger
        self.performance_logger = PerformanceLogger(root_logger)
        
        # Create main application logger
        main_logger = logging.getLogger('agi_forex')
        self.loggers['main'] = main_logger
        
        main_logger.info("🚀 Logging system initialized successfully")
        main_logger.info(f"📁 Log files location: {log_dir.absolute()}")
        
        return main_logger
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger with the given name"""
        if name not in self.loggers:
            self.loggers[name] = logging.getLogger(name)
        return self.loggers[name]
    
    def get_performance_logger(self) -> PerformanceLogger:
        """Get the performance logger"""
        return self.performance_logger
    
    def set_log_level(self, level: str):
        """Set log level for all handlers"""
        log_level = getattr(logging, level.upper(), logging.INFO)
        
        for handler in self.handlers.values():
            if handler != self.handlers.get('error'):  # Keep error handler at ERROR level
                handler.setLevel(log_level)
        
        logging.getLogger().info(f"Log level changed to {level.upper()}")
    
    def add_custom_handler(self, name: str, handler: logging.Handler):
        """Add a custom handler"""
        self.handlers[name] = handler
        logging.getLogger().addHandler(handler)
        logging.getLogger().info(f"Added custom handler: {name}")
    
    def remove_handler(self, name: str):
        """Remove a handler"""
        if name in self.handlers:
            logging.getLogger().removeHandler(self.handlers[name])
            del self.handlers[name]
            logging.getLogger().info(f"Removed handler: {name}")
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        stats = {
            'handlers': list(self.handlers.keys()),
            'loggers': list(self.loggers.keys()),
            'log_files': []
        }
        
        # Get log file information
        log_dir = Path("logs")
        if log_dir.exists():
            for log_file in log_dir.glob("*.log"):
                stats['log_files'].append({
                    'name': log_file.name,
                    'size': log_file.stat().st_size,
                    'modified': datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
                })
        
        return stats

def setup_logging(config: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """Setup logging system with default configuration"""
    if config is None:
        config = {
            'telegram': {
                'bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
                'chats': {
                    'admin': {
                        'chat_id': os.getenv('TELEGRAM_ADMIN_CHAT_ID')
                    }
                }
            }
        }
    
    logging_manager = LoggingManager(config)
    return logging_manager.setup_logging()

# Context manager for performance logging
class performance_timer:
    """Context manager for performance timing"""
    
    def __init__(self, name: str, logger: Optional[logging.Logger] = None, log_level: int = logging.INFO):
        self.name = name
        self.logger = logger or logging.getLogger()
        self.log_level = log_level
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.logger.log(self.log_level, f"Performance: {self.name} took {duration:.4f} seconds",
                          extra={'performance_metric': self.name, 'duration': duration})

# Decorator for function performance logging
def log_performance(name: Optional[str] = None, log_level: int = logging.INFO):
    """Decorator to log function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            timer_name = name or f"{func.__module__}.{func.__name__}"
            with performance_timer(timer_name, log_level=log_level):
                return func(*args, **kwargs)
        return wrapper
    return decorator

# Example usage
if __name__ == "__main__":
    # Test the logging system
    logger = setup_logging()
    
    # Test different log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Test performance logging
    with performance_timer("test_operation", logger):
        time.sleep(0.1)
    
    # Test performance decorator
    @log_performance("test_function")
    def test_function():
        time.sleep(0.05)
        return "test result"
    
    result = test_function()
    logger.info(f"Function result: {result}")
    
    # Test exception logging
    try:
        raise ValueError("Test exception")
    except Exception as e:
        logger.exception("An error occurred during testing")
    
    logger.info("Logging system test completed")