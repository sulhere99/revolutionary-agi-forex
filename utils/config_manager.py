"""
Configuration Manager
====================

Manages system configuration with environment variable substitution,
validation, and hot reloading capabilities.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import re
from datetime import datetime
import threading
import time

logger = logging.getLogger(__name__)

class ConfigManager:
    """Advanced configuration manager with environment variable substitution"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.last_modified: Optional[datetime] = None
        self.watch_thread: Optional[threading.Thread] = None
        self.watching = False
        
        # Load initial configuration
        self.load_config()
        
        # Start watching for changes
        self.start_watching()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file with environment variable substitution"""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            # Check if file was modified
            current_modified = datetime.fromtimestamp(self.config_path.stat().st_mtime)
            if self.last_modified and current_modified <= self.last_modified:
                return self.config
            
            logger.info(f"Loading configuration from {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as file:
                raw_config = yaml.safe_load(file)
            
            # Substitute environment variables
            self.config = self._substitute_env_vars(raw_config)
            
            # Validate configuration
            self._validate_config()
            
            self.last_modified = current_modified
            logger.info("Configuration loaded successfully")
            
            return self.config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def _substitute_env_vars(self, obj: Any) -> Any:
        """Recursively substitute environment variables in configuration"""
        if isinstance(obj, dict):
            return {key: self._substitute_env_vars(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            return self._substitute_string_env_vars(obj)
        else:
            return obj
    
    def _substitute_string_env_vars(self, text: str) -> str:
        """Substitute environment variables in a string"""
        # Pattern to match ${VAR_NAME} or ${VAR_NAME:default_value}
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        
        def replace_var(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else ""
            
            env_value = os.getenv(var_name)
            if env_value is not None:
                return env_value
            elif default_value:
                return default_value
            else:
                logger.warning(f"Environment variable {var_name} not found and no default provided")
                return f"${{{var_name}}}"  # Return original if not found
        
        return re.sub(pattern, replace_var, text)
    
    def _validate_config(self):
        """Validate configuration structure and required fields"""
        required_sections = [
            'system',
            'database',
            'redis',
            'agi_brain',
            'data_collection',
            'telegram',
            'api'
        ]
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Required configuration section '{section}' not found")
        
        # Validate specific fields
        self._validate_database_config()
        self._validate_api_config()
        self._validate_telegram_config()
        
        logger.info("Configuration validation passed")
    
    def _validate_database_config(self):
        """Validate database configuration"""
        db_config = self.config.get('database', {})
        
        if 'primary' not in db_config:
            raise ValueError("Primary database configuration not found")
        
        primary_db = db_config['primary']
        required_fields = ['type', 'host', 'port', 'name', 'username']
        
        for field in required_fields:
            if field not in primary_db:
                raise ValueError(f"Required database field '{field}' not found")
    
    def _validate_api_config(self):
        """Validate API configuration"""
        api_config = self.config.get('api', {})
        
        if 'security' not in api_config:
            raise ValueError("API security configuration not found")
        
        security_config = api_config['security']
        if 'api_keys' not in security_config or not security_config['api_keys']:
            raise ValueError("API keys not configured")
    
    def _validate_telegram_config(self):
        """Validate Telegram configuration"""
        telegram_config = self.config.get('telegram', {})
        
        if 'bot_token' not in telegram_config:
            raise ValueError("Telegram bot token not configured")
        
        # Check if bot token looks valid (should start with a number followed by colon)
        bot_token = telegram_config['bot_token']
        if not re.match(r'^\d+:', bot_token):
            logger.warning("Telegram bot token format appears invalid")
    
    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration"""
        return self.config.copy()
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get a specific configuration section"""
        return self.config.get(section, {})
    
    def get_value(self, path: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation (e.g., 'database.primary.host')"""
        keys = path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set_value(self, path: str, value: Any):
        """Set a configuration value using dot notation"""
        keys = path.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the value
        config[keys[-1]] = value
        logger.info(f"Configuration value updated: {path} = {value}")
    
    def start_watching(self):
        """Start watching configuration file for changes"""
        if self.watching:
            return
        
        self.watching = True
        self.watch_thread = threading.Thread(target=self._watch_config_file, daemon=True)
        self.watch_thread.start()
        logger.info("Started watching configuration file for changes")
    
    def stop_watching(self):
        """Stop watching configuration file"""
        self.watching = False
        if self.watch_thread:
            self.watch_thread.join(timeout=5)
        logger.info("Stopped watching configuration file")
    
    def _watch_config_file(self):
        """Watch configuration file for changes"""
        while self.watching:
            try:
                if self.config_path.exists():
                    current_modified = datetime.fromtimestamp(self.config_path.stat().st_mtime)
                    if self.last_modified and current_modified > self.last_modified:
                        logger.info("Configuration file changed, reloading...")
                        self.load_config()
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error watching configuration file: {e}")
                time.sleep(10)  # Wait longer on error
    
    def reload_config(self):
        """Manually reload configuration"""
        logger.info("Manually reloading configuration...")
        self.load_config()
    
    def export_config(self, output_path: str, include_env_vars: bool = False):
        """Export current configuration to a file"""
        try:
            config_to_export = self.config.copy()
            
            if not include_env_vars:
                # Remove sensitive information
                config_to_export = self._remove_sensitive_data(config_to_export)
            
            with open(output_path, 'w', encoding='utf-8') as file:
                yaml.dump(config_to_export, file, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")
            raise
    
    def _remove_sensitive_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive data from configuration"""
        sensitive_keys = [
            'password', 'token', 'key', 'secret', 'api_key', 'bot_token',
            'access_token', 'refresh_token', 'webhook_url'
        ]
        
        def clean_dict(obj):
            if isinstance(obj, dict):
                cleaned = {}
                for key, value in obj.items():
                    if any(sensitive_key in key.lower() for sensitive_key in sensitive_keys):
                        cleaned[key] = "***REDACTED***"
                    else:
                        cleaned[key] = clean_dict(value)
                return cleaned
            elif isinstance(obj, list):
                return [clean_dict(item) for item in obj]
            else:
                return obj
        
        return clean_dict(config)
    
    def validate_environment_variables(self) -> Dict[str, bool]:
        """Validate that all required environment variables are set"""
        required_env_vars = []
        
        def find_env_vars(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    find_env_vars(value, f"{path}.{key}" if path else key)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    find_env_vars(item, f"{path}[{i}]")
            elif isinstance(obj, str):
                # Find environment variable references
                pattern = r'\$\{([^}:]+)(?::[^}]*)?\}'
                matches = re.findall(pattern, obj)
                for match in matches:
                    required_env_vars.append((match, path))
        
        find_env_vars(self.config)
        
        # Check which environment variables are set
        env_var_status = {}
        for env_var, path in required_env_vars:
            is_set = os.getenv(env_var) is not None
            env_var_status[f"{env_var} (used in {path})"] = is_set
            
            if not is_set:
                logger.warning(f"Environment variable {env_var} is not set (used in {path})")
        
        return env_var_status
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get information about the current environment"""
        return {
            "config_file": str(self.config_path),
            "config_exists": self.config_path.exists(),
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
            "watching": self.watching,
            "environment_variables": self.validate_environment_variables(),
            "system_info": {
                "python_version": os.sys.version,
                "platform": os.name,
                "working_directory": os.getcwd(),
                "user": os.getenv("USER", "unknown")
            }
        }

# Example usage and testing
if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    
    # Test configuration manager
    try:
        config_manager = ConfigManager("config/config.yaml")
        
        # Test getting configuration
        config = config_manager.get_config()
        print(f"System name: {config_manager.get_value('system.name')}")
        print(f"Database host: {config_manager.get_value('database.primary.host')}")
        
        # Test environment info
        env_info = config_manager.get_environment_info()
        print(f"Environment info: {env_info}")
        
        # Test export
        config_manager.export_config("config_export.yaml", include_env_vars=False)
        
    except Exception as e:
        print(f"Error testing configuration manager: {e}")