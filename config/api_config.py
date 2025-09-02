import os
import json
from typing import Dict, Any, Optional

class APIConfig:
    """API configuration management."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize API configuration.
        
        Args:
            config_file: Optional path to a JSON config file
        """
        # Default configuration
        self.default_config = {
            "openai": {
                "default_model": "gpt-3.5-turbo-0125",
                "advanced_model": "gpt-4o",
                "max_tokens": 4000,
                "temperature": 0.2,
                "request_timeout": 60,
                "retry_count": 3,
                "rate_limit_wait": 20  # seconds to wait if rate limited
            },
            "url_processing": {
                "timeout": 30,
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "max_content_size": 1000000,  # bytes
                "respect_robots_txt": True
            },
            "analysis": {
                "max_entities": 30,
                "max_topics": 10,
                "similarity_threshold": 0.5,
                "content_chunk_size": 4000
            }
        }
        
        # Load from file if provided
        self.config = self.default_config.copy()
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                    self._update_nested_dict(self.config, file_config)
            except Exception as e:
                print(f"Error loading config file: {e}")
    
    def _update_nested_dict(self, d: Dict, u: Dict) -> Dict:
        """Recursively update a nested dictionary."""
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self._update_nested_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            default: Default value if not found
            
        Returns:
            Configuration value or default
        """
        return self.config.get(section, {}).get(key, default)
    
    def get_section(self, section: str) -> Dict:
        """
        Get an entire configuration section.
        
        Args:
            section: Configuration section
            
        Returns:
            Section dictionary or empty dict
        """
        return self.config.get(section, {})
    
    def update(self, section: str, key: str, value: Any) -> None:
        """
        Update a configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            value: New value
        """
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
    
    def save(self, config_file: str) -> bool:
        """
        Save configuration to file.
        
        Args:
            config_file: Path to save the config
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
