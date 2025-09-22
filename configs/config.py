import yaml
from pathlib import Path
from typing import Any

class RDMSettings:
    """Simplified RDM Configuration Loader"""
    
    def __init__(self):
        # Load config.dev.yaml by default
        config_path = Path(__file__).parent / "config.yaml"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                self._config = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'path.base_path')"""
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    @property
    def path(self):
        """Path settings accessor"""
        return self._config.get('path', {})
    
    @property
    def llm(self):
        """LLM settings accessor"""
        return self._config.get('llm', {})

# Global settings instance
settings = RDMSettings()