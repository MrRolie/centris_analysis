"""
Configuration management for centris_analysis project.

This module handles loading environment variables from .env file
and provides cross-platform path resolution.
"""

import os
import platform
from pathlib import Path
from typing import Optional


def load_env_file(env_path: Optional[str] = None) -> dict:
    """
    Load environment variables from .env file.
    
    Parameters
    ----------
    env_path : str, optional
        Path to .env file. If None, looks for .env in project root.
        
    Returns
    -------
    dict
        Dictionary of environment variables
    """
    if env_path is None:
        # Find project root (look for .env file)
        current_dir = Path(__file__).parent
        while current_dir != current_dir.parent:
            env_file = current_dir / '.env'
            if env_file.exists():
                env_path = str(env_file)
                break
            current_dir = current_dir.parent
    
    env_vars = {}
    
    if env_path and os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    
    return env_vars


def get_data_path() -> str:
    """
    Get the data file path based on platform and configuration.
    
    Returns
    -------
    str
        Path to the data file
        
    Raises
    ------
    FileNotFoundError
        If no valid data file path is found
    """
    # Load environment variables
    env_vars = load_env_file()
    
    # Check if direct path is set in environment
    direct_path = os.environ.get('PLEX_CSV_PATH')
    if direct_path and os.path.exists(direct_path):
        return direct_path
    
    # Platform-specific path resolution
    system = platform.system()
    
    if system == "Darwin":  # macOS
        path_key = 'PLEX_CSV_PATH_MACOS'
    elif system == "Windows":
        path_key = 'PLEX_CSV_PATH_WINDOWS'
    else:
        path_key = 'PLEX_CSV_PATH_FALLBACK'
    
    # Try platform-specific path from .env
    platform_path = env_vars.get(path_key)
    if platform_path and os.path.exists(platform_path):
        return platform_path
    
    # Try fallback path
    fallback_path = env_vars.get('PLEX_CSV_PATH_FALLBACK', '../../data/centris_comprehensive_plex_data.csv')
    if os.path.exists(fallback_path):
        return fallback_path
    
    # Last resort: try some common locations
    common_paths = [
        '../../data/centris_comprehensive_plex_data.csv',
        '../data/centris_comprehensive_plex_data.csv',
        './data/centris_comprehensive_plex_data.csv',
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    # If nothing works, provide helpful error message
    raise FileNotFoundError(
        f"Data file not found. Please check your .env file or set PLEX_CSV_PATH environment variable.\n"
        f"Current platform: {system}\n"
        f"Looking for key: {path_key}\n"
        f"Searched paths: {[platform_path, fallback_path] + common_paths}"
    )


def get_config() -> dict:
    """
    Get all configuration settings.
    
    Returns
    -------
    dict
        Configuration dictionary
    """
    env_vars = load_env_file()
    
    config = {
        'data_path': get_data_path(),
        'target_year': int(env_vars.get('TARGET_YEAR', 2025)),
        'assessment_growth_rate': float(env_vars.get('ASSESSMENT_GROWTH_RATE', 0.057)),
        'random_seed': int(env_vars.get('RANDOM_SEED', 42)),
        'platform': platform.system(),
    }
    
    return config


def print_config():
    """Print current configuration for debugging."""
    try:
        config = get_config()
        print("Configuration:")
        print("-" * 40)
        for key, value in config.items():
            print(f"{key:25}: {value}")
        print("-" * 40)
    except Exception as e:
        print(f"Error loading configuration: {e}")


if __name__ == "__main__":
    print_config()
