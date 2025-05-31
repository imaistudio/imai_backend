from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class APIConfig(BaseSettings):
    """API configuration settings."""
    openai_api_key: str = Field(..., env='OPENAI_API_KEY')
    claude_api_key: str = Field(..., env='CLAUDE_API_KEY')
    flux_api_key: str = Field(..., env='FLUX_API_KEY')
    gemini_api_key: str = Field(..., env='GEMINI_API_KEY')
    
    openai_api_base: str = Field('https://api.openai.com/v1', env='OPENAI_API_BASE')
    claude_api_base: str = Field('https://api.anthropic.com/v1', env='CLAUDE_API_BASE')
    flux_api_base: str = Field('https://api.flux.ai/v1', env='FLUX_API_BASE')
    
    max_retries: int = Field(3, env='MAX_RETRIES')
    timeout: int = Field(30, env='TIMEOUT')
    max_concurrent_requests: int = Field(5, env='MAX_CONCURRENT_REQUESTS')

class ImageConfig(BaseSettings):
    """Image processing configuration settings."""
    max_image_size: int = Field(4096, env='MAX_IMAGE_SIZE')
    supported_formats: List[str] = Field(['jpg', 'jpeg', 'png', 'webp'], env='SUPPORTED_FORMATS')
    temp_dir: Path = Field(Path('./temp'), env='TEMP_DIR')
    
    @validator('temp_dir')
    def create_temp_dir(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v

class OutputConfig(BaseSettings):
    """Output configuration settings."""
    output_dir: Path = Field(Path('./output'), env='OUTPUT_DIR')
    backup_dir: Path = Field(Path('./backup'), env='BACKUP_DIR')
    
    @validator('output_dir', 'backup_dir')
    def create_dirs(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v

class LogConfig(BaseSettings):
    """Logging configuration settings."""
    log_level: str = Field('INFO', env='LOG_LEVEL')
    log_file: Optional[Path] = Field(Path('./logs/app.log'), env='LOG_FILE')
    
    @validator('log_file')
    def create_log_dir(cls, v: Optional[Path]) -> Optional[Path]:
        if v:
            v.parent.mkdir(parents=True, exist_ok=True)
        return v

class Settings(BaseSettings):
    """Main settings class combining all configurations."""
    api: APIConfig = APIConfig()
    image: ImageConfig = ImageConfig()
    output: OutputConfig = OutputConfig()
    log: LogConfig = LogConfig()
    
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

# Create global settings instance
settings = Settings()

# Export settings for easy access
__all__ = ['settings'] 