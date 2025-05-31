from pydantic_settings import BaseSettings
from functools import lru_cache
import os
from pathlib import Path

class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: str
    AIML_API_KEY: str
    
    # Directory settings
    UPLOAD_DIR: str = "uploads"
    OUTPUT_DIR: str = "outputs"
    
    # API settings
    OPENAI_MODEL: str = "gpt-4-turbo"
    AIML_MODEL: str = "flux-pro/v1.1-ultra"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"

@lru_cache()
def get_settings():
    return Settings() 