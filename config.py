from pydantic_settings import BaseSettings
from decouple import config
import os

class Settings(BaseSettings):
    # wandb settings
    WANDB_API_KEY: str = config("WANDB_API_KEY", cast=str)
    WANDB_TEAM: str = config("WANDB_TEAM", cast=str)

    # other settings
    USER: str = str(os.getlogin())
    class Config:
        case_sensitive = True


settings = Settings()