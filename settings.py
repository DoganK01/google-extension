from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_API_VERSION: str
    AZURE_OPENAI_ENDPOINT: str
    AZURE_DEPLOYEMENT: str
    TAVILY_API_KEY: str
    LANGGRAPH_CLOUD_LICENSE_KEY: str

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
