from setuptools import setup, find_packages

setup(
    name="main_main_workflow",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "python-multipart>=0.0.6",
        "openai>=1.12.0",
        "anthropic>=0.8.0",
        "requests>=2.31.0",
        "google-generativeai>=0.3.2",
        "Pillow>=10.2.0",
        "numpy>=1.26.0",
        "opencv-python>=4.9.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.6.0",
        "pydantic-settings>=0.1.1",
        "pyyaml>=6.0.1",
        "loguru>=0.7.2",
        "prometheus-client>=0.19.0",
    ],
) 