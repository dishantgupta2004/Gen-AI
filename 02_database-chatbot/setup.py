"""
Setup script for the Database Chatbot application
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="database-chatbot",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A LangChain-based chatbot for MySQL database interaction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dishantgupta2004/database-chatbot",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        "streamlit>=1.45.1",
        "langchain>=0.3.25",
        "langchain-openai>=0.3.17",
        "langchain-groq>=0.3.2",
        "langchain-community>=0.3.24",
        "mysql-connector-python>=9.3.0",
        "pandas>=2.2.3",
        "plotly>=5.0.0",
        "python-dotenv>=1.1.0",
        "SQLAlchemy>=2.0.41",
        "numpy>=2.2.6",
        "pydantic>=2.11.4",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "run-chatbot=app:main",
        ],
    },
)