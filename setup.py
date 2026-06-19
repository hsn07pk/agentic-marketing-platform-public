# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="agentic-ai-agent",
    version="1.0.0",
    author="Agentic",
    author_email="info@example.com",
    description="AI-driven simulation and orchestration platform for autonomous marketing agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/agentic/ai-agent-platform",
    packages=find_packages(exclude=["tests", "tests.*", "docs", "scripts"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "mypy>=1.7.1",
        ],
        "docs": [
            "mkdocs>=1.5.3",
            "mkdocs-material>=9.5.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "agentic-api=src.api.main:app",
            "agentic-dashboard=dashboard.app:main",
            "agentic-seed=scripts.seed_data:main",
            "agentic-validate=scripts.validate_simulation:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    zip_safe=False,
)