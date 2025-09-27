# setup.py
# Setup script for the lab2 package, specifying metadata and dependencies.

from setuptools import setup, find_packages # Import setup tools

# Setup configuration
setup(
    name="lab2",
    version="0.1.0",
    packages=find_packages(),   # finds src/ because it has __init__.py
    python_requires=">=3.8",
)