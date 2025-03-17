from setuptools import setup, find_packages

setup(
    name="recurrent_memory_transformer",
    version="0.1.0",
    description="Recurrent Memory Transformer for handling long context",
    author="KotShinZ",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
    ],
    python_requires=">=3.6",
    license="Apache License 2.0",
)
