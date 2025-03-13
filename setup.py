from setuptools import setup, find_packages

setup(
    name="litforge",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers"
    ],
    description="A DIY Python library for text generation using LLaMA and PyTorch.",
    author="Amey Kamble",
    author_email="nekotensor@gmail.com",
    url="https://github.com/NekoTensor/litforge",
)
