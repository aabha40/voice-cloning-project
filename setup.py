from setuptools import setup, find_packages

setup(
    name="voiceforge",
    version="1.0.0",
    description="Zero-shot AI voice cloning system using Coqui XTTS-v2",
    authors="Aabha Shukla, Prachi Jha",
    author_email="aabhasiddhishukla@gmail.com, prachijhaa.2901@gmail.com",
    url="https://github.com/aabha40/voice-cloning-project",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "TTS>=0.22.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.1",
        "flask>=3.0.0",
        "numpy>=1.24.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
)
