from setuptools import setup, find_packages

setup(
    name="uvrules",
    version="0.1.0",
    description="An algorithm for generating uv-complete radio arrays (RULES: Regular UV Layout Engineering Strategy)",
    author="Vincent MacKay",
    packages=find_packages(),  # automatically includes uvrules/ and submodules
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "astropy",
        "finufft",
        "ipython",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Or update this to your real license
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
)
