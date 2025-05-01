from setuptools import setup, find_packages

setup(
    name='uvrules',
    version='0.1',
    packages=find_packages(),
    description='A package to generate uv-complete radio arrays.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Vincent MacKay',
    author_email='vince.mackay@gmail.com',
    url='https://github.com/vincentmackay/uvrules',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'astropy',
    ],
    python_requires='>=3.8',
)
