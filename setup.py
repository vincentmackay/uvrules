from setuptools import setup, find_packages

setup(
    name='uv_complete',
    version='0.1',
    packages=find_packages(),
    description='A package to generate uv complete arrays.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Vincent MacKay',    author_email='vince.mackay@gmail.com',
    url='git@github.com:vincentmackay/uv-complete.git',
    install_requires=[
        'numpy','scipy','matplotlib'
    ],
)

