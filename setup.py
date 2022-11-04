from setuptools import setup, find_packages


if __name__ == "__main__":
    setup(
    name='lxFlowAlign',
    version='0.0.1',
    author='sherif-med',
    description='A package for 2d flow estimation',
    long_description='',
    url='https://github.com/sherif-med/lxFlowAlign',
    keywords='DL, flow, correlation',
    python_requires='>=3.7',
    install_requires=[],
    packages=find_packages()
    )