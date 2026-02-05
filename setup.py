from setuptools import setup, find_packages

setup(
    name='structured_optics',
    version='0.1',
    packages=find_packages(),
    author='Altilano C. Barbosa',
    isntall_requires=[
        'numpy >=2.4.2',
        'PyQt6 >=6.10.2',
        'scipy >=1.17.0'
    ]
)