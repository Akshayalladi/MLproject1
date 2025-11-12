from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'
def get_requirements(file_path :str) -> List[str]:
    """Reads the requirements from a file and returns them as a list."""
    requirements : List[str] = []

    with open(file_path, 'r') as file:
        requirements = file.readlines()
        Finalrequirements = [req.replace("\n","") for req in requirements]
    
        if HYPEN_E_DOT in Finalrequirements:
            Finalrequirements.remove(HYPEN_E_DOT)
    return Finalrequirements






setup(
    name='my_package',
    version='0.1.0',
    author='Akshay Kumar',
    author_email= 'alladiakshay44@gmail.com',
    packages= find_packages(),
    install_requires=get_requirements('requirements.txt'),
    description='A sample Python package'
    )