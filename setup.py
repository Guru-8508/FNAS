from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT="-e ."



def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        # Properly strip whitespace
        requirements = [req.strip() for req in requirements]
        
        # Remove empty lines, comments, and hash lines
        requirements = [req for req in requirements if req 
                       and not req.startswith('#') 
                       and not req.startswith('--hash')]
        
        # Remove editable install
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements
     


setup(
    name="facial-thermal-health-detector",
    version="1.0.0",
    description="AI-powered health deficiency detection using facial and thermal analysis",
    author="VITA FACE",
    author_email="gururathinam21@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
    
