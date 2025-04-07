from setuptools import setup, find_packages

setup(name="machine_learning_course",
      version="1.0.0",
      description="Machine Learning Course",
      author="Adrien Dorise",
      author_email="adrien.dorise@hotmail.com",
      url="https://github.com/Adrien-Dorise/machine_learning_course",
      packages = [],
      install_requires=[
            "numpy >= 1.24.2",
            "matplotlib >= 3.8.1",
            "scikit-learn >= 1.3.2",
            "pandas >= 2.1.2",
            "seaborn >= 0.13.2",
            "liac-arff >= 2.5.0",
            "scipy >= 1.15.2"
      ], 
      python_requires=">=3.8.0",
     )