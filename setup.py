import pathlib

from setuptools import find_packages, setup

setuptools.setup(
    name="iMSminer",
    version="1.0.0",
    description="iMSminer provides user-friendly, partially GPU- or compiler-accelerated multi-condition, multi-ROI, and multi-dataset preprocessing and mining of larger-than-memory imaging mass spectrometry datasets in Python.",
    long_description=pathlib.Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/Prentice-lab-UF/iMSminer",
    author="Yu Tin Lin",
    author_email="yutinlin@stanford.edu",
    license="Apache Software License",
    install_requires=[
        "bokeh==3.4.1",
        "opencv-python==4.10.0.84",
        "matplotlib==3.7.1",
        "msalign==0.2.0",
        "networkx==3.2.1",
        "numba==0.60.0",
        "numpy==1.26.0",
        "pandas==1.5.3",
        "psutil==5.9.0",
        "pyimzml==1.5.4",
        "scikit-learn==1.6.0",
        "scipy==1.14.1",
        "seaborn==0.11.2",
        "statsmodels==0.14.2",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
)
