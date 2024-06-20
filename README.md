iMSminer Alpha

## **Welcome to iMSminer!**
**iMSminer** provides user-friendly, partially GPU- or compiler-accelerated multi-condition, multi-ROI, and multi-dataset preprocessing and mining of larger-than-memory imaging mass spectrometry datasets in Python.

## **Resources**
- [**Quick Start**](https://drive.google.com/drive/folders/12Qjz5zlSMwL42W0X_yZxZVZaVXtlhylo?usp=drive_link) 
- [**Tutorials and Documentation**](https://github.com/Prentice-lab-UF/iMSminer/blob/main/iMSminer/docs/_build/html/index.html) 
- [**Feedback Form**](https://forms.gle/C16Hrp9ibdtWgyH17)

## **Features**
- Interactive question prompts to enhance user-friendliness
- Preprocesses imzML datasets with peak picking, baseline subtraction, mass alignment, and peak integration
- Enables interactive ROI annotation and selection
- Optional data normalization, internal calibration, MS1 search, MS2 confirmation, and analyte filtering
- Unsupervised learning to extract patterns based on molecular co-localization or *in situ* molecular profile
- Univariate fold-change statistics with ROI comparisons
- Visualiztion of ion image and ion statistics
- Easily run on Google Colab

## **Installation (Local)**
### **iMSminer**
```python
pip install iMSminer
```
### **GPU-Accelerated Packages**
#### [**Cupy**](https://docs.cupy.dev/en/stable/install.html)
#### [**RAPIDS**](https://docs.rapids.ai/install?_gl=1*1p3fcd0*_ga*MTQxMDQwNDI5NC4xNzE0ODU0NzQx*_ga_RKXFW6CM42*MTcxODg1NzY3MS4xMS4xLjE3MTg4NTc4NTYuNjAuMC4w#wsl2)

## **Citation**
Please consider citing iMSminer and related packages if iMSminer is helpful to your work
```
@software{imsminer2024,
  author = {Yu Tin Lin and Haohui Bao and Troy R. Scoggings IV and Boone M. Prentice},
  title = {{iMSminer}: A Data Processing and Machine Learning Package for Imaging Mass Spectrometry},
  url = {https://github.com/Prentice-lab-UF/iMSminer},
  version = {1.0.0},
  year = {2024},
}

@software{pyimzml,
  author = {Alexandrov Team, EMBL},
  title = {{pyimzML}: A Parser to Read .imzML Files},
  url = {https://github.com/alexandrovteam/pyimzML},
  version = {1.5.4},
  year = {2024},
}

@software{msalign2024,
  author = {Lukasz G. Migas},
  title = {{msalign}: Spectral alignment based on MATLAB's `msalign` function},
  url = {https://github.com/lukasz-migas/msalign},
  version = {0.2.0},
  year = {2024},
}
```
