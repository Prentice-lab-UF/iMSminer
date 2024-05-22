iMSminer Beta

## **Welcome to iMSminer!**
**iMSminer** provides user-friendly, partial GPU- or compiler-accelerated multi-condition, multi-ROI, and multi-dataset preprocessing and mining of larger-than-memory imaging mass spectrometry datasets in Python.

## **Portals**
- **Quick Start**: https://drive.google.com/drive/folders/12Qjz5zlSMwL42W0X_yZxZVZaVXtlhylo?usp=drive_link 
- **Website**: {insert website link}
- **Setup Instructions**: {insert setup tutorials}
- **Usage Tutorials**: {insert hyperlink to documentation}
- **Documentation**: https://github.com/Prentice-lab-UF/iMSminer/blob/main/iMSminer/docs/_build/html/index.html (to be put on public domain)
- **Support Group**: {insert Google support}
- **Feedback Forms**: https://forms.gle/W7TwYy7NvewKvb5n8

## **Features**
- Interactive question prompting to enhance user-friendliness
- Preprocesses imzML datasets with peak picking, mass alignment, and peak integration
- Enables interactive ROI annotation and selection
- Optional data normalization, internal calibration, MS1 search, and analyte filtering
- Unsupervised learning to cluster by molecular co-localization or *in situ* molecular profile
- Univariate fold-change statistics with pairwise ROI comparisons
- Visualiztion of ion image and ion statistics 

## **Citation**
Please consider citing iMSminer and related modules if iMSminer is helpful to your work
```
@software{imsminer2024,
  author = {Yu Tin Lin, Haohui Bao, Troy R. Scoggings IV, Boone M. Prentice},
  title = {{iMSminer}: A Data Processing and Machine Learning Package for Imaging Mass Spectrometry},
  url = {https://github.com/Prentice-lab-UF/iMSminer},
  version = {beta},
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
