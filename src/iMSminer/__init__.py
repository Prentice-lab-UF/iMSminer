"""
iMSminer: A Data Processing and Machine Learning Package for Imaging Mass Spectrometry
@author: Yu Tin Lin (yutinlin@stanford.edu)
@author: Haohui Bao (susanab20020911@gmail.com)
@author: Troy R. Scoggins IV (t.scoggins@ufl.edu)
@author: Boone M. Prentice (booneprentice@ufl.chem.edu)
License: Apache-2.0

## **Call for Contributions**
We appreciate contributions of any form, from feedback to debugging to method development. We enthusiastically welcome developers to interface their published models with iMSminer and host quickstart guides on Google Colab. Please feel free to contact us at prenticelabuf@gmail.com. 

-----
Please consider citing iMSminer and related packages if iMSminer is helpful to your work

@article{Lin2024,
  title = {iMSminer: A Data Processing and Machine Learning Package for Imaging Mass Spectrometry},
  url = {http://dx.doi.org/10.26434/chemrxiv-2024-kxjgg},
  DOI = {10.26434/chemrxiv-2024-kxjgg},
  publisher = {American Chemical Society (ACS)},
  author = {Lin,  Yu Tin and Bao,  Haohui and Scoggins,  Troy and Prentice,  Boone},
  year = {2024},
  month = jun 
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
"""


from . import ImzMLParser_chunk, utils
from .data_analysis import DataAnalysis
from .data_preprocessing import Preprocess

__all__ = ['DataAnalysis', 'Preprocess']
