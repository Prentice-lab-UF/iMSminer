## **Welcome to iMSminer!**
**iMSminer** provides user-friendly, partially GPU- or compiler-accelerated multi-ROI and multi-dataset preprocessing and mining of larger-than-memory imaging mass spectrometry datasets in Python.

## **Resources**
- [**Colab Notebooks**](https://drive.google.com/drive/folders/12Qjz5zlSMwL42W0X_yZxZVZaVXtlhylo?usp=sharing)
- [**Case Studies**](https://docs.google.com/spreadsheets/d/1esn5cARyUs4zbKMVBf7dozmiyD8pQCzWL2dn3S29MAE/edit?usp=sharing)
- [**Tutorials and Documentation**](https://prentice-lab-uf.github.io/iMSminer/)
- [**PyPI**](https://pypi.org/project/iMSminer/)
- [**Support Group**](https://groups.google.com/g/imsminer-help)
- [**Feedback Form**](https://forms.gle/C16Hrp9ibdtWgyH17)


## **Features**
- Interactive input prompts to enhance user-friendliness
- Preprocesses imzML datasets via peak picking, baseline subtraction (optional), mass alignment (optional), and peak integration
- Interactive ROI annotation and selection
- Optional data normalization, internal calibration, MS1 search, MS2 confirmation, and analyte filtering
- Unsupervised learning to extract patterns based on molecular co-localization or *in situ* molecular profile
- Univariate fold-change statistics with ROI statistics
- Visualiztion of ion image and ion statistics
- Quickstart guides on Google Colab

## **Installation (Local)**
### **iMSminer** 
```python
pip install iMSminer
```
### **GPU-Accelerated Packages**
For optional NVIDIA® CUDA® GPU acceleration, install:  
#### [**Cupy**](https://docs.cupy.dev/en/stable/install.html)
#### [**RAPIDS**](https://docs.rapids.ai/install?_gl=1*1p3fcd0*_ga*MTQxMDQwNDI5NC4xNzE0ODU0NzQx*_ga_RKXFW6CM42*MTcxODg1NzY3MS4xMS4xLjE3MTg4NTc4NTYuNjAuMC4w#wsl2)

## **Usage**
Usage guide with commonly tuned parameters
```python
# =====Load iMSminer Modules=====#
from iMSminer import data_preprocessing, data_analysis, utils, ImzMLParser_chunk

# =====Preprocess imzML=====#
## specify folder path containing imzML's to preprocess and folder path to save preprocessed data and figures 
preprocess = data_preprocessing.Preprocess()
## peak picking with optional mass alignment (if `peak_alignment=True`) and baseline subtraction (if `baseline_subtract=True`)
preprocess.peak_pick(
    percent_RAM=5,
    pp_method="automatic",
    rel_height=0.9,
    peak_alignment=True,
    align_threshold=1,
    align_halfwidth=100,
    grid_iter_num=20,
    align_reduce=False,
    reduce_halfwidth=200,
    plot_aligned_peak=True,
    index_peak_plot=50,
    plot_num_peaks=10,
    baseline_subtract=True,
    baseline_method="regression",
)
## peak integration with bounds rel_height and optional mass alignment (if `peak_alignment=True`)
preprocess.run(
    percent_RAM=5,
    peak_alignment=True,
    integrate_method="peak_width",
    align_halfwidth=100,
    grid_iter_num=20,
    align_reduce=False,
    reduce_halfwidth=200,
    plot_aligned_peak=True,
    index_peak_plot=50,
    plot_num_peaks=10,
)

# =====Analyze Preprocessed Data=====#
# FOR OPTIONAL FUNCTIONS, SKIP THE LINE IF NOT USING THE CAPABILITY
## specify folder path containing preprocessed data
analyze = data_analysis.DataAnalysis()
## ROI annotation and selection
analyze.load_preprocessed_data()
## optional normalization 
analyze.normalize_pixel(method="TIC")
## optional internal calibration
analyze.calibrate_mz()
## optional MS1_search 
analyze.MS1_search(
    ppm_threshold=5, MS1_search_method="avg_sepctrum", filter_db=True, percent_RAM=5
)
## optional analyte filtering 
analyze.filter_analytes(method="MS1")
## optional evaluation of image cluster validity  
analyze.optimize_image_clustering(k_max=min(10, data_analysis.mz.shape[0] - 1))
## optional evaluation of validity of in situ molecular profile 
analyze.optimize_insitu_clustering(k_max=10)
## image clustering with optional 3D t-SNE mapped in situ (if `insitu_tsne=True`)
analyze.image_clustering(
    k=5,
    perplexity=5,
    insitu_tsne=False,
    insitu_perplexity=3,
    zoom=0.15,
    quantile=99.9,
    replicate=0,
    img_plot_method="plot_ROI",
    feature_label="mz",
    jitter_amount=2,
    jitter_factor=5,
    font_size=20,
    ROI_size_divisor=10
)
# in situ segmentation
analyze.insitu_clustering(
    k=5, perplexity=15, show_ROI=True, show_square=True, replicate=0, ROI_size_divisor=10
) 
# volcano plot; heatmap (if `get_hm=True`) 
analyze.make_FC_plot(
    legend_label="condition",
    feature_label="mz",
    jitter_amount=0.5,
    jitter_factor=3,
    get_hm=True,
    hm_width_factor=10,
    hm_height_factor=20,
    hm_fontsize=20,
    hm_wspace=1.5,
    font_size=20,
)
# box plot ROI statistics
analyze.make_boxplot()
# ion image visualization
analyze.get_ion_image(
    replicate=0,
    show_ROI=True,
    show_square=True,
    color_scheme="inferno",
    quantile=99.9,
    ROI_size_divisor=10
)
```

## **Call for Contributions**
We appreciate contributions of any form, from feedback to debugging to method development. We enthusiastically welcome developers to interface their published models with iMSminer and host quickstart guides on Google Colab. Please feel free to contact us at [prenticelabuf@gmail.com](mailto:prenticelabuf@gmail.com). 

## **Citation**
Please consider citing iMSminer and related packages if iMSminer is helpful to your work
```
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
```
