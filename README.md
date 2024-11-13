<!-- <h1 align="center">SonicBatt</h1> -->
<!-- ![SonicBatt](assets/SonicBatt_logo.png)
 -->
 <img src="assets/SonicBatt_logo.png" alt="SonicBatt" width="200"/>
<a href="https://codecov.io/gh/EliasGaliounas/SonicBatt" > 
 <img src="https://codecov.io/gh/EliasGaliounas/SonicBatt/branch/main/graph/badge.svg?token=O7VLF7G0P9"/> 
 </a>

<h1 align="left">Documentation</h1>
https://eliasgaliounas.github.io/SonicBatt/

<h1 align="left">Setting up</h1>

1) Clone the repository to a local folder
   - Create a local directory
   - Clone the github repository to that directory: `git clone https://github.com/EliasGaliounas/SonicBatt.git`
2) Create anaconda environment:
   - Launch the Anaconda prompt
   - Activate your base environment (it's likely to have packages needed to create a new environment)
   - `conda env create -f <path/to/cloned_repository>\environment.yml` (replace <path/to/cloned_repository> with you path. Don't keep the <>).
3) Install SonicBatt
   - `cd` to the <path/to/cloned_repository>
   - `pip install .` (or alternatively `pip install -e .` to install in editable mode)

<h1 align="left">Published studies and reproducibility</h1>

**1) Investigations into the Dynamic Acoustic Response of Lithium-Ion Batteries During Lifetime Testing**
   - Manuscript DOI: [10.1149/1945-7111/ad5d21](https://doi.org/10.1149/1945-7111/ad5d21)
   - Dataset DOI: [10.5522/04/25343527.v1](https://doi.org/10.5522/04/25343527.v1)
   - Instructions:
      - Download `Raw data.zip` from the dataset, unzip, and place into the `studies/degradation` directory. There will now be a folder with all the data at: `studies/degradation/Raw data`
      - Run the notebooks inside studies/degradation sequentially to process the dataset, create ancillary files, and produce the main plots of the publication.

**2) The generalisation challenge: assessment of the efficacy of acoustic signals for state estimation of lithium-ion batteries via machine learning**
   - Manuscript DOI: [10.26434/chemrxiv-2024-93b2q](https://doi.org/10.26434/chemrxiv-2024-93b2q)
   - Dataset DOI: [10.5522/04/26843797.v1](https://doi.org/10.5522/04/26843797.v1)
   - Instructions:
      - Download `Raw data.zip` from the dataset, unzip, and place into the `studies/multi-cell_ml`. There will now be a folder with all the data at: `studies/multi-cell_ml/Raw data`
      - Download `Models.zip` from the dataset, unzip, and place into the `studies/multi-cell_ml`. There will now be a folder with all the data at: `studies/multi-cell_ml/Models`
      - Run the notebooks inside `studies/multi-cell_ml` sequentially to process the dataset, create ancillary files, and produce the main plots of the publication. SInce you have downloaded the models, you don't need to run notebooks 4 to 10.
      - You can also download separately the `Animations.zip` from the dataset, and then you won't need to run notebook 2.