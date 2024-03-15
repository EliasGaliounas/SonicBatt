<h1 align="center">SonicBatt</h1>

[![codecov](https://codecov.io/gh/EliasGaliounas/SonicBatt/graph/badge.svg?token=O7VLF7G0P9)](https://codecov.io/gh/EliasGaliounas/SonicBatt)

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

1) Investigations into the acoustic response of batteries during dynamic tests, through life
   - Manuscript available at: (Add link)
   - Dataset available at: https://rdr.ucl.ac.uk/articles/dataset/Acoustic_response_of_batteries_during_dynamic_tests_through_life/25343527/1
   - Instructions:
      - Download dataset, unzip, and place into the studies/degradation directory. I.e. there will now be a folder with all the data at studies/degradation/Raw data
      - Run each notebook inside studies/degradation to produce the main plots of the study
