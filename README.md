# TCGA Radiogenomics Processing Protocol

Radiology image processing of TCGA/TCIA data.

![alt text](https://github.com/pachterlab/cellsweep/blob/main/figures/fig1.png?raw=true)

## Getting Started

To run the code in this repository, follow these steps:

```sh
git clone https://github.com/pachterlab/tcga-radiogenomics.git
cd tcga-radiogenomics
```

We recommend using an environment manager such as conda. Some additional non-python packages must be installed for full functionality. If using conda (recommended), simply run the following:

```sh
conda env create -f environment.yml
conda activate tcga_radiogenomics
```

Otherwise, install these packages manually as-needed (see environment.yml and pyproject.toml for the list of packages and recommended versions).

(Optional) For development, explicitely install the package in editable mode with the following command:

```sh
pip install -e .
```

## Notebooks
See the [notebooks](notebooks) directory for the data processing protocol in a Jupyter notebook format.

## License  
This project is licensed under the **BSD 2-Clause License**. See the [LICENSE](LICENSE) file for details.

---

For any issues or contributions, feel free to open a pull request or issue in this repository.
