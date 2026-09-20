# Large magnetoresistance from interlayer hybridization in CrSBr: data and analysis code

Data, analysis scripts and Jupyter notebooks for the paper
([arXiv:2608.11389](https://doi.org/10.48550/arXiv.2608.11389))

> **Large bias-tunable magnetoresistance from spin-dependent interlayer
> hybridization in van der Waals antiferromagnet CrSBr-based heterostructures**

The work studies vertical tunnelling transport through a graphene / CrSBr /
graphene junction. Current-voltage curves
I(V) were recorded as a function of magnetic field applied along the crystal
**b** axis and the **c** axis, at temperatures from 3 K to 160 K. The
notebooks turn these into the magnetoresistance, the conductance-peak
spectroscopy, the Fowler-Nordheim analysis and the barrier-height versus
spin-canting-angle results reported in the paper. Bulk SQUID magnetometry and
an AFM topography scan of the device are included as well.

## Citation

If you use this code or data, please cite the paper (see also
[CITATION.cff](CITATION.cff)):

> S. Hameed, A. Kumar, C. Yu, A. P. Balan, X. Wang, L. Zhang, Y. Mokrousov and
> M. Kläui, *Large bias-tunable magnetoresistance from spin-dependent interlayer
> hybridization in van der Waals antiferromagnet CrSBr-based heterostructures*,
> arXiv:2608.11389 (2026). https://doi.org/10.48550/arXiv.2608.11389

```bibtex
@misc{hameed2026crsbr,
  title         = {Large bias-tunable magnetoresistance from spin-dependent interlayer hybridization in van der Waals antiferromagnet CrSBr-based heterostructures},
  author        = {Hameed, Sadeed and Kumar, Aditya and Yu, Chengjie and Balan, Aravind P. and Wang, Xinran and Zhang, Lichuan and Mokrousov, Yuriy and Kl{\"a}ui, Mathias},
  year          = {2026},
  eprint        = {2608.11389},
  archivePrefix = {arXiv},
  primaryClass  = {cond-mat.mes-hall},
  doi           = {10.48550/arXiv.2608.11389}
}
```

## Repository layout

```
├── data/                      Raw measurement files (read-only inputs)
│   ├── device 2/
│   │   ├── b_scans/           Field along b: H_scans/ (field sweeps) and IV_H_scans/ (I(V) at each H)
│   │   └── c_scans/           Field along c: same structure
│   ├── SQUID_bulk_CrSBr/      Bulk SQUID M(H) and M(T)
│   └── AFM data/              Bruker NanoScope scan of Device 2
├── output/IV_H_scans/         Processed tables the notebooks read (see below)
├── scripts/                   Loaders and analysis modules (importable package `scripts`)
├── notebooks/                 Analysis notebooks; run these
│   ├── Device_2/              Main analysis (b_scans/, c_scans/, barrier and canting analysis/, ...)
│   ├── compose figures/       Fig. 1c-e panels
│   ├── AFM data.ipynb         AFM topography and thickness line cut
│   └── SQUID_Bulk_MvsH.ipynb  Bulk magnetometry
└── pyproject.toml             Dependencies
```

## Setup

Python 3.12 or 3.13. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[notebook]"
jupyter lab
```

The editable install (`-e`) is what makes `import scripts` work from any
notebook. `pandas` is pinned below version 3: the analysis code was written
against pandas 2.x.

## Running the notebooks

- Open a notebook and run it top to bottom. Notebooks are stored **without
  outputs**, so figures appear only after you run them.
- Most notebooks start with
  `from scripts.utils import setup_notebook; PROJECT_ROOT, np, pd, plt, Path = setup_notebook()`,
  which sets the plotting style and finds the repository root. Paths inside the
  notebooks are relative to it or to the notebook's own folder, so launch
  Jupyter from the repository root or the notebook's folder.
- Notebooks write figures and intermediate files to `output/`. Rendered images
  (`*.png`, `*.pdf`) under `output/` are git-ignored; processed tables
  (`*.csv`, `*.pkl`) are tracked.

Where to start:

| Goal | Notebook |
|---|---|
| One temperature, I(H) curves, I(V) at fixed field, MR vs bias | `notebooks/Device_2/b_scans/analysis_<T>K.ipynb` and `c_scans/analysis_<T>K.ipynb` |
| MR across all temperatures | `notebooks/Device_2/MR_summary_main.ipynb` |
| Conductance-peak spectroscopy and Fowler-Nordheim analysis | `notebooks/Device_2/barrier and canting analysis/Spectroscopy and FN analysis_<T>K.ipynb` |
| Barrier height vs canting angle | `notebooks/Device_2/barrier and canting analysis/Barrier_vs_canting_<T>K.ipynb`, `Summary_T_dependence.ipynb` |
| Transport regimes (Arrhenius, canting, hopping) | `notebooks/Device_2/Transport_regimes_story.ipynb`, `Tunneling regime analysis*.ipynb` |
| Fig. 1c-e panels | `notebooks/compose figures/fig1cde_compose.ipynb` |

## Data

### Raw files (`data/`)

Each measurement is a tab-separated `.dat` file with a `!`-prefixed header
(date, sample name and ID, instruments, comments) followed by a column table
(date, time, `Tcryo`, `Tsample`, `Hx`, `Hy`, `Hz`, `|H|`, `phi`, `theta`, and
lock-in channels `L1_Ch1` ... `L5_Ch2`).

`IV_H_scans/` holds one `*.datfolder` per temperature and field sweep. The
folder name records the start timestamp, scan type (`b_scan` or `c_scan`),
temperature, number of field steps, maximum field and field angles, for example
`20251106153846b_scan_30K_92steps_H0.90T_phi_84.0to84.0_theta_0.0to0.0°_v0.datfolder`.
Inside, each `*.txt.dat` file is one I(V) curve at one field value.

### Processed tables (`output/IV_H_scans/`)

| Folder | Contents |
|---|---|
| `dataframes/{b,c}_scans/` | `IV_gaussian_<T>K.pkl/.csv`: Gaussian-filtered I(V, H) for each temperature (sigma = 1.5). `TMR_ratio_vs_V_<T>K.*`: MR vs bias. |
| `MR_summary/` | Per-temperature and combined MR summary tables. |
| `barrier_vs_canting/` | Fit results: band-edge peak, barrier height and exchange splitting, per temperature. |
| `fn_T_sweep/` | Fowler-Nordheim temperature sweep. |

The `.pkl` files are pandas pickles. They load with pandas 2.x and should be
treated as trusted local files (unpickling runs arbitrary code, so do not open
pickles from untrusted sources). Each has a `.csv` twin for the scalar columns.


## License

Code (`scripts/`, `notebooks/`) is released under the MIT License
([LICENSE.txt](LICENSE.txt)). The data in `data/` is released under
CC BY 4.0 ([LICENSE-DATA.txt](LICENSE-DATA.txt)), which requires citing the
paper above.

## Windows note

Some raw-data paths are longer than Windows' default 260-character limit. Clone
with `git clone -c core.longpaths=true <url>` and keep the checkout path short.
