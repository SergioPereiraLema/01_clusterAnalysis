# ClusterAnalytics

Tool to analyze Open Clusters using Gaia. UNDER CONSTRUCTION

## Project structure

```
clusterAnalytics/
├── modules/                 # Functional modules
│   ├── simbad_queries/     # SIMBAD queries
│   ├── gaia_data/          # Gaia data extraction
│   ├── visualization/      # Graphics
│   └── analysis/           # Analysis algorithms
├── notebooks/              # Jupyter notebooks
├── data/                   # Data
│   ├── raw/               # Raw data
│   └── processed/         # Processed data
├── figures/                # Figures generated
├── tests/                  # Tests
├── config/                 # Config files
└── docs/                   # Documentation
```

## Environment installation

```bash
# Create conda environment
conda create -n cluster_env python=3.12 jupyter pandas matplotlib astropy astroquery -c conda-forge

# Activate environment
conda activate cluster_env
```

## Use

```bash
# Activate environment
conda activate cluster_env

# Project directory
cd clusterAnalytics

# Init Jupyter from notebooks
cd notebooks
jupyter notebook
```

## Modules (Under construction)

### 🔍 simbad_queries
- Find clusters by name
- Get coordinates and basic astrometric data
- SIMBAD data validation

### 🌌 gaia_data  
- Queries to Gaia DR3
- Data filtering
- Astrometric data processing

### 📊 visualization
- Hertzsprung-Russell diagram
- Spatial distribution maps
- Proper motion graphs

### 📈 analysis
- Membership analysis
- Distance analysis
- Physical properties

## Dependencies

- `astroquery` 
- `astropy` 
- `pandas` 
- `matplotlib` 
- `seaborn` 
- `jupyter` 

## Contributions

1. Fork the repository
2. Create new branch for new funcionality
3. Developement and testing
4. Create Pull Request

## Licence

MIT License - ver archivo LICENSE para detalles.