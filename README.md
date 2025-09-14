# ClusterAnalytics

Herramientas modulares para análisis de cúmulos abiertos usando SIMBAD y Gaia DR3.

## Estructura del proyecto

```
clusterAnalytics/
├── modules/                 # Módulos funcionales
│   ├── simbad_queries/     # Consultas a SIMBAD
│   ├── gaia_data/          # Obtención datos Gaia
│   ├── visualization/      # Gráficos y plots
│   └── analysis/           # Algoritmos de análisis
├── notebooks/              # Cuadernos Jupyter principales
├── data/                   # Datos del proyecto
│   ├── raw/               # Datos sin procesar
│   └── processed/         # Datos procesados
├── figures/                # Visualizaciones generadas
├── tests/                  # Tests unitarios
├── config/                 # Archivos de configuración
└── docs/                   # Documentación
```

## Instalación del entorno

```bash
# Crear entorno conda
conda create -n cluster_env python=3.10 jupyter pandas matplotlib astropy astroquery -c conda-forge

# Activar entorno
conda activate cluster_env
```

## Uso

```bash
# Activar entorno
conda activate cluster_env

# Ir al directorio del proyecto
cd clusterAnalytics

# Iniciar Jupyter desde notebooks
cd notebooks
jupyter notebook
```

## Módulos disponibles

### 🔍 simbad_queries
- Búsqueda de cúmulos por nombre
- Obtención de coordenadas y parámetros básicos
- Validación de datos SIMBAD

### 🌌 gaia_data  
- Consultas a Gaia DR3
- Filtrado por región y magnitud
- Procesamiento de datos astrométricos

### 📊 visualization
- Diagramas Hertzsprung-Russell
- Mapas de distribución espacial
- Gráficos de movimientos propios

### 📈 analysis
- Determinación de membership
- Cálculo de distancias
- Análisis de parámetros físicos

## Dependencias principales

- `astroquery` - Consultas astronómicas
- `astropy` - Cálculos astronómicos
- `pandas` - Manejo de datos
- `matplotlib` - Visualización
- `seaborn` - Gráficos estadísticos
- `jupyter` - Cuadernos interactivos

## Contribuir

1. Fork el repositorio
2. Crear rama para nueva funcionalidad
3. Realizar cambios y tests
4. Crear Pull Request

## Licencia

MIT License - ver archivo LICENSE para detalles.