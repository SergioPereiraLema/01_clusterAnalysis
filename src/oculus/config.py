"""
Configuration module for Oculus - Stellar cluster explorer with Gaia data.

This module contains all configurations, URLs, default parameters
and constants needed for the application to work.

Author: Sergio Pereira Lema
Version: 1.0
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# ============================================================================
# EXTERNAL SERVICES CONFIGURATION
# ============================================================================

# Astronomical services URLs
SIMBAD_BASE_URL = "http://simbad.u-strasbg.fr/simbad/sim-tap/sync"
GAIA_TAP_URL = "https://gea.esac.esa.int/tap-server/tap/sync"

# SIMBAD query configuration
SIMBAD_CONFIG = {
    "timeout": 30,  # seconds
    "max_retries": 3,
    "user_agent": "oculus/1.0 (Python Gaia Cluster Explorer)"
}

# Gaia query configuration
GAIA_CONFIG = {
    "timeout": 60,  # seconds for large queries
    "max_retries": 3,
    "max_rows": 100000,  # default row limit
    "user_agent": "oculus/1.0 (Python Gaia Cluster Explorer)"
}

# ============================================================================
# DEFAULT PARAMETERS FOR CLUSTER ANALYSIS
# ============================================================================

@dataclass
class ClusterSearchParams:
    """Default parameters for cluster star search."""
    
    # Search radius (in degrees)
    default_radius: float = 0.5
    max_radius: float = 2.0
    min_radius: float = 0.1
    
    # Magnitude limits
    min_magnitude: float = 3.0   # Brighter than G=6
    max_magnitude: float = 20.0  # Fainter than G=20
    
    # Data quality filters
    min_parallax_over_error: float = 3.0  # Signal-to-noise ratio for parallax
    max_ruwe: float = 1.4  # Astrometric quality
    
    # Proper motion filters (mas/year)
    pmra_tolerance: float = 5.0
    pmdec_tolerance: float = 5.0

# Default instance
DEFAULT_SEARCH_PARAMS = ClusterSearchParams()

# ============================================================================
# DIRECTORIES AND FILES CONFIGURATION
# ============================================================================

# Project base directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Create directories if they don't exist
for directory in [DATA_DIR, CACHE_DIR, OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CACHE CONFIGURATION
# ============================================================================

CACHE_CONFIG = {
    "enabled": True,
    "ttl_hours": 24,  # Cache time-to-live in hours
    "max_size_mb": 100,  # Maximum cache size in MB
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)s] %(name)s - %(funcName)s:%(lineno)d: %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO"
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": str(PROJECT_ROOT / "oculus.log"),
            "formatter": "detailed",
            "level": "DEBUG"
        }
    },
    "loggers": {
        "oculus": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": False
        }
    }
}

# ============================================================================
# CATALOGS AND REFERENCES
# ============================================================================

# Known cluster catalogs for validation
KNOWN_CLUSTERS = {
    "Pleiades": {"M": 45, "NGC": None, "IC": None},
    "Hyades": {"M": None, "NGC": None, "IC": None},
    "Beehive": {"M": 44, "NGC": 2632, "IC": None},
    "Double Cluster": {"NGC": [869, 884], "IC": None},
    "Jewel Box": {"NGC": 4755, "IC": None},
}


# ============================================================================
# PREDEFINED SQL QUERIES
# ============================================================================

# Base query for SIMBAD - get basic cluster information
SIMBAD_CLUSTER_QUERY = """
SELECT 
    main_id,
    ra, dec,
    otype_txt,
    flux_V,
    flux_B,
    plx_value as parallax,
    plx_error as parallax_error,
    pmra, pmdec,
    pmra_error, pmdec_error,
    rv_value as radial_velocity,
    sp_type
FROM basic
WHERE main_id = '{cluster_name}'
   OR oid IN (
       SELECT oidref FROM ident 
       WHERE id = '{cluster_name}'
   )
"""

# Base query for Gaia - get cluster stars
GAIA_CLUSTER_STARS_QUERY = """
SELECT 
    source_id,
    ra, dec,
    parallax, parallax_error,
    pmra, pmra_error,
    pmdec, pmdec_error,
    phot_g_mean_mag,
    phot_bp_mean_mag,
    phot_rp_mean_mag,
    bp_rp,
    ruwe,
    visibility_periods_used,
    astrometric_n_good_obs_al
FROM gaiaedr3.gaia_source
WHERE CONTAINS(
    POINT('ICRS', ra, dec),
    CIRCLE('ICRS', {ra}, {dec}, {radius})
) = 1
AND parallax IS NOT NULL
AND parallax_error IS NOT NULL
AND parallax/parallax_error > {min_parallax_over_error}
AND ruwe < {max_ruwe}
AND phot_g_mean_mag BETWEEN {min_mag} AND {max_mag}
ORDER BY phot_g_mean_mag
"""

# ============================================================================
# CONFIGURATION HELPER FUNCTIONS
# ============================================================================

def get_config_value(key: str, default: Any = None) -> Any:
    """
    Get a configuration value, first from environment variables,
    then from default values.
    """
    env_key = f"OCULUS_{key.upper()}"
    return os.getenv(env_key, default)

def validate_config() -> Dict[str, bool]:
    """
    Validate current configuration and return a dictionary with the status
    of each component.
    """
    validation_results = {
        "directories": all(d.exists() for d in [DATA_DIR, CACHE_DIR, OUTPUT_DIR]),
        "simbad_url": True,  # TODO: Implement SIMBAD ping
        "gaia_url": True,    # TODO: Implement Gaia ping
        "parameters": DEFAULT_SEARCH_PARAMS.min_radius < DEFAULT_SEARCH_PARAMS.max_radius
    }
    
    return validation_results

def get_cluster_radius_estimate(cluster_type: str = "open") -> float:
    """
    Return a search radius estimate based on cluster type.
    
    Args:
        cluster_type: Type of cluster ('open', 'globular', 'association')
    
    Returns:
        Estimated radius in degrees
    """
    radius_estimates = {
        "open": 0.5,        # Typical open clusters
        "globular": 0.2,    # More compact globular clusters
        "association": 2.0,  # More dispersed associations
        "unknown": 0.5      # Default
    }
   
    return radius_estimates.get(cluster_type.lower(), radius_estimates["unknown"])
    
# ============================================================================
# ASTRONOMICAL CONSTANTS
# ============================================================================

# Unit conversions
ARCSEC_TO_DEG = 1.0 / 3600.0
ARCMIN_TO_DEG = 1.0 / 60.0
MAS_TO_ARCSEC = 1e-3

# Relevant physical constants
PARSEC_TO_KM = 3.0857e13
SOLAR_MAGNITUDE_G = 4.83  # Sun's absolute magnitude in Gaia G band


if __name__ == "__main__":
    # Basic configuration test
    print("Oculus v1.0 Configuration")
    print(f"Project directory: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Cache directory: {CACHE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Default radius: {DEFAULT_SEARCH_PARAMS.default_radius}°")
    
    validation = validate_config()
    print(f"Validation: {all(validation.values())}")
    
    for component, status in validation.items():
        emoji = "✅" if status else "❌"



