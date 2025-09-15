"""
Data loading module for Oculus v1.0 - Open Cluster Unveiling & Limiting Uncertainty Studies

This module handles all external data connections to SIMBAD and Gaia TAP services,
including caching, error handling, and data validation.

Classes:
    - SimbadLoader: Handles SIMBAD database queries
    - GaiaLoader: Handles Gaia TAP service queries
    - DataCache: Simple file-based caching system

Author: Sergio Pereira Lema
Version: 1.0
"""
# ============================================================================
# IMPORTS AND CONFIGURATIONS
# ============================================================================

import os
import json
import hashlib
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urlencode
import requests
import pandas as pd
from io import StringIO

# Import configuration
from oculus.config import (
    SIMBAD_BASE_URL, GAIA_TAP_URL,
    SIMBAD_CONFIG, GAIA_CONFIG,
    SIMBAD_CLUSTER_QUERY, GAIA_CLUSTER_STARS_QUERY,
    CACHE_DIR, CACHE_CONFIG,
    DEFAULT_SEARCH_PARAMS
)

# Setup logging
logger = logging.getLogger('oculus.data_loader')

# ============================================================================
# EXCEPTIONS
# ============================================================================

class DataLoaderError(Exception):
    """Base exception for data loading operations."""
    pass

class SimbadError(DataLoaderError):
    """Exception raised for SIMBAD-specific errors."""
    pass

class GaiaError(DataLoaderError):
    """Exception raised for Gaia TAP service errors."""
    pass

class CacheError(DataLoaderError):
    """Exception raised for caching operations."""
    pass

# ============================================================================
# CACHE SYSTEM
# ============================================================================

class DataCache:
    """Simple file-based caching system for query results."""
    
    def __init__(self, cache_dir: Path = CACHE_DIR, config: Dict = CACHE_CONFIG):
        """
        Initialize cache system.
        
        Args:
            cache_dir: Directory to store cache files
            config: Cache configuration dictionary
        """
        self.cache_dir = cache_dir
        self.config = config
        self.enabled = config.get('enabled', True)
        self.ttl_hours = config.get('ttl_hours', 24)
        self.max_size_mb = config.get('max_size_mb', 100)
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"Cache initialized: {cache_dir}, enabled={self.enabled}")
    
    def _get_cache_key(self, query: str, params: Dict = None) -> str:
        """Generate a unique cache key for a query."""
        cache_data = {
            'query': query,
            'params': params or {}
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{cache_key}.json"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cached data is still valid based on TTL."""
        if not cache_path.exists():
            return False
        
        # Check file age
        file_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age_hours = (datetime.now() - file_time).total_seconds() / 3600
        
        return age_hours < self.ttl_hours
    
    def get(self, query: str, params: Dict = None) -> Optional[Any]:
        """
        Retrieve cached data for a query.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Cached data if available and valid, None otherwise
        """
        if not self.enabled:
            return None
        
        try:
            cache_key = self._get_cache_key(query, params)
            cache_path = self._get_cache_path(cache_key)
            
            if self._is_cache_valid(cache_path):
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                logger.debug(f"Cache hit for key: {cache_key[:8]}...")
                return cached_data['data']
            else:
                logger.debug(f"Cache miss for key: {cache_key[:8]}...")
                return None
                
        except Exception as e:
            logger.warning(f"Cache retrieval error: {e}")
            return None
    
    def set(self, query: str, data: Any, params: Dict = None) -> bool:
        """
        Store data in cache.
        
        Args:
            query: SQL query string
            data: Data to cache
            params: Query parameters
            
        Returns:
            True if successfully cached, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            cache_key = self._get_cache_key(query, params)
            cache_path = self._get_cache_path(cache_key)
            
            cached_data = {
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'params': params,
                'data': data
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cached_data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"Data cached with key: {cache_key[:8]}...")
            return True
            
        except Exception as e:
            logger.warning(f"Cache storage error: {e}")
            return False
    
    def clear(self) -> int:
        """Clear all cached data. Returns number of files removed."""
        removed_count = 0
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
                removed_count += 1
            logger.info(f"Cleared {removed_count} cache files")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
        
        return removed_count

# ============================================================================
# SIMBAD LOADER
# ============================================================================

class SimbadLoader:
    """Handles connections and queries to SIMBAD database."""
    
    def __init__(self, cache: Optional[DataCache] = None):
        """
        Initialize SIMBAD loader.
        
        Args:
            cache: Cache instance (optional)
        """
        self.base_url = SIMBAD_BASE_URL
        self.config = SIMBAD_CONFIG
        self.cache = cache or DataCache()
        
        # Setup session with default headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.config['user_agent'],
            'Accept': 'application/json',
            'Content-Type': 'application/x-www-form-urlencoded'
        })
        
        logger.debug("SIMBAD loader initialized")
    
    def _execute_query(self, query: str, format_type: str = 'json') -> Dict:
        """
        Execute a TAP query against SIMBAD.
        
        Args:
            query: ADQL query string
            format_type: Output format ('json', 'csv', 'votable')
            
        Returns:
            Query results as dictionary
            
        Raises:
            SimbadError: If query fails
        """
        # Check cache first
        cache_params = {'format': format_type}
        cached_result = self.cache.get(query, cache_params)
        if cached_result is not None:
            return cached_result
        
        # Prepare request data
        data = {
            'REQUEST': 'doQuery',
            'LANG': 'ADQL',
            'FORMAT': format_type,
            'QUERY': query
        }
        
        # Execute query with retries
        last_exception = None
        for attempt in range(self.config['max_retries']):
            try:
                logger.debug(f"SIMBAD query attempt {attempt + 1}/{self.config['max_retries']}")
                
                response = self.session.post(
                    self.base_url,
                    data=data,
                    timeout=self.config['timeout']
                )
                
                response.raise_for_status()
                
                # Parse response based on format
                if format_type == 'json':
                    result = response.json()
                else:
                    result = {'text': response.text}
                
                # Cache successful result
                self.cache.set(query, result, cache_params)
                
                logger.debug(f"SIMBAD query successful, {len(response.content)} bytes")
                return result
                
            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < self.config['max_retries'] - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"SIMBAD query failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"SIMBAD query failed after {self.config['max_retries']} attempts")
        
        raise SimbadError(f"Failed to execute SIMBAD query: {last_exception}")
    
    def get_cluster_info(self, cluster_name: str) -> Optional[Dict]:
        """
        Get basic information about a cluster from SIMBAD.
        
        Args:
            cluster_name: Name of the cluster to search
            
        Returns:
            Dictionary with cluster information or None if not found
            
        Raises:
            SimbadError: If query fails
        """
        logger.info(f"Querying SIMBAD for cluster: {cluster_name}")
        
        # Format the query with cluster name
        query = SIMBAD_CLUSTER_QUERY.format(cluster_name=cluster_name)
        
        try:
            result = self._execute_query(query)
            
            # Check if we got valid data
            if 'data' not in result or not result['data']:
                logger.warning(f"No data found in SIMBAD for cluster: {cluster_name}")
                return None
            
            # Extract first result (should be only one)
            cluster_data = result['data'][0]
            
            # Convert to more convenient format
            cluster_info = {
                'main_id': cluster_data.get('main_id'),
                'ra': cluster_data.get('ra'),
                'dec': cluster_data.get('dec'),
                'object_type': cluster_data.get('otype_txt'),
                'magnitude_v': cluster_data.get('flux_V'),
                'magnitude_b': cluster_data.get('flux_B'),
                'parallax': cluster_data.get('parallax'),
                'parallax_error': cluster_data.get('parallax_error'),
                'pmra': cluster_data.get('pmra'),
                'pmdec': cluster_data.get('pmdec'),
                'pmra_error': cluster_data.get('pmra_error'),
                'pmdec_error': cluster_data.get('pmdec_error'),
                'radial_velocity': cluster_data.get('rv_value'),
                'spectral_type': cluster_data.get('sp_type')
            }
            
            logger.info(f"Found cluster in SIMBAD: {cluster_info['main_id']} at "
                       f"({cluster_info['ra']:.4f}, {cluster_info['dec']:.4f})")
            
            return cluster_info
            
        except Exception as e:
            raise SimbadError(f"Error retrieving cluster info: {e}") from e
    
    def search_cluster_variants(self, cluster_name: str) -> List[str]:
        """
        Search for alternative names/identifiers for a cluster.
        
        Args:
            cluster_name: Cluster name to search variants for
            
        Returns:
            List of alternative identifiers
        """
        logger.debug(f"Searching SIMBAD for variants of: {cluster_name}")
        
        # Query to get all identifiers
        query = f"""
        SELECT DISTINCT id 
        FROM ident 
        WHERE oidref IN (
            SELECT oid FROM basic 
            WHERE main_id = '{cluster_name}'
            OR oid IN (
                SELECT oidref FROM ident WHERE id = '{cluster_name}'
            )
        )
        """
        
        try:
            result = self._execute_query(query)
            
            if 'data' in result and result['data']:
                variants = [row['id'] for row in result['data'] if row['id']]
                logger.debug(f"Found {len(variants)} variants for {cluster_name}")
                return variants
            else:
                return []
                
        except Exception as e:
            logger.warning(f"Error searching cluster variants: {e}")
            return []

# ============================================================================
# GAIA LOADER
# ============================================================================

class GaiaLoader:
    """Handles connections and queries to Gaia TAP service."""
    
    def __init__(self, cache: Optional[DataCache] = None):
        """
        Initialize Gaia loader.
        
        Args:
            cache: Cache instance (optional)
        """
        self.base_url = GAIA_TAP_URL
        self.config = GAIA_CONFIG
        self.cache = cache or DataCache()
        
        # Setup session with default headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.config['user_agent'],
            'Accept': 'text/csv',
            'Content-Type': 'application/x-www-form-urlencoded'
        })
        
        logger.debug("Gaia loader initialized")
    
    def _execute_query(self, query: str, format_type: str = 'csv') -> pd.DataFrame:
        """
        Execute a TAP query against Gaia archive.
        
        Args:
            query: ADQL query string
            format_type: Output format ('csv', 'votable')
            
        Returns:
            Query results as pandas DataFrame
            
        Raises:
            GaiaError: If query fails
        """
        # Check cache first
        cache_params = {'format': format_type}
        cached_result = self.cache.get(query, cache_params)
        if cached_result is not None:
            # Convert cached dict back to DataFrame
            return pd.DataFrame(cached_result)
        
        # Prepare request data
        data = {
            'REQUEST': 'doQuery',
            'LANG': 'ADQL',
            'FORMAT': format_type,
            'QUERY': query
        }
        
        # Execute query with retries
        last_exception = None
        for attempt in range(self.config['max_retries']):
            try:
                logger.debug(f"Gaia query attempt {attempt + 1}/{self.config['max_retries']}")
                
                response = self.session.post(
                    self.base_url,
                    data=data,
                    timeout=self.config['timeout']
                )
                
                response.raise_for_status()
                
                # Parse CSV response
                if format_type == 'csv':
                    df = pd.read_csv(StringIO(response.text))
                else:
                    # Handle other formats if needed
                    raise GaiaError(f"Unsupported format: {format_type}")
                
                # Cache successful result (convert DataFrame to dict for JSON serialization)
                self.cache.set(query, df.to_dict('records'), cache_params)
                
                logger.debug(f"Gaia query successful, {len(df)} rows returned")
                return df
                
            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < self.config['max_retries'] - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Gaia query failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Gaia query failed after {self.config['max_retries']} attempts")
        
        raise GaiaError(f"Failed to execute Gaia query: {last_exception}")
    
    def get_cluster_stars(self, 
                         ra: float, 
                         dec: float, 
                         radius: float = None,
                         search_params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Get stars in a cluster region from Gaia.
        
        Args:
            ra: Right ascension in degrees
            dec: Declination in degrees
            radius: Search radius in degrees (uses default if None)
            search_params: Custom search parameters (uses defaults if None)
            
        Returns:
            DataFrame with star data from Gaia
            
        Raises:
            GaiaError: If query fails
        """
        # Use provided parameters or defaults
        if search_params is None:
            search_params = {
                'min_parallax_over_error': DEFAULT_SEARCH_PARAMS.min_parallax_over_error,
                'max_ruwe': DEFAULT_SEARCH_PARAMS.max_ruwe,
                'min_mag': DEFAULT_SEARCH_PARAMS.min_magnitude,
                'max_mag': DEFAULT_SEARCH_PARAMS.max_magnitude
            }
        
        if radius is None:
            radius = DEFAULT_SEARCH_PARAMS.default_radius
        
        logger.info(f"Querying Gaia for stars around ({ra:.4f}, {dec:.4f}) "
                   f"with radius {radius:.3f}°")
        
        # Format the query
        query = GAIA_CLUSTER_STARS_QUERY.format(
            ra=ra,
            dec=dec,
            radius=radius,
            min_parallax_over_error=search_params['min_parallax_over_error'],
            max_ruwe=search_params['max_ruwe'],
            min_mag=search_params['min_mag'],
            max_mag=search_params['max_mag']
        )
        
        try:
            df = self._execute_query(query)
            
            logger.info(f"Retrieved {len(df)} stars from Gaia")
            
            # Add some computed columns
            if 'parallax' in df.columns and 'parallax_error' in df.columns:
                df['parallax_snr'] = df['parallax'] / df['parallax_error']
            
            if 'parallax' in df.columns:
                # Distance in parsecs (with error handling for negative parallax)
                df['distance_pc'] = 1000.0 / df['parallax'].where(df['parallax'] > 0)
            
            return df
            
        except Exception as e:
            raise GaiaError(f"Error retrieving cluster stars: {e}") from e
    
    def get_star_count_estimate(self, 
                               ra: float, 
                               dec: float, 
                               radius: float) -> int:
        """
        Get a quick estimate of how many stars would be returned.
        
        Args:
            ra: Right ascension in degrees
            dec: Declination in degrees
            radius: Search radius in degrees
            
        Returns:
            Estimated number of stars
        """
        query = f"""
        SELECT COUNT(*) as star_count
        FROM gaiaedr3.gaia_source
        WHERE CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra}, {dec}, {radius})
        ) = 1
        """
        
        try:
            df = self._execute_query(query)
            count = int(df.iloc[0]['star_count'])
            logger.debug(f"Estimated {count} stars in region")
            return count
        except Exception as e:
            logger.warning(f"Error estimating star count: {e}")
            return -1

# ============================================================================
# HIGH-LEVEL DATA LOADER
# ============================================================================

class ClusterDataLoader:
    """High-level interface for loading cluster data from multiple sources."""
    
    def __init__(self, cache_enabled: bool = True):
        """
        Initialize cluster data loader.
        
        Args:
            cache_enabled: Whether to enable caching
        """
        self.cache = DataCache() if cache_enabled else None
        self.simbad = SimbadLoader(self.cache)
        self.gaia = GaiaLoader(self.cache)
        
        logger.info("Cluster data loader initialized")
    
    def load_cluster_data(self, 
                         cluster_name: str,
                         radius: Optional[float] = None,
                         custom_params: Optional[Dict] = None) -> Tuple[Dict, pd.DataFrame]:
        """
        Load complete cluster data from SIMBAD and Gaia.
        
        Args:
            cluster_name: Name of the cluster
            radius: Custom search radius (degrees)
            custom_params: Custom search parameters
            
        Returns:
            Tuple of (cluster_info_dict, stars_dataframe)
            
        Raises:
            DataLoaderError: If data loading fails
        """
        logger.info(f"Loading complete data for cluster: {cluster_name}")
        
        try:
            # Step 1: Get cluster info from SIMBAD
            cluster_info = self.simbad.get_cluster_info(cluster_name)
            if cluster_info is None:
                raise DataLoaderError(f"Cluster '{cluster_name}' not found in SIMBAD")
            
            # Step 2: Extract coordinates
            ra = cluster_info['ra']
            dec = cluster_info['dec']
            
            if ra is None or dec is None:
                raise DataLoaderError(f"No coordinates available for cluster '{cluster_name}'")
            
            # Step 3: Determine search radius
            if radius is None:
                # Try to estimate based on object type
                obj_type = cluster_info.get('object_type', 'unknown').lower()
                if 'open' in obj_type or 'cluster' in obj_type:
                    radius = DEFAULT_SEARCH_PARAMS.default_radius
                elif 'globular' in obj_type:
                    radius = 0.2
                elif 'association' in obj_type:
                    radius = 1.0
                else:
                    radius = DEFAULT_SEARCH_PARAMS.default_radius
            
            logger.info(f"Using search radius: {radius:.3f}° for {obj_type}")
            
            # Step 4: Get stars from Gaia
            stars_df = self.gaia.get_cluster_stars(ra, dec, radius, custom_params)
            
            # Step 5: Add metadata to cluster info
            cluster_info['search_radius'] = radius
            cluster_info['stars_found'] = len(stars_df)
            cluster_info['query_timestamp'] = datetime.now().isoformat()
            
            logger.info(f"Successfully loaded data for {cluster_name}: "
                       f"{len(stars_df)} stars in {radius:.3f}° radius")
            
            return cluster_info, stars_df
            
        except Exception as e:
            logger.error(f"Failed to load cluster data: {e}")
            raise DataLoaderError(f"Failed to load data for '{cluster_name}': {e}") from e
    
    def clear_cache(self) -> int:
        """Clear all cached data."""
        if self.cache:
            return self.cache.clear()
        return 0

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_cluster_lookup(cluster_name: str) -> Optional[Dict]:
    """
    Quick lookup of cluster basic info from SIMBAD.
    
    Args:
        cluster_name: Name of cluster to look up
        
    Returns:
        Basic cluster information or None if not found
    """
    loader = SimbadLoader()
    try:
        return loader.get_cluster_info(cluster_name)
    except Exception as e:
        logger.error(f"Quick lookup failed for {cluster_name}: {e}")
        return None

def estimate_query_size(ra: float, dec: float, radius: float) -> int:
    """
    Estimate how many stars a Gaia query would return.
    
    Args:
        ra: Right ascension in degrees
        dec: Declination in degrees  
        radius: Search radius in degrees
        
    Returns:
        Estimated number of stars
    """
    loader = GaiaLoader()
    try:
        return loader.get_star_count_estimate(ra, dec, radius)
    except Exception:
        return -1

if __name__ == "__main__":
    # Basic testing
    print("🔍 Testing Oculus v1.0 Data Loader")
    
    # Test SIMBAD lookup
    print("\n📡 Testing SIMBAD connection...")
    pleiades_info = quick_cluster_lookup("Pleiades")
    if pleiades_info:
        print(f"✅ Found Pleiades at ({pleiades_info['ra']:.4f}, {pleiades_info['dec']:.4f})")
    else:
        print("❌ Failed to find Pleiades")
    
    # Test Gaia estimate
    if pleiades_info:
        print("\n🌟 Testing Gaia connection...")
        star_count = estimate_query_size(
            pleiades_info['ra'], 
            pleiades_info['dec'], 
            0.5
        )
        if star_count > 0:
            print(f"✅ Estimated {star_count} stars in Pleiades region")
        else:
            print("❌ Failed to estimate star count")
    
    print("\n🎯 Data loader testing complete!")