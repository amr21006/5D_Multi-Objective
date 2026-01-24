"""
5D Fuzzy Weather-Sensitive Multi-Objective Project Scheduling Framework
========================================================================
A comprehensive framework for infrastructure project optimization using
NSGA-III algorithm with OPA-TOPSIS MCDM ranking and weather-sensitive scheduling.

Objectives: Duration, Cost, Quality, Resources, Weather Risk

Author: Research Implementation
Date: January 2026
"""

# =============================================================================
# PART 1: IMPORTS AND CONFIGURATION
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import warnings
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import time
from datetime import datetime

# Progress bar
try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: Install tqdm for progress bars (pip install tqdm)")

# Pymoo imports
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.algorithms.moo.rnsga3 import RNSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.indicators.gd import GD
from pymoo.termination import get_termination

# Statistical analysis
from scipy import stats
from scipy.spatial.distance import cdist

# Parallel processing
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("Warning: joblib not available. Parallel execution disabled.")

warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'pop_size': 100,
    'n_gen': 200,
    'n_runs': 30,
    'n_jobs': -1,  # Use all available cores
    'seed_base': 42,
    'alpha_congestion': 0.3,  # Safety congestion sensitivity
    'output_dir': 'results',
    'fig_dpi': 300,
    'random_state': 42
}

# =============================================================================
# PART 2: FUZZY LOGIC AND WEATHER SYSTEMS
# =============================================================================

@dataclass
class FuzzyNumber:
    """Triangular fuzzy number (low, mid, high) for uncertainty modeling."""
    low: float   # pessimistic (worst case)
    mid: float   # most likely
    high: float  # optimistic (best case)
    
    def defuzzify(self, method: str = 'centroid') -> float:
        """Convert fuzzy to crisp value using defuzzification method."""
        if method == 'centroid':
            return (self.low + self.mid + self.high) / 3
        elif method == 'bisector':
            return (self.low + 2 * self.mid + self.high) / 4
        elif method == 'mom':  # Mean of Maximum
            return self.mid
        return self.mid
    
    def alpha_cut(self, alpha: float) -> Tuple[float, float]:
        """Return interval at confidence level α ∈ [0, 1]."""
        left = self.low + alpha * (self.mid - self.low)
        right = self.high - alpha * (self.high - self.mid)
        return (left, right)
    
    def __add__(self, other: 'FuzzyNumber') -> 'FuzzyNumber':
        """Fuzzy addition."""
        return FuzzyNumber(
            self.low + other.low,
            self.mid + other.mid,
            self.high + other.high
        )
    
    def __mul__(self, scalar: float) -> 'FuzzyNumber':
        """Scalar multiplication."""
        return FuzzyNumber(
            self.low * scalar,
            self.mid * scalar,
            self.high * scalar
        )
    
    def __rmul__(self, scalar: float) -> 'FuzzyNumber':
        return self.__mul__(scalar)
    
    def expected_value(self) -> float:
        """PERT-like expected value: (low + 4*mid + high) / 6."""
        return (self.low + 4 * self.mid + self.high) / 6
    
    def variance(self) -> float:
        """PERT variance estimate: ((high - low) / 6)^2."""
        return ((self.high - self.low) / 6) ** 2


@dataclass
class MonthlyWeather:
    """Monthly weather characteristics for construction scheduling."""
    month: int                    # 1-12
    rain_probability: float       # 0-1, probability of rain day
    temp_productivity: float      # 0-1, temperature-based productivity factor
    wind_severity: float          # 0-1, wind impact on crane/high work
    daylight_hours: float         # available work hours per day
    extreme_weather_prob: float   # probability of work stoppage


class WeatherProfile:
    """
    Annual weather profile with seasonal patterns for construction scheduling.
    Models impact of weather on outdoor construction activities.
    """
    
    def __init__(self, region: str = 'temperate'):
        """Initialize weather profile for a given region."""
        self.region = region
        self.monthly_data = self._generate_profile(region)
    
    def _generate_profile(self, region: str) -> List[MonthlyWeather]:
        """Generate 12-month weather profile based on region."""
        profiles = {
            'temperate': [
                # Winter (Jan-Feb): Cold, some rain, short days
                MonthlyWeather(1, 0.35, 0.65, 0.40, 8.0, 0.15),
                MonthlyWeather(2, 0.30, 0.70, 0.35, 9.0, 0.12),
                # Spring (Mar-May): Improving conditions
                MonthlyWeather(3, 0.35, 0.80, 0.30, 10.5, 0.08),
                MonthlyWeather(4, 0.40, 0.85, 0.25, 12.0, 0.06),
                MonthlyWeather(5, 0.30, 0.92, 0.20, 13.5, 0.04),
                # Summer (Jun-Aug): Best conditions
                MonthlyWeather(6, 0.20, 0.95, 0.15, 14.5, 0.02),
                MonthlyWeather(7, 0.15, 0.98, 0.12, 14.5, 0.02),
                MonthlyWeather(8, 0.18, 0.96, 0.15, 13.5, 0.03),
                # Autumn (Sep-Nov): Declining conditions
                MonthlyWeather(9, 0.25, 0.90, 0.22, 12.0, 0.05),
                MonthlyWeather(10, 0.35, 0.82, 0.30, 10.5, 0.08),
                MonthlyWeather(11, 0.40, 0.72, 0.38, 9.0, 0.12),
                # Winter (Dec)
                MonthlyWeather(12, 0.38, 0.62, 0.42, 8.0, 0.18),
            ],
            'mediterranean': [
                MonthlyWeather(1, 0.45, 0.75, 0.25, 9.0, 0.10),
                MonthlyWeather(2, 0.40, 0.78, 0.22, 10.0, 0.08),
                MonthlyWeather(3, 0.30, 0.85, 0.18, 11.5, 0.05),
                MonthlyWeather(4, 0.20, 0.90, 0.15, 13.0, 0.03),
                MonthlyWeather(5, 0.10, 0.92, 0.12, 14.0, 0.02),
                MonthlyWeather(6, 0.05, 0.88, 0.10, 15.0, 0.02),  # Hot
                MonthlyWeather(7, 0.02, 0.82, 0.08, 15.0, 0.03),  # Very hot
                MonthlyWeather(8, 0.03, 0.80, 0.10, 14.5, 0.04),  # Very hot
                MonthlyWeather(9, 0.12, 0.88, 0.12, 13.0, 0.03),
                MonthlyWeather(10, 0.25, 0.85, 0.18, 11.5, 0.05),
                MonthlyWeather(11, 0.38, 0.78, 0.22, 10.0, 0.08),
                MonthlyWeather(12, 0.42, 0.72, 0.28, 9.0, 0.12),
            ],
            'desert': [
                MonthlyWeather(1, 0.05, 0.90, 0.20, 10.0, 0.05),
                MonthlyWeather(2, 0.05, 0.92, 0.18, 11.0, 0.04),
                MonthlyWeather(3, 0.03, 0.88, 0.22, 12.0, 0.06),
                MonthlyWeather(4, 0.02, 0.80, 0.28, 13.0, 0.08),
                MonthlyWeather(5, 0.01, 0.65, 0.35, 14.0, 0.15),  # Heat
                MonthlyWeather(6, 0.01, 0.50, 0.40, 14.5, 0.25),  # Extreme heat
                MonthlyWeather(7, 0.02, 0.45, 0.45, 14.5, 0.30),  # Extreme heat
                MonthlyWeather(8, 0.02, 0.48, 0.42, 14.0, 0.28),  # Extreme heat
                MonthlyWeather(9, 0.02, 0.70, 0.32, 13.0, 0.12),
                MonthlyWeather(10, 0.03, 0.85, 0.25, 12.0, 0.06),
                MonthlyWeather(11, 0.04, 0.90, 0.20, 11.0, 0.04),
                MonthlyWeather(12, 0.05, 0.88, 0.22, 10.0, 0.05),
            ],
        }
        return profiles.get(region, profiles['temperate'])
    
    def get_productivity_factor(self, day: int, weather_sensitivity: float, 
                                 start_month: int = 1) -> float:
        """
        Calculate expected productivity for a given day considering weather.
        
        Args:
            day: Day number from project start (0-indexed)
            weather_sensitivity: Activity's sensitivity to weather (0-1)
            start_month: Project start month (1-12)
            
        Returns:
            Productivity factor (0-1), where 1 = full productivity
        """
        # Calculate which month we're in
        month_idx = ((start_month - 1) + (day // 30)) % 12
        data = self.monthly_data[month_idx]
        
        # Base productivity from temperature
        base_factor = data.temp_productivity
        
        # Rain impact: reduces productivity proportionally
        rain_impact = data.rain_probability * 0.4  # Rain can reduce up to 40%
        
        # Wind impact for sensitive activities
        wind_impact = data.wind_severity * weather_sensitivity * 0.3
        
        # Combined factor
        productivity = base_factor * (1 - rain_impact) * (1 - wind_impact)
        
        # Apply activity's weather sensitivity
        final_factor = 1.0 - weather_sensitivity * (1.0 - productivity)
        
        return max(0.3, min(1.0, final_factor))  # Clamp between 30% and 100%
    
    def get_delay_probability(self, day: int, start_month: int = 1) -> float:
        """Get probability of complete work stoppage for a given day."""
        month_idx = ((start_month - 1) + (day // 30)) % 12
        return self.monthly_data[month_idx].extreme_weather_prob
    
    def get_seasonal_risk_index(self, start_day: int, duration: int, 
                                 weather_sensitivity: float,
                                 start_month: int = 1) -> float:
        """
        Calculate weather risk index for an activity.
        
        Args:
            start_day: Activity start day
            duration: Activity duration in days
            weather_sensitivity: Activity's weather sensitivity (0-1)
            start_month: Project start month
            
        Returns:
            Weather risk index (higher = more risk)
        """
        total_risk = 0.0
        for d in range(start_day, start_day + duration):
            month_idx = ((start_month - 1) + (d // 30)) % 12
            data = self.monthly_data[month_idx]
            
            # Risk components
            rain_risk = data.rain_probability * weather_sensitivity
            wind_risk = data.wind_severity * weather_sensitivity * 0.5
            extreme_risk = data.extreme_weather_prob * 2.0
            
            daily_risk = rain_risk + wind_risk + extreme_risk
            total_risk += daily_risk
        
        return total_risk


# =============================================================================
# PART 2B: PROJECT DATA STRUCTURES
# =============================================================================

@dataclass
class Method:
    """Construction method for an activity with fuzzy parameters and weather sensitivity."""
    id: int
    duration: int                    # crisp duration (days)
    cost: float                      # crisp direct cost ($)
    quality: float                   # 0-1 scale
    labor: int                       # workers required
    equipment: int                   # equipment units
    environmental: float             # impact index
    social: float                    # social cost index
    safety_risk: float               # base risk factor
    # Fuzzy parameters for uncertainty modeling
    fuzzy_duration: Optional[FuzzyNumber] = None   # (optimistic, likely, pessimistic)
    fuzzy_cost: Optional[FuzzyNumber] = None       # (low, likely, high)
    # Weather sensitivity
    weather_sensitivity: float = 0.3               # 0-1, how affected by weather

@dataclass
class Activity:
    """Project activity with multiple execution methods."""
    id: int
    name: str
    methods: List[Method]
    predecessors: List[int] = field(default_factory=list)
    weight: float = 1.0  # importance weight for quality calculation

@dataclass
class Project:
    """Construction project with activities and relationships."""
    name: str
    project_type: str
    activities: List[Activity]
    daily_indirect_cost: float = 5000.0
    max_labor: int = 50
    max_equipment: int = 20
    
    @property
    def n_activities(self) -> int:
        return len(self.activities)
    
    @property
    def search_space_size(self) -> int:
        size = 1
        for act in self.activities:
            size *= len(act.methods)
        return size


def create_highway_project() -> Project:
    """Create Highway Renovation project (8 activities)."""
    activities = [
        Activity(0, "Site Preparation", [
            Method(0, 5, 15000, 0.75, 8, 2, 0.4, 0.3, 0.25),
            Method(1, 4, 18000, 0.80, 10, 3, 0.5, 0.4, 0.30),
            Method(2, 3, 22000, 0.85, 12, 4, 0.6, 0.5, 0.35),
        ], [], 1.0),
        Activity(1, "Earthwork", [
            Method(0, 8, 45000, 0.70, 15, 5, 0.7, 0.5, 0.40),
            Method(1, 6, 55000, 0.78, 18, 7, 0.8, 0.6, 0.45),
            Method(2, 5, 68000, 0.85, 22, 9, 0.9, 0.7, 0.50),
        ], [0], 1.2),
        Activity(2, "Drainage Installation", [
            Method(0, 6, 35000, 0.72, 10, 3, 0.5, 0.4, 0.30),
            Method(1, 5, 42000, 0.80, 12, 4, 0.6, 0.5, 0.35),
            Method(2, 4, 52000, 0.88, 15, 5, 0.7, 0.6, 0.40),
        ], [0], 1.1),
        Activity(3, "Base Course", [
            Method(0, 7, 60000, 0.73, 12, 6, 0.6, 0.5, 0.35),
            Method(1, 5, 72000, 0.82, 15, 8, 0.7, 0.6, 0.40),
            Method(2, 4, 88000, 0.90, 18, 10, 0.8, 0.7, 0.45),
        ], [1, 2], 1.3),
        Activity(4, "Asphalt Paving", [
            Method(0, 6, 80000, 0.75, 14, 8, 0.8, 0.7, 0.45),
            Method(1, 4, 95000, 0.85, 18, 10, 0.9, 0.8, 0.50),
            Method(2, 3, 115000, 0.92, 22, 12, 1.0, 0.9, 0.55),
        ], [3], 1.5),
        Activity(5, "Road Markings", [
            Method(0, 4, 20000, 0.78, 6, 2, 0.3, 0.3, 0.20),
            Method(1, 3, 25000, 0.85, 8, 3, 0.4, 0.4, 0.25),
            Method(2, 2, 32000, 0.92, 10, 4, 0.5, 0.5, 0.30),
        ], [4], 0.9),
        Activity(6, "Signage Installation", [
            Method(0, 3, 18000, 0.80, 5, 2, 0.2, 0.2, 0.20),
            Method(1, 2, 23000, 0.88, 7, 3, 0.3, 0.3, 0.25),
            Method(2, 2, 28000, 0.95, 9, 4, 0.4, 0.4, 0.30),
        ], [4], 0.8),
        Activity(7, "Final Inspection", [
            Method(0, 3, 12000, 0.82, 4, 1, 0.1, 0.1, 0.15),
            Method(1, 2, 15000, 0.90, 6, 2, 0.2, 0.2, 0.20),
            Method(2, 2, 19000, 0.96, 8, 3, 0.3, 0.3, 0.25),
        ], [5, 6], 1.0),
    ]
    return Project("Highway Renovation", "Highway", activities, 8000, 60, 25)


def create_bridge_project() -> Project:
    """Create Bridge Construction project (10 activities)."""
    activities = [
        Activity(0, "Site Survey", [
            Method(0, 4, 20000, 0.78, 6, 2, 0.3, 0.2, 0.20),
            Method(1, 3, 26000, 0.85, 8, 3, 0.4, 0.3, 0.25),
            Method(2, 2, 34000, 0.92, 10, 4, 0.5, 0.4, 0.30),
        ], [], 1.0),
        Activity(1, "Foundation Excavation", [
            Method(0, 10, 85000, 0.70, 20, 8, 0.8, 0.6, 0.50),
            Method(1, 8, 105000, 0.78, 25, 10, 0.9, 0.7, 0.55),
            Method(2, 6, 130000, 0.85, 30, 12, 1.0, 0.8, 0.60),
        ], [0], 1.3),
        Activity(2, "Pile Driving", [
            Method(0, 12, 150000, 0.72, 18, 10, 0.9, 0.7, 0.60),
            Method(1, 9, 185000, 0.80, 22, 12, 1.0, 0.8, 0.65),
            Method(2, 7, 225000, 0.88, 28, 15, 1.1, 0.9, 0.70),
        ], [1], 1.5),
        Activity(3, "Abutment Construction", [
            Method(0, 14, 120000, 0.74, 16, 6, 0.7, 0.5, 0.45),
            Method(1, 11, 148000, 0.82, 20, 8, 0.8, 0.6, 0.50),
            Method(2, 8, 180000, 0.90, 25, 10, 0.9, 0.7, 0.55),
        ], [2], 1.4),
        Activity(4, "Pier Construction", [
            Method(0, 16, 200000, 0.73, 22, 10, 0.8, 0.6, 0.55),
            Method(1, 12, 245000, 0.82, 28, 12, 0.9, 0.7, 0.60),
            Method(2, 9, 300000, 0.90, 35, 15, 1.0, 0.8, 0.65),
        ], [2], 1.5),
        Activity(5, "Beam Installation", [
            Method(0, 8, 180000, 0.75, 15, 12, 0.7, 0.5, 0.55),
            Method(1, 6, 220000, 0.84, 18, 15, 0.8, 0.6, 0.60),
            Method(2, 4, 270000, 0.92, 22, 18, 0.9, 0.7, 0.65),
        ], [3, 4], 1.6),
        Activity(6, "Deck Construction", [
            Method(0, 10, 140000, 0.76, 18, 8, 0.6, 0.5, 0.45),
            Method(1, 8, 172000, 0.84, 22, 10, 0.7, 0.6, 0.50),
            Method(2, 6, 210000, 0.92, 28, 12, 0.8, 0.7, 0.55),
        ], [5], 1.4),
        Activity(7, "Railing Installation", [
            Method(0, 5, 45000, 0.80, 8, 3, 0.3, 0.3, 0.30),
            Method(1, 4, 56000, 0.87, 10, 4, 0.4, 0.4, 0.35),
            Method(2, 3, 70000, 0.94, 12, 5, 0.5, 0.5, 0.40),
        ], [6], 1.0),
        Activity(8, "Approach Roads", [
            Method(0, 7, 65000, 0.74, 12, 5, 0.5, 0.5, 0.35),
            Method(1, 5, 80000, 0.82, 15, 7, 0.6, 0.6, 0.40),
            Method(2, 4, 98000, 0.90, 18, 9, 0.7, 0.7, 0.45),
        ], [6], 1.1),
        Activity(9, "Final Testing", [
            Method(0, 4, 25000, 0.82, 6, 2, 0.2, 0.2, 0.20),
            Method(1, 3, 32000, 0.90, 8, 3, 0.3, 0.3, 0.25),
            Method(2, 2, 40000, 0.96, 10, 4, 0.4, 0.4, 0.30),
        ], [7, 8], 1.0),
    ]
    return Project("Bridge Construction", "Bridge", activities, 12000, 80, 35)


def create_metro_project() -> Project:
    """
    Create Metro Station project (12 activities) with fuzzy parameters and weather sensitivity.
    
    Each method includes:
    - fuzzy_duration: Triangular fuzzy number (optimistic, most_likely, pessimistic)
    - fuzzy_cost: Triangular fuzzy number (low, likely, high)
    - weather_sensitivity: 0-1 scale (0=indoor/protected, 1=fully outdoor exposed)
    """
    activities = [
        # Activity 0: Site Clearance (OUTDOOR - High weather sensitivity)
        Activity(0, "Site Clearance", [
            Method(0, 5, 30000, 0.76, 10, 4, 0.5, 0.4, 0.30,
                   FuzzyNumber(4, 5, 7), FuzzyNumber(27000, 30000, 36000), 0.75),
            Method(1, 4, 38000, 0.84, 12, 5, 0.6, 0.5, 0.35,
                   FuzzyNumber(3, 4, 6), FuzzyNumber(34000, 38000, 46000), 0.70),
            Method(2, 3, 48000, 0.91, 15, 6, 0.7, 0.6, 0.40,
                   FuzzyNumber(2, 3, 5), FuzzyNumber(43000, 48000, 58000), 0.65),
        ], [], 1.0),
        
        # Activity 1: Deep Excavation (OUTDOOR - Very high weather sensitivity)
        Activity(1, "Deep Excavation", [
            Method(0, 18, 250000, 0.68, 35, 15, 1.0, 0.8, 0.65,
                   FuzzyNumber(15, 18, 25), FuzzyNumber(225000, 250000, 300000), 0.85),
            Method(1, 14, 310000, 0.76, 42, 18, 1.1, 0.9, 0.70,
                   FuzzyNumber(12, 14, 20), FuzzyNumber(280000, 310000, 370000), 0.80),
            Method(2, 10, 380000, 0.84, 50, 22, 1.2, 1.0, 0.75,
                   FuzzyNumber(8, 10, 15), FuzzyNumber(340000, 380000, 460000), 0.75),
        ], [0], 1.5),
        
        # Activity 2: Retaining Wall (OUTDOOR - High weather sensitivity)
        Activity(2, "Retaining Wall", [
            Method(0, 12, 180000, 0.72, 25, 10, 0.8, 0.6, 0.50,
                   FuzzyNumber(10, 12, 16), FuzzyNumber(162000, 180000, 216000), 0.70),
            Method(1, 9, 220000, 0.80, 30, 12, 0.9, 0.7, 0.55,
                   FuzzyNumber(7, 9, 13), FuzzyNumber(198000, 220000, 264000), 0.65),
            Method(2, 7, 270000, 0.88, 38, 15, 1.0, 0.8, 0.60,
                   FuzzyNumber(5, 7, 10), FuzzyNumber(243000, 270000, 324000), 0.60),
        ], [1], 1.3),
        
        # Activity 3: Foundation Slab (PARTIALLY OUTDOOR - Medium weather sensitivity)
        Activity(3, "Foundation Slab", [
            Method(0, 10, 160000, 0.74, 22, 8, 0.7, 0.5, 0.45,
                   FuzzyNumber(8, 10, 14), FuzzyNumber(144000, 160000, 192000), 0.55),
            Method(1, 8, 195000, 0.82, 28, 10, 0.8, 0.6, 0.50,
                   FuzzyNumber(6, 8, 11), FuzzyNumber(175000, 195000, 234000), 0.50),
            Method(2, 6, 240000, 0.90, 35, 12, 0.9, 0.7, 0.55,
                   FuzzyNumber(5, 6, 9), FuzzyNumber(216000, 240000, 288000), 0.45),
        ], [2], 1.4),
        
        # Activity 4: Platform Structure (PARTIALLY PROTECTED - Medium weather sensitivity)
        Activity(4, "Platform Structure", [
            Method(0, 14, 220000, 0.73, 28, 12, 0.8, 0.6, 0.50,
                   FuzzyNumber(11, 14, 19), FuzzyNumber(198000, 220000, 264000), 0.50),
            Method(1, 11, 270000, 0.81, 35, 15, 0.9, 0.7, 0.55,
                   FuzzyNumber(9, 11, 15), FuzzyNumber(243000, 270000, 324000), 0.45),
            Method(2, 8, 330000, 0.89, 42, 18, 1.0, 0.8, 0.60,
                   FuzzyNumber(6, 8, 12), FuzzyNumber(297000, 330000, 396000), 0.40),
        ], [3], 1.5),
        
        # Activity 5: Tunnel Connection (UNDERGROUND - Low weather sensitivity)
        Activity(5, "Tunnel Connection", [
            Method(0, 20, 350000, 0.70, 40, 18, 1.0, 0.8, 0.70,
                   FuzzyNumber(16, 20, 28), FuzzyNumber(315000, 350000, 420000), 0.15),
            Method(1, 15, 430000, 0.78, 48, 22, 1.1, 0.9, 0.75,
                   FuzzyNumber(12, 15, 21), FuzzyNumber(387000, 430000, 516000), 0.12),
            Method(2, 12, 520000, 0.86, 58, 26, 1.2, 1.0, 0.80,
                   FuzzyNumber(10, 12, 17), FuzzyNumber(468000, 520000, 624000), 0.10),
        ], [3], 1.6),
        
        # Activity 6: Roof Structure (OUTDOOR - High weather sensitivity)
        Activity(6, "Roof Structure", [
            Method(0, 12, 200000, 0.75, 24, 14, 0.7, 0.5, 0.45,
                   FuzzyNumber(10, 12, 17), FuzzyNumber(180000, 200000, 240000), 0.80),
            Method(1, 9, 245000, 0.83, 30, 17, 0.8, 0.6, 0.50,
                   FuzzyNumber(7, 9, 13), FuzzyNumber(220000, 245000, 294000), 0.75),
            Method(2, 7, 300000, 0.91, 38, 20, 0.9, 0.7, 0.55,
                   FuzzyNumber(5, 7, 10), FuzzyNumber(270000, 300000, 360000), 0.70),
        ], [4, 5], 1.4),
        
        # Activity 7: MEP Systems (INDOOR - Low weather sensitivity)
        Activity(7, "MEP Systems", [
            Method(0, 15, 280000, 0.72, 30, 10, 0.6, 0.4, 0.40,
                   FuzzyNumber(12, 15, 20), FuzzyNumber(252000, 280000, 336000), 0.20),
            Method(1, 12, 340000, 0.80, 38, 12, 0.7, 0.5, 0.45,
                   FuzzyNumber(10, 12, 16), FuzzyNumber(306000, 340000, 408000), 0.15),
            Method(2, 9, 420000, 0.88, 45, 15, 0.8, 0.6, 0.50,
                   FuzzyNumber(7, 9, 12), FuzzyNumber(378000, 420000, 504000), 0.12),
        ], [6], 1.3),
        
        # Activity 8: Finishing Works (INDOOR - Very low weather sensitivity)
        Activity(8, "Finishing Works", [
            Method(0, 10, 150000, 0.78, 25, 6, 0.4, 0.4, 0.30,
                   FuzzyNumber(8, 10, 14), FuzzyNumber(135000, 150000, 180000), 0.15),
            Method(1, 8, 185000, 0.85, 32, 8, 0.5, 0.5, 0.35,
                   FuzzyNumber(6, 8, 11), FuzzyNumber(166000, 185000, 222000), 0.12),
            Method(2, 6, 225000, 0.92, 40, 10, 0.6, 0.6, 0.40,
                   FuzzyNumber(5, 6, 9), FuzzyNumber(202000, 225000, 270000), 0.10),
        ], [7], 1.2),
        
        # Activity 9: Escalator Installation (INDOOR - Very low weather sensitivity)
        Activity(9, "Escalator Installation", [
            Method(0, 8, 180000, 0.80, 15, 8, 0.4, 0.3, 0.40,
                   FuzzyNumber(6, 8, 11), FuzzyNumber(162000, 180000, 216000), 0.10),
            Method(1, 6, 220000, 0.87, 18, 10, 0.5, 0.4, 0.45,
                   FuzzyNumber(5, 6, 9), FuzzyNumber(198000, 220000, 264000), 0.08),
            Method(2, 5, 270000, 0.94, 22, 12, 0.6, 0.5, 0.50,
                   FuzzyNumber(4, 5, 7), FuzzyNumber(243000, 270000, 324000), 0.06),
        ], [8], 1.1),
        
        # Activity 10: Safety Systems (INDOOR - Very low weather sensitivity)
        Activity(10, "Safety Systems", [
            Method(0, 6, 120000, 0.82, 12, 5, 0.3, 0.2, 0.25,
                   FuzzyNumber(5, 6, 8), FuzzyNumber(108000, 120000, 144000), 0.12),
            Method(1, 5, 148000, 0.89, 15, 6, 0.4, 0.3, 0.30,
                   FuzzyNumber(4, 5, 7), FuzzyNumber(133000, 148000, 178000), 0.10),
            Method(2, 4, 180000, 0.95, 18, 8, 0.5, 0.4, 0.35,
                   FuzzyNumber(3, 4, 6), FuzzyNumber(162000, 180000, 216000), 0.08),
        ], [8], 1.2),
        
        # Activity 11: Commissioning (INDOOR - Minimal weather sensitivity)
        Activity(11, "Commissioning", [
            Method(0, 5, 50000, 0.84, 10, 4, 0.2, 0.2, 0.20,
                   FuzzyNumber(4, 5, 7), FuzzyNumber(45000, 50000, 60000), 0.05),
            Method(1, 4, 62000, 0.91, 12, 5, 0.3, 0.3, 0.25,
                   FuzzyNumber(3, 4, 6), FuzzyNumber(56000, 62000, 74000), 0.04),
            Method(2, 3, 76000, 0.97, 15, 6, 0.4, 0.4, 0.30,
                   FuzzyNumber(2, 3, 5), FuzzyNumber(68000, 76000, 91000), 0.03),
        ], [9, 10], 1.0),
    ]
    return Project("Metro Station", "Metro", activities, 50000, 100, 45)


def create_highway_interchange_project() -> Project:
    """
    Create Highway Interchange project (10 activities) - Infrastructure.
    
    Complex network structure with parallel branches and multiple predecessors:
    
                    ┌─→ [2] Northbound Ramp ─→┐
    [0] Site ─→ [1] ─┤                        ├─→ [6] Bridge Deck ─→ [8] ─→ [9]
                    └─→ [3] Southbound Ramp ─→┘       ↑
                              ↓                       │
                          [4] Drainage ─→ [5] Retaining ─┘
                              ↓
                          [7] Signage ─────────────────────→
    
    All activities are OUTDOOR with high weather sensitivity.
    Weather profile: Temperate (seasonal variation)
    """
    activities = [
        # Activity 0: Site Preparation & Clearing (OUTDOOR)
        Activity(0, "Site Preparation", [
            Method(0, 12, 85000, 0.74, 18, 8, 0.6, 0.5, 0.35,
                   FuzzyNumber(10, 12, 16), FuzzyNumber(76000, 85000, 102000), 0.85),
            Method(1, 9, 105000, 0.82, 22, 10, 0.7, 0.6, 0.40,
                   FuzzyNumber(7, 9, 12), FuzzyNumber(94000, 105000, 126000), 0.80),
            Method(2, 7, 130000, 0.89, 28, 12, 0.8, 0.7, 0.45,
                   FuzzyNumber(5, 7, 10), FuzzyNumber(117000, 130000, 156000), 0.75),
        ], [], 1.0),  # No predecessors - start activity
        
        # Activity 1: Earthwork & Grading (OUTDOOR - Very High sensitivity)
        Activity(1, "Earthwork & Grading", [
            Method(0, 20, 320000, 0.70, 35, 20, 0.9, 0.7, 0.55,
                   FuzzyNumber(16, 20, 28), FuzzyNumber(288000, 320000, 384000), 0.90),
            Method(1, 15, 400000, 0.78, 45, 25, 1.0, 0.8, 0.60,
                   FuzzyNumber(12, 15, 21), FuzzyNumber(360000, 400000, 480000), 0.85),
            Method(2, 12, 500000, 0.86, 55, 30, 1.1, 0.9, 0.65,
                   FuzzyNumber(9, 12, 17), FuzzyNumber(450000, 500000, 600000), 0.80),
        ], [0], 1.5),  # Depends on Site Preparation
        
        # Activity 2: Northbound Ramp Construction (OUTDOOR - Parallel Branch 1)
        Activity(2, "Northbound Ramp", [
            Method(0, 18, 450000, 0.73, 30, 18, 0.7, 0.5, 0.50,
                   FuzzyNumber(14, 18, 25), FuzzyNumber(405000, 450000, 540000), 0.80),
            Method(1, 14, 550000, 0.81, 38, 22, 0.8, 0.6, 0.55,
                   FuzzyNumber(11, 14, 19), FuzzyNumber(495000, 550000, 660000), 0.75),
            Method(2, 11, 680000, 0.88, 48, 28, 0.9, 0.7, 0.60,
                   FuzzyNumber(8, 11, 15), FuzzyNumber(612000, 680000, 816000), 0.70),
        ], [1], 1.4),  # Depends on Earthwork - PARALLEL with Activity 3
        
        # Activity 3: Southbound Ramp Construction (OUTDOOR - Parallel Branch 2)
        Activity(3, "Southbound Ramp", [
            Method(0, 18, 450000, 0.73, 30, 18, 0.7, 0.5, 0.50,
                   FuzzyNumber(14, 18, 25), FuzzyNumber(405000, 450000, 540000), 0.80),
            Method(1, 14, 550000, 0.81, 38, 22, 0.8, 0.6, 0.55,
                   FuzzyNumber(11, 14, 19), FuzzyNumber(495000, 550000, 660000), 0.75),
            Method(2, 11, 680000, 0.88, 48, 28, 0.9, 0.7, 0.60,
                   FuzzyNumber(8, 11, 15), FuzzyNumber(612000, 680000, 816000), 0.70),
        ], [1], 1.4),  # Depends on Earthwork - PARALLEL with Activity 2
        
        # Activity 4: Drainage System (OUTDOOR - High sensitivity)
        Activity(4, "Drainage System", [
            Method(0, 15, 180000, 0.75, 20, 10, 0.5, 0.4, 0.40,
                   FuzzyNumber(12, 15, 21), FuzzyNumber(162000, 180000, 216000), 0.75),
            Method(1, 12, 225000, 0.83, 25, 12, 0.6, 0.5, 0.45,
                   FuzzyNumber(9, 12, 17), FuzzyNumber(202000, 225000, 270000), 0.70),
            Method(2, 9, 280000, 0.90, 32, 15, 0.7, 0.6, 0.50,
                   FuzzyNumber(7, 9, 13), FuzzyNumber(252000, 280000, 336000), 0.65),
        ], [3], 1.3),  # Depends on Southbound Ramp
        
        # Activity 5: Retaining Walls (OUTDOOR - Medium-High sensitivity)
        Activity(5, "Retaining Walls", [
            Method(0, 16, 280000, 0.76, 25, 15, 0.6, 0.5, 0.45,
                   FuzzyNumber(13, 16, 22), FuzzyNumber(252000, 280000, 336000), 0.70),
            Method(1, 12, 350000, 0.84, 32, 18, 0.7, 0.6, 0.50,
                   FuzzyNumber(9, 12, 17), FuzzyNumber(315000, 350000, 420000), 0.65),
            Method(2, 10, 430000, 0.91, 40, 22, 0.8, 0.7, 0.55,
                   FuzzyNumber(7, 10, 14), FuzzyNumber(387000, 430000, 516000), 0.60),
        ], [4], 1.4),  # Depends on Drainage
        
        # Activity 6: Bridge Deck Construction (OUTDOOR - Very High sensitivity)
        # CONVERGENCE POINT: Requires both ramps AND retaining walls
        Activity(6, "Bridge Deck", [
            Method(0, 25, 650000, 0.72, 40, 25, 0.8, 0.6, 0.55,
                   FuzzyNumber(20, 25, 35), FuzzyNumber(585000, 650000, 780000), 0.85),
            Method(1, 20, 800000, 0.80, 50, 30, 0.9, 0.7, 0.60,
                   FuzzyNumber(15, 20, 28), FuzzyNumber(720000, 800000, 960000), 0.80),
            Method(2, 15, 980000, 0.88, 62, 38, 1.0, 0.8, 0.65,
                   FuzzyNumber(12, 15, 21), FuzzyNumber(882000, 980000, 1176000), 0.75),
        ], [2, 3, 5], 1.6),  # COMPLEX: Depends on BOTH ramps AND retaining walls
        
        # Activity 7: Signage & Markings (OUTDOOR - Medium sensitivity)
        Activity(7, "Signage & Markings", [
            Method(0, 8, 95000, 0.80, 12, 6, 0.3, 0.3, 0.30,
                   FuzzyNumber(6, 8, 11), FuzzyNumber(85000, 95000, 114000), 0.55),
            Method(1, 6, 120000, 0.87, 15, 8, 0.4, 0.4, 0.35,
                   FuzzyNumber(5, 6, 9), FuzzyNumber(108000, 120000, 144000), 0.50),
            Method(2, 5, 150000, 0.93, 18, 10, 0.5, 0.5, 0.40,
                   FuzzyNumber(4, 5, 7), FuzzyNumber(135000, 150000, 180000), 0.45),
        ], [4], 1.1),  # Depends on Drainage (can start early)
        
        # Activity 8: Asphalt Paving (OUTDOOR - Extreme weather sensitivity)
        Activity(8, "Asphalt Paving", [
            Method(0, 14, 380000, 0.77, 30, 15, 0.6, 0.5, 0.50,
                   FuzzyNumber(11, 14, 19), FuzzyNumber(342000, 380000, 456000), 0.95),
            Method(1, 10, 470000, 0.85, 38, 20, 0.7, 0.6, 0.55,
                   FuzzyNumber(8, 10, 14), FuzzyNumber(423000, 470000, 564000), 0.90),
            Method(2, 8, 580000, 0.92, 48, 25, 0.8, 0.7, 0.60,
                   FuzzyNumber(6, 8, 11), FuzzyNumber(522000, 580000, 696000), 0.85),
        ], [6], 1.5),  # Depends on Bridge Deck
        
        # Activity 9: Final Inspection & Opening (OUTDOOR - Low sensitivity)
        # FINAL CONVERGENCE: Requires paving AND signage
        Activity(9, "Inspection & Opening", [
            Method(0, 6, 45000, 0.85, 10, 4, 0.2, 0.2, 0.20,
                   FuzzyNumber(5, 6, 8), FuzzyNumber(40000, 45000, 54000), 0.25),
            Method(1, 4, 58000, 0.91, 12, 5, 0.3, 0.3, 0.25,
                   FuzzyNumber(3, 4, 6), FuzzyNumber(52000, 58000, 70000), 0.20),
            Method(2, 3, 72000, 0.96, 15, 6, 0.4, 0.4, 0.30,
                   FuzzyNumber(2, 3, 5), FuzzyNumber(65000, 72000, 86000), 0.15),
        ], [7, 8], 1.0),  # COMPLEX: Depends on BOTH signage AND paving
    ]
    return Project("Highway Interchange", "Highway", activities, 45000, 80, 40)


def create_pipeline_project() -> Project:
    """
    Create Pipeline Network project (8 activities) - Infrastructure.
    
    Complex network with parallel excavation branches:
    
    [0] Survey ─→ [1] ROW ─┬─→ [2] Excavation A ─→ [4] Pipe A ─┐
                          │                                    ├─→ [6] Testing ─→ [7]
                          └─→ [3] Excavation B ─→ [5] Pipe B ─┘
    
    Features:
    - Parallel trenching operations (A & B branches)
    - Convergence at testing phase
    - All activities OUTDOOR with extreme weather sensitivity
    - Ground conditions highly affected by rain/frost
    
    Weather profile: Temperate (rain delays excavation, frost stops work)
    """
    activities = [
        # Activity 0: Survey & Staking (OUTDOOR - Medium sensitivity)
        Activity(0, "Survey & Staking", [
            Method(0, 8, 65000, 0.80, 10, 5, 0.3, 0.3, 0.25,
                   FuzzyNumber(6, 8, 11), FuzzyNumber(58000, 65000, 78000), 0.60),
            Method(1, 6, 82000, 0.87, 12, 6, 0.4, 0.4, 0.30,
                   FuzzyNumber(5, 6, 9), FuzzyNumber(74000, 82000, 98000), 0.55),
            Method(2, 5, 100000, 0.93, 15, 8, 0.5, 0.5, 0.35,
                   FuzzyNumber(4, 5, 7), FuzzyNumber(90000, 100000, 120000), 0.50),
        ], [], 1.0),  # Start activity
        
        # Activity 1: Right-of-Way Clearing (OUTDOOR - High sensitivity)
        Activity(1, "ROW Clearing", [
            Method(0, 12, 145000, 0.74, 18, 10, 0.7, 0.5, 0.40,
                   FuzzyNumber(9, 12, 17), FuzzyNumber(130000, 145000, 174000), 0.80),
            Method(1, 9, 185000, 0.82, 22, 12, 0.8, 0.6, 0.45,
                   FuzzyNumber(7, 9, 13), FuzzyNumber(166000, 185000, 222000), 0.75),
            Method(2, 7, 230000, 0.89, 28, 15, 0.9, 0.7, 0.50,
                   FuzzyNumber(5, 7, 10), FuzzyNumber(207000, 230000, 276000), 0.70),
        ], [0], 1.3),  # Depends on Survey
        
        # Activity 2: Excavation Branch A (OUTDOOR - Extreme sensitivity)
        # PARALLEL with Activity 3
        Activity(2, "Excavation A", [
            Method(0, 18, 380000, 0.71, 30, 18, 0.9, 0.7, 0.55,
                   FuzzyNumber(14, 18, 25), FuzzyNumber(342000, 380000, 456000), 0.95),
            Method(1, 14, 470000, 0.79, 38, 22, 1.0, 0.8, 0.60,
                   FuzzyNumber(11, 14, 20), FuzzyNumber(423000, 470000, 564000), 0.90),
            Method(2, 11, 580000, 0.87, 48, 28, 1.1, 0.9, 0.65,
                   FuzzyNumber(8, 11, 15), FuzzyNumber(522000, 580000, 696000), 0.85),
        ], [1], 1.5),  # Depends on ROW - PARALLEL BRANCH
        
        # Activity 3: Excavation Branch B (OUTDOOR - Extreme sensitivity)
        # PARALLEL with Activity 2
        Activity(3, "Excavation B", [
            Method(0, 16, 350000, 0.72, 28, 16, 0.9, 0.7, 0.55,
                   FuzzyNumber(13, 16, 22), FuzzyNumber(315000, 350000, 420000), 0.95),
            Method(1, 12, 430000, 0.80, 35, 20, 1.0, 0.8, 0.60,
                   FuzzyNumber(9, 12, 17), FuzzyNumber(387000, 430000, 516000), 0.90),
            Method(2, 10, 530000, 0.88, 44, 25, 1.1, 0.9, 0.65,
                   FuzzyNumber(7, 10, 14), FuzzyNumber(477000, 530000, 636000), 0.85),
        ], [1], 1.5),  # Depends on ROW - PARALLEL BRANCH
        
        # Activity 4: Pipe Installation A (OUTDOOR - Very High sensitivity)
        Activity(4, "Pipe Installation A", [
            Method(0, 15, 520000, 0.75, 25, 15, 0.6, 0.5, 0.50,
                   FuzzyNumber(12, 15, 21), FuzzyNumber(468000, 520000, 624000), 0.85),
            Method(1, 12, 640000, 0.83, 32, 18, 0.7, 0.6, 0.55,
                   FuzzyNumber(9, 12, 17), FuzzyNumber(576000, 640000, 768000), 0.80),
            Method(2, 9, 780000, 0.90, 40, 22, 0.8, 0.7, 0.60,
                   FuzzyNumber(7, 9, 13), FuzzyNumber(702000, 780000, 936000), 0.75),
        ], [2], 1.6),  # Depends on Excavation A
        
        # Activity 5: Pipe Installation B (OUTDOOR - Very High sensitivity)
        Activity(5, "Pipe Installation B", [
            Method(0, 14, 480000, 0.76, 24, 14, 0.6, 0.5, 0.50,
                   FuzzyNumber(11, 14, 20), FuzzyNumber(432000, 480000, 576000), 0.85),
            Method(1, 11, 590000, 0.84, 30, 17, 0.7, 0.6, 0.55,
                   FuzzyNumber(8, 11, 15), FuzzyNumber(531000, 590000, 708000), 0.80),
            Method(2, 8, 720000, 0.91, 38, 21, 0.8, 0.7, 0.60,
                   FuzzyNumber(6, 8, 11), FuzzyNumber(648000, 720000, 864000), 0.75),
        ], [3], 1.6),  # Depends on Excavation B
        
        # Activity 6: Hydrostatic Testing (OUTDOOR - Medium sensitivity)
        # CONVERGENCE: Requires BOTH pipe installations complete
        Activity(6, "Hydrostatic Testing", [
            Method(0, 10, 180000, 0.82, 15, 8, 0.4, 0.3, 0.35,
                   FuzzyNumber(8, 10, 14), FuzzyNumber(162000, 180000, 216000), 0.55),
            Method(1, 8, 225000, 0.89, 18, 10, 0.5, 0.4, 0.40,
                   FuzzyNumber(6, 8, 11), FuzzyNumber(202000, 225000, 270000), 0.50),
            Method(2, 6, 280000, 0.95, 22, 12, 0.6, 0.5, 0.45,
                   FuzzyNumber(5, 6, 9), FuzzyNumber(252000, 280000, 336000), 0.45),
        ], [4, 5], 1.4),  # COMPLEX: Depends on BOTH pipe installations
        
        # Activity 7: Final Inspection & Backfill (OUTDOOR - High sensitivity)
        Activity(7, "Inspection & Backfill", [
            Method(0, 10, 120000, 0.83, 18, 10, 0.5, 0.4, 0.40,
                   FuzzyNumber(8, 10, 14), FuzzyNumber(108000, 120000, 144000), 0.70),
            Method(1, 8, 150000, 0.90, 22, 12, 0.6, 0.5, 0.45,
                   FuzzyNumber(6, 8, 11), FuzzyNumber(135000, 150000, 180000), 0.65),
            Method(2, 6, 190000, 0.96, 28, 15, 0.7, 0.6, 0.50,
                   FuzzyNumber(5, 6, 9), FuzzyNumber(171000, 190000, 228000), 0.60),
        ], [6], 1.2),  # Depends on Testing
    ]
    return Project("Pipeline Network", "Pipeline", activities, 35000, 60, 35)


# =============================================================================
# PART 3: CPM SCHEDULER
# =============================================================================

class CPMScheduler:
    """Critical Path Method scheduler for project networks."""
    
    def __init__(self, project: Project):
        self.project = project
        self.n = project.n_activities
        
    def schedule(self, solution: np.ndarray) -> Dict:
        """
        Compute schedule using forward pass CPM.
        
        Args:
            solution: Array of method indices for each activity
            
        Returns:
            Dictionary with schedule details
        """
        # Get durations for selected methods
        durations = np.zeros(self.n, dtype=int)
        for i, act in enumerate(self.project.activities):
            method_idx = int(solution[i])
            durations[i] = act.methods[method_idx].duration
        
        # Forward pass
        es = np.zeros(self.n, dtype=int)  # Early Start
        ef = np.zeros(self.n, dtype=int)  # Early Finish
        
        for i, act in enumerate(self.project.activities):
            if act.predecessors:
                es[i] = max(ef[p] for p in act.predecessors)
            ef[i] = es[i] + durations[i]
        
        makespan = max(ef)
        
        # Compute daily resource usage and active activities
        daily_labor = np.zeros(makespan, dtype=int)
        daily_equipment = np.zeros(makespan, dtype=int)
        daily_active = np.zeros(makespan, dtype=int)
        
        for i, act in enumerate(self.project.activities):
            method = act.methods[int(solution[i])]
            for t in range(es[i], ef[i]):
                daily_labor[t] += method.labor
                daily_equipment[t] += method.equipment
                daily_active[t] += 1
        
        return {
            'es': es,
            'ef': ef,
            'durations': durations,
            'makespan': makespan,
            'daily_labor': daily_labor,
            'daily_equipment': daily_equipment,
            'daily_active': daily_active
        }


# =============================================================================
# PART 4: OBJECTIVE FUNCTIONS (Z1-Z5) - 5D Framework
# =============================================================================

class ObjectiveCalculator:
    """
    Calculate all 5 objective functions for 5D-MOPSP Framework.
    
    This is a streamlined framework focusing on core construction objectives
    with weather-sensitive scheduling under uncertainty.
    
    Objectives:
        Z1: Project Duration (minimize)
        Z2: Total Cost (minimize)
        Z3: Quality Index (maximize → converted to minimize)
        Z4: Resource Moment (minimize)
        Z5: Weather Risk Index (minimize)
    """
    
    def __init__(self, project: Project,
                 weather_profile: WeatherProfile = None,
                 start_month: int = 4,  # Default: April (good construction weather)
                 use_fuzzy: bool = True):
        """
        Initialize the objective calculator.
        
        Args:
            project: Project instance
            weather_profile: Weather profile for location (default: temperate)
            start_month: Project start month (1-12)
            use_fuzzy: Whether to use fuzzy durations for risk calculations
        """
        self.project = project
        self.scheduler = CPMScheduler(project)
        self.weather = weather_profile or WeatherProfile('temperate')
        self.start_month = start_month
        self.use_fuzzy = use_fuzzy
        
    def evaluate(self, solution: np.ndarray) -> np.ndarray:
        """
        Evaluate all 5 objectives for a solution.
        
        Args:
            solution: Array of method indices
            
        Returns:
            Array of 5 objective values
        """
        schedule = self.scheduler.schedule(solution)
        
        z1 = self._calc_duration(schedule)
        z2 = self._calc_cost(solution, schedule)
        z3 = self._calc_quality(solution)
        z4 = self._calc_resource_moment(schedule)
        z5 = self._calc_weather_risk(solution, schedule)
        
        # Note: Z3 is maximization, convert to minimization
        return np.array([z1, z2, -z3, z4, z5])
    
    def _calc_duration(self, schedule: Dict) -> float:
        """Z1: Project Duration (minimize)."""
        return float(schedule['makespan'])
    
    def _calc_cost(self, solution: np.ndarray, schedule: Dict) -> float:
        """Z2: Total Cost (minimize)."""
        direct_cost = 0.0
        for i, act in enumerate(self.project.activities):
            method = act.methods[int(solution[i])]
            direct_cost += method.cost
        
        indirect_cost = self.project.daily_indirect_cost * schedule['makespan']
        return direct_cost + indirect_cost
    
    def _calc_quality(self, solution: np.ndarray) -> float:
        """Z3: Quality Index (maximize, return positive)."""
        total_weighted_quality = 0.0
        total_weight = 0.0
        
        for i, act in enumerate(self.project.activities):
            method = act.methods[int(solution[i])]
            total_weighted_quality += method.quality * act.weight
            total_weight += act.weight
        
        return total_weighted_quality / total_weight if total_weight > 0 else 0.0
    
    def _calc_resource_moment(self, schedule: Dict) -> float:
        """Z4: Resource Moment (minimize)."""
        labor = schedule['daily_labor']
        equipment = schedule['daily_equipment']
        
        if len(labor) == 0:
            return 0.0
        
        # Combined resource moment
        mean_labor = np.mean(labor)
        mean_equip = np.mean(equipment)
        
        moment_labor = np.sum((labor - mean_labor) ** 2)
        moment_equip = np.sum((equipment - mean_equip) ** 2)
        
        return moment_labor + moment_equip
    
    def _calc_environmental(self, solution: np.ndarray) -> float:
        """Z5: Environmental Impact Index (minimize)."""
        total = 0.0
        for i, act in enumerate(self.project.activities):
            method = act.methods[int(solution[i])]
            total += method.environmental * method.duration
        return total
    
    def _calc_social(self, solution: np.ndarray) -> float:
        """Z6: Social Cost Index (minimize)."""
        total = 0.0
        for i, act in enumerate(self.project.activities):
            method = act.methods[int(solution[i])]
            total += method.social * method.duration
        return total
    
    def _calc_safety(self, solution: np.ndarray, schedule: Dict) -> float:
        """Z7: Safety Risk Index with congestion penalty (minimize)."""
        total_risk = 0.0
        daily_active = schedule['daily_active']
        es = schedule['es']
        ef = schedule['ef']
        
        for t in range(schedule['makespan']):
            # Congestion penalty: delta_t = exp(alpha * (n_t - 1))
            n_t = daily_active[t]
            delta_t = np.exp(self.alpha * (n_t - 1))
            
            # Sum risk for all active activities at time t
            daily_risk = 0.0
            for i, act in enumerate(self.project.activities):
                if es[i] <= t < ef[i]:
                    method = act.methods[int(solution[i])]
                    # Risk = base_risk * exposure (1 day) * congestion
                    daily_risk += method.safety_risk * delta_t
            
            total_risk += daily_risk
        
        return total_risk
    
    def _calc_weather_risk(self, solution: np.ndarray, schedule: Dict) -> float:
        """
        Z8: Weather Risk Index (minimize).
        
        Calculates the cumulative weather-related risk for the entire project,
        considering each activity's weather sensitivity and the seasonal weather
        patterns at the time of execution.
        
        Higher values indicate:
        - More weather-sensitive activities scheduled during adverse weather months
        - Longer exposure to weather delays
        - Higher probability of schedule disruption
        """
        total_risk = 0.0
        es = schedule['es']
        ef = schedule['ef']
        
        for i, act in enumerate(self.project.activities):
            method = act.methods[int(solution[i])]
            weather_sensitivity = method.weather_sensitivity
            
            # Calculate weather risk for this activity's execution window
            activity_risk = self.weather.get_seasonal_risk_index(
                start_day=es[i],
                duration=method.duration,
                weather_sensitivity=weather_sensitivity,
                start_month=self.start_month
            )
            
            # Weight by activity importance
            total_risk += activity_risk * act.weight
        
        return total_risk
    
    def get_fuzzy_duration_estimate(self, solution: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate fuzzy project duration using α-cut analysis.
        
        Returns:
            Tuple of (optimistic, most_likely, pessimistic) durations
        """
        # For a more accurate fuzzy CPM, we'd need to propagate fuzzy numbers
        # through the network. Here we use a simplified approach.
        crisp_schedule = self.scheduler.schedule(solution)
        crisp_duration = crisp_schedule['makespan']
        
        # Calculate duration variance from fuzzy activity durations
        total_variance = 0.0
        for i, act in enumerate(self.project.activities):
            method = act.methods[int(solution[i])]
            if method.fuzzy_duration:
                total_variance += method.fuzzy_duration.variance()
        
        # Standard deviation of project duration (sqrt of sum of variances on critical path)
        std_dev = np.sqrt(total_variance)
        
        # Return fuzzy estimate
        optimistic = crisp_duration - 2 * std_dev
        pessimistic = crisp_duration + 2 * std_dev
        
        return (max(0, optimistic), crisp_duration, pessimistic)
    
    def get_fuzzy_cost_estimate(self, solution: np.ndarray, schedule: Dict) -> Tuple[float, float, float]:
        """
        Calculate fuzzy total cost using α-cut analysis.
        
        Returns:
            Tuple of (low, likely, high) costs
        """
        low_cost = 0.0
        likely_cost = 0.0
        high_cost = 0.0
        
        for i, act in enumerate(self.project.activities):
            method = act.methods[int(solution[i])]
            if method.fuzzy_cost:
                low_cost += method.fuzzy_cost.low
                likely_cost += method.fuzzy_cost.mid
                high_cost += method.fuzzy_cost.high
            else:
                low_cost += method.cost
                likely_cost += method.cost
                high_cost += method.cost
        
        # Add indirect cost based on fuzzy duration
        opt_dur, likely_dur, pess_dur = self.get_fuzzy_duration_estimate(solution)
        indirect = self.project.daily_indirect_cost
        
        low_cost += indirect * opt_dur
        likely_cost += indirect * likely_dur
        high_cost += indirect * pess_dur
        
        return (low_cost, likely_cost, high_cost)


# =============================================================================
# PART 5: PYMOO PROBLEM DEFINITION
# =============================================================================

class SchedulingProblem(Problem):
    """
    Pymoo problem class for 5D-MOPSP Framework.
    
    This problem optimizes construction project scheduling with 5 objectives:
        Z1: Duration (minimize)
        Z2: Cost (minimize)
        Z3: Quality (maximize → minimize)
        Z4: Resource Moment (minimize)
        Z5: Weather Risk (minimize)
    """
    
    def __init__(self, project: Project,
                 weather_profile: WeatherProfile = None,
                 start_month: int = 4):
        """
        Initialize the scheduling problem.
        
        Args:
            project: Project instance
            weather_profile: Weather profile (default: temperate)
            start_month: Project start month (1-12), default April
        """
        self.project = project
        self.calculator = ObjectiveCalculator(
            project, weather_profile, start_month
        )
        
        # Variable bounds: method indices for each activity
        n_vars = project.n_activities
        xl = np.zeros(n_vars)
        xu = np.array([len(act.methods) - 1 for act in project.activities])
        
        super().__init__(
            n_var=n_vars,
            n_obj=5,  # 5D Framework
            n_ieq_constr=0,
            xl=xl,
            xu=xu,
            vtype=int
        )
    
    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluate population of solutions."""
        F = np.zeros((X.shape[0], 5))  # 5 objectives
        
        for i, x in enumerate(X):
            F[i] = self.calculator.evaluate(x.astype(int))
        
        out["F"] = F


# =============================================================================
# PART 6: ALGORITHM CONFIGURATION (NSGA-III for 5D Framework)
# =============================================================================

def get_ref_dirs(n_obj: int = 5, pop_size: int = 100) -> np.ndarray:
    """
    Get reference directions for NSGA-III algorithm.
    
    For 5 objectives, use energy-based method for well-distributed directions.
    """
    try:
        ref_dirs = get_reference_directions("energy", n_obj, pop_size, seed=42)
    except:
        # Fallback to das-dennis with moderate partitions
        # For 5 objectives, n_partitions=4 gives reasonable distribution
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=4)
    return ref_dirs


def create_algorithm(n_obj: int = 5, pop_size: int = 100) -> Any:
    """
    Create NSGA-III algorithm instance for 5D-MOPSP Framework.
    
    NSGA-III is chosen as the primary algorithm for this framework because:
    - Designed specifically for many-objective optimization
    - Uses reference-direction-based selection
    - Excellent convergence and diversity balance
    
    Args:
        n_obj: Number of objectives (default: 5)
        pop_size: Population size (default: 100)
        
    Returns:
        Configured NSGA-III algorithm instance
    """
    # Get reference directions
    ref_dirs = get_ref_dirs(n_obj, pop_size)
    ref_pop_size = len(ref_dirs)
    
    # Operators for integer encoding
    sampling = IntegerRandomSampling()
    crossover = SBX(prob=0.9, eta=15, repair=RoundingRepair())
    mutation = PM(eta=20, repair=RoundingRepair())
    
    # Create NSGA-III algorithm
    algorithm = NSGA3(
        ref_dirs=ref_dirs,
        pop_size=ref_pop_size,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True
    )
    
    return algorithm


# Single algorithm for framework (not benchmarking)
ALGORITHM_NAMES = ["NSGA-III"]


# =============================================================================
# PART 7: PERFORMANCE METRICS
# =============================================================================

class PerformanceMetrics:
    """Calculate optimization performance metrics."""
    
    def __init__(self, ref_point: np.ndarray = None):
        """
        Initialize metrics calculator.
        
        Args:
            ref_point: Reference point for hypervolume (anti-ideal)
        """
        self.ref_point = ref_point
    
    def hypervolume(self, F: np.ndarray, ref_point: np.ndarray = None) -> float:
        """Calculate hypervolume indicator."""
        if ref_point is None:
            ref_point = self.ref_point
        if ref_point is None:
            # Use worst values + delta
            ref_point = np.max(F, axis=0) * 1.1
        
        try:
            hv = HV(ref_point=ref_point)
            return hv(F)
        except Exception:
            return 0.0
    
    def igd(self, F: np.ndarray, pf: np.ndarray) -> float:
        """Calculate Inverted Generational Distance."""
        try:
            igd = IGD(pf)
            return igd(F)
        except Exception:
            return float('inf')
    
    def gd(self, F: np.ndarray, pf: np.ndarray) -> float:
        """Calculate Generational Distance."""
        try:
            gd = GD(pf)
            return gd(F)
        except Exception:
            return float('inf')
    
    def spacing(self, F: np.ndarray) -> float:
        """Calculate spacing metric (distribution uniformity)."""
        if len(F) < 2:
            return 0.0
        
        # Calculate distances to nearest neighbor
        distances = cdist(F, F)
        np.fill_diagonal(distances, np.inf)
        min_distances = np.min(distances, axis=1)
        
        # Spacing = std of nearest neighbor distances
        d_mean = np.mean(min_distances)
        spacing = np.sqrt(np.sum((min_distances - d_mean) ** 2) / (len(F) - 1))
        
        return spacing
    
    def spread(self, F: np.ndarray) -> float:
        """Calculate spread/diversity metric."""
        if len(F) < 2:
            return 0.0
        
        # Calculate extent in each dimension
        extents = np.max(F, axis=0) - np.min(F, axis=0)
        return np.prod(extents[extents > 0]) ** (1.0 / np.sum(extents > 0))


# =============================================================================
# PART 8: OPA-TOPSIS MCDM FRAMEWORK (5D)
# =============================================================================

class OPATOPSIS:
    """
    Ordinal Priority Approach with TOPSIS for 5D solution ranking.
    
    Criteria:
        Z1: Duration (cost - minimize)
        Z2: Cost (cost - minimize)
        Z3: Quality (benefit - maximize)
        Z4: Resource Moment (cost - minimize)
        Z5: Weather Risk (cost - minimize)
    """
    
    def __init__(self, n_criteria: int = 5):
        self.n_criteria = n_criteria
        # Default priority order: Time > Cost > Quality > Resources > Weather
        # 1=highest priority
        self.priority_order = [1, 2, 3, 4, 5]  # Ranks for Z1-Z5
        
        # Benefit (1) vs Cost (-1) criteria
        # Z3 (Quality) is benefit, others are cost
        self.criteria_type = [-1, -1, 1, -1, -1]
    
    def set_priority(self, priority_order: List[int]):
        """Set priority order for criteria (1=highest)."""
        if len(priority_order) != self.n_criteria:
            raise ValueError(f"Priority order must have {self.n_criteria} elements")
        self.priority_order = priority_order
    
    def opa_weights(self) -> np.ndarray:
        """Calculate OPA weights from ordinal rankings."""
        ranks = np.array(self.priority_order)
        reciprocals = 1.0 / ranks
        weights = reciprocals / np.sum(reciprocals)
        return weights
    
    def normalize(self, F: np.ndarray) -> np.ndarray:
        """Vector normalization of decision matrix."""
        norm = np.sqrt(np.sum(F ** 2, axis=0))
        norm[norm == 0] = 1  # Avoid division by zero
        return F / norm
    
    def topsis_rank(self, F: np.ndarray, weights: np.ndarray = None) -> Tuple[np.ndarray, int]:
        """
        Rank solutions using TOPSIS.
        
        Args:
            F: Objective values matrix (n_solutions x n_objectives)
            weights: Criteria weights (if None, use OPA weights)
            
        Returns:
            Tuple of (closeness coefficients, best solution index)
        """
        if weights is None:
            weights = self.opa_weights()
        
        # Normalize
        F_norm = self.normalize(F)
        
        # Apply weights
        F_weighted = F_norm * weights
        
        # Determine ideal solutions
        pis = np.zeros(self.n_criteria)  # Positive Ideal Solution
        nis = np.zeros(self.n_criteria)  # Negative Ideal Solution
        
        for j in range(self.n_criteria):
            if self.criteria_type[j] == 1:  # Benefit
                pis[j] = np.max(F_weighted[:, j])
                nis[j] = np.min(F_weighted[:, j])
            else:  # Cost
                pis[j] = np.min(F_weighted[:, j])
                nis[j] = np.max(F_weighted[:, j])
        
        # Calculate distances
        d_plus = np.sqrt(np.sum((F_weighted - pis) ** 2, axis=1))
        d_minus = np.sqrt(np.sum((F_weighted - nis) ** 2, axis=1))
        
        # Closeness coefficient
        cc = d_minus / (d_plus + d_minus + 1e-10)
        
        # Best solution has highest CC
        best_idx = np.argmax(cc)
        
        return cc, best_idx
    
    def rank_solutions(self, F: np.ndarray, X: np.ndarray = None) -> pd.DataFrame:
        """
        Rank all solutions and return DataFrame.
        
        Args:
            F: Objective values (5 columns for 5D framework)
            X: Solution vectors (optional)
            
        Returns:
            DataFrame with rankings and objective values
        """
        weights = self.opa_weights()
        cc, best_idx = self.topsis_rank(F, weights)
        
        # Create DataFrame with 5D objectives
        df = pd.DataFrame({
            'Rank': np.argsort(-cc) + 1,
            'CC': cc,
            'Z1_Duration': F[:, 0],
            'Z2_Cost': F[:, 1],
            'Z3_Quality': -F[:, 2],  # Convert back to positive
            'Z4_Resources': F[:, 3],
            'Z5_Weather': F[:, 4]
        })
        
        if X is not None:
            df['Solution'] = [str(list(x.astype(int))) for x in X]
        
        return df.sort_values('Rank')


# =============================================================================
# PART 9: PARALLEL EXECUTION ENGINE
# =============================================================================

def run_single_optimization(project: Project, algo_name: str, seed: int,
                           pop_size: int = 100, n_gen: int = 200,
                           weather_profile: WeatherProfile = None,
                           start_month: int = 4) -> Dict:
    """
    Run a single optimization experiment with NSGA-III.
    
    Args:
        project: Project instance
        algo_name: Algorithm name (only 'NSGA-III' supported)
        seed: Random seed
        pop_size: Population size
        n_gen: Number of generations
        weather_profile: Weather profile for the project location
        start_month: Project start month (1-12)
        
    Returns:
        Dictionary with results
    """
    np.random.seed(seed)
    
    start_time = time.time()
    
    # Create problem with weather parameters
    problem = SchedulingProblem(project, weather_profile, start_month)
    
    # Create NSGA-III algorithm (5D framework)
    algorithm = create_algorithm(n_obj=5, pop_size=pop_size)
    termination = get_termination("n_gen", n_gen)
    
    try:
        # Run optimization
        result = minimize(
            problem,
            algorithm,
            termination,
            seed=seed,
            verbose=False
        )
        
        runtime = time.time() - start_time
        
        # Extract results
        F = result.F if result.F is not None else np.array([])
        X = result.X if result.X is not None else np.array([])
        
        return {
            'project': project.name,
            'algorithm': algo_name,
            'seed': seed,
            'F': F,
            'X': X,
            'n_solutions': len(F) if len(F) > 0 else 0,
            'runtime': runtime,
            'success': True
        }
    except Exception as e:
        runtime = time.time() - start_time
        return {
            'project': project.name,
            'algorithm': algo_name,
            'seed': seed,
            'F': np.array([]),
            'X': np.array([]),
            'n_solutions': 0,
            'runtime': runtime,
            'success': False,
            'error': str(e)
        }


def run_parallel_optimization(projects: List[Project], algo_names: List[str],
                             n_runs: int = 30, n_jobs: int = -1,
                             pop_size: int = 100, n_gen: int = 200) -> List[Dict]:
    """
    Run all optimization experiments in parallel.
    
    Args:
        projects: List of projects
        algo_names: List of algorithm names
        n_runs: Number of independent runs per configuration
        n_jobs: Number of parallel jobs (-1 for all cores)
        pop_size: Population size
        n_gen: Number of generations
        
    Returns:
        List of result dictionaries
    """
    tasks = [
        (project, algo, seed)
        for project in projects
        for algo in algo_names
        for seed in range(CONFIG['seed_base'], CONFIG['seed_base'] + n_runs)
    ]
    
    total_tasks = len(tasks)
    print(f"Starting {total_tasks} optimization runs...")
    print(f"  Projects: {len(projects)}")
    print(f"  Algorithms: {len(algo_names)}")
    print(f"  Runs per config: {n_runs}")
    
    if JOBLIB_AVAILABLE and n_jobs != 1:
        # Parallel execution with progress bar
        if TQDM_AVAILABLE:
            # Use tqdm with joblib
            from joblib import Parallel, delayed
            results = []
            with tqdm(total=total_tasks, desc="Optimization Progress", 
                     unit="run", ncols=100, 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                # Process in batches to update progress
                batch_size = max(1, n_jobs if n_jobs > 0 else 11)
                for i in range(0, total_tasks, batch_size):
                    batch_tasks = tasks[i:i+batch_size]
                    batch_results = Parallel(n_jobs=n_jobs, verbose=0)(
                        delayed(run_single_optimization)(
                            project, algo, seed, pop_size, n_gen
                        )
                        for project, algo, seed in batch_tasks
                    )
                    results.extend(batch_results)
                    pbar.update(len(batch_tasks))
        else:
            results = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(run_single_optimization)(
                    project, algo, seed, pop_size, n_gen
                )
                for project, algo, seed in tasks
            )
    else:
        # Sequential execution with progress bar
        results = []
        iterator = tasks
        if TQDM_AVAILABLE:
            iterator = tqdm(tasks, desc="Optimization Progress", 
                           unit="run", ncols=100,
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for project, algo, seed in iterator:
            if TQDM_AVAILABLE:
                iterator.set_postfix({'project': project.name[:15], 'algo': algo[:10]})
            else:
                print(f"  Running: {project.name} - {algo} (seed={seed})")
            result = run_single_optimization(project, algo, seed, pop_size, n_gen)
            results.append(result)
    
    return results


def aggregate_results(results: List[Dict], ref_point: np.ndarray = None) -> pd.DataFrame:
    """
    Aggregate optimization results and compute metrics.
    
    Args:
        results: List of result dictionaries
        ref_point: Reference point for HV (optional)
        
    Returns:
        DataFrame with aggregated metrics
    """
    metrics_calc = PerformanceMetrics(ref_point)
    records = []
    
    # Group results by project to avoid memory issues with large combined fronts
    projects = list(set(r['project'] for r in results if r.get('success', False)))
    
    # Build per-project reference fronts (much smaller, avoids memory error)
    project_ref_pfs = {}
    project_ref_points = {}
    
    for proj in projects:
        proj_results = [r for r in results if r['project'] == proj and r.get('success', False)]
        proj_F = []
        for r in proj_results:
            if len(r['F']) > 0:
                # Sample to avoid memory issues (max 200 per run)
                F = r['F']
                if len(F) > 200:
                    idx = np.random.choice(len(F), 200, replace=False)
                    F = F[idx]
                proj_F.append(F)
        
        if proj_F:
            combined_F = np.vstack(proj_F)
            
            # Limit total samples to avoid memory issues
            if len(combined_F) > 5000:
                idx = np.random.choice(len(combined_F), 5000, replace=False)
                combined_F = combined_F[idx]
            
            # Simple non-dominated sorting for reference PF
            try:
                from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
                nds = NonDominatedSorting()
                fronts = nds.do(combined_F)
                ref_pf = combined_F[fronts[0]] if len(fronts) > 0 else combined_F[:500]
            except MemoryError:
                # Fallback: use sampled front
                ref_pf = combined_F[:min(500, len(combined_F))]
            
            project_ref_pfs[proj] = ref_pf
            project_ref_points[proj] = np.max(combined_F, axis=0) * 1.1
        else:
            project_ref_pfs[proj] = None
            project_ref_points[proj] = None
    
    # Default reference point if none computed
    if ref_point is None:
        # Use first available project ref point
        for proj in projects:
            if project_ref_points.get(proj) is not None:
                ref_point = project_ref_points[proj]
                break
    
    for r in results:
        proj = r['project']
        record = {
            'Project': proj,
            'Algorithm': r['algorithm'],
            'Seed': r['seed'],
            'Runtime': r['runtime'],
            'N_Solutions': r['n_solutions'],
            'Success': r['success']
        }
        
        if r['success'] and len(r['F']) > 0:
            F = r['F']
            # Use project-specific reference point and PF
            proj_ref_point = project_ref_points.get(proj, ref_point)
            proj_ref_pf = project_ref_pfs.get(proj)
            
            record['HV'] = metrics_calc.hypervolume(F, proj_ref_point)
            record['Spacing'] = metrics_calc.spacing(F)
            record['Spread'] = metrics_calc.spread(F)
            
            if proj_ref_pf is not None:
                record['IGD'] = metrics_calc.igd(F, proj_ref_pf)
                record['GD'] = metrics_calc.gd(F, proj_ref_pf)
            else:
                record['IGD'] = np.nan
                record['GD'] = np.nan
        else:
            record['HV'] = 0.0
            record['Spacing'] = np.nan
            record['Spread'] = np.nan
            record['IGD'] = np.nan
            record['GD'] = np.nan
        
        records.append(record)
    
    return pd.DataFrame(records)


# =============================================================================
# PART 10: STATISTICAL ANALYSIS
# =============================================================================

class StatisticalAnalysis:
    """Statistical tests for algorithm comparison."""
    
    @staticmethod
    def friedman_test(df: pd.DataFrame, metric: str = 'HV', 
                      group_col: str = 'Algorithm') -> Dict:
        """
        Perform Friedman test for algorithm ranking.
        
        Args:
            df: Results DataFrame
            metric: Metric to compare
            group_col: Grouping column
            
        Returns:
            Dictionary with test results
        """
        # Pivot data for Friedman test
        groups = df[group_col].unique()
        data = []
        
        for group in groups:
            values = df[df[group_col] == group][metric].dropna().values
            data.append(values)
        
        # Ensure equal sample sizes (truncate to minimum)
        min_len = min(len(d) for d in data)
        data = [d[:min_len] for d in data]
        
        if min_len < 3:
            return {'error': 'Insufficient data for Friedman test'}
        
        try:
            stat, p_value = stats.friedmanchisquare(*data)
            
            # Calculate mean ranks
            ranks = np.zeros((min_len, len(groups)))
            for i in range(min_len):
                row = [data[j][i] for j in range(len(groups))]
                # Higher is better for HV, lower for others
                if metric in ['HV', 'Spread']:
                    ranks[i] = stats.rankdata([-v for v in row])
                else:
                    ranks[i] = stats.rankdata(row)
            
            mean_ranks = np.mean(ranks, axis=0)
            
            return {
                'statistic': stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'mean_ranks': dict(zip(groups, mean_ranks)),
                'ranking': [groups[i] for i in np.argsort(mean_ranks)]
            }
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def wilcoxon_pairwise(df: pd.DataFrame, metric: str = 'HV',
                        group_col: str = 'Algorithm', alpha: float = 0.05) -> pd.DataFrame:
        """
        Perform pairwise Wilcoxon signed-rank tests.
        
        Args:
            df: Results DataFrame
            metric: Metric to compare
            group_col: Grouping column
            alpha: Significance level
            
        Returns:
            DataFrame with p-values
        """
        groups = list(df[group_col].unique())
        n_groups = len(groups)
        p_matrix = np.ones((n_groups, n_groups))
        
        for i in range(n_groups):
            for j in range(i + 1, n_groups):
                vals_i = df[df[group_col] == groups[i]][metric].dropna().values
                vals_j = df[df[group_col] == groups[j]][metric].dropna().values
                
                min_len = min(len(vals_i), len(vals_j))
                if min_len >= 5:
                    try:
                        _, p = stats.wilcoxon(vals_i[:min_len], vals_j[:min_len])
                        p_matrix[i, j] = p
                        p_matrix[j, i] = p
                    except Exception:
                        pass
        
        result_df = pd.DataFrame(p_matrix, index=groups, columns=groups)
        return result_df


# =============================================================================
# PART 11: VISUALIZATION SUITE
# =============================================================================

class Visualizer:
    """Generate publication-quality figures."""
    
    def __init__(self, output_dir: str = 'results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = plt.cm.tab10.colors
        self.dpi = CONFIG['fig_dpi']
    
    def plot_convergence(self, results: List[Dict], metric: str = 'HV',
                        save: bool = True) -> plt.Figure:
        """Plot convergence curves for algorithms."""
        projects = list(set(r['project'] for r in results))
        algorithms = list(set(r['algorithm'] for r in results))
        n_projects = len(projects)
        
        fig, axes = plt.subplots(1, max(n_projects, 1), figsize=(5 * max(n_projects, 1), 5))
        if n_projects == 1:
            axes = [axes]  # Make iterable
        
        for idx, project in enumerate(projects):
            ax = axes[idx]
            for i, algo in enumerate(algorithms):
                proj_results = [r for r in results 
                              if r['project'] == project and r['algorithm'] == algo and r['success']]
                if proj_results:
                    hvs = [r.get('HV', 0) for r in proj_results]
                    ax.bar(i, np.mean(hvs), yerr=np.std(hvs) if len(hvs) > 1 else 0, 
                          color=self.colors[i % len(self.colors)], alpha=0.7,
                          label=algo if idx == 0 else "")
            
            ax.set_title(project, fontsize=12, fontweight='bold')
            ax.set_ylabel('Hypervolume' if idx == 0 else '')
            ax.set_xticks(range(len(algorithms)))
            ax.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=8)
        
        fig.suptitle('Algorithm Performance Comparison (Hypervolume)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'fig_convergence.png', dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_pareto_fronts(self, results: List[Dict], project_name: str,
                          objectives: Tuple[int, int] = (0, 1),
                          save: bool = True) -> plt.Figure:
        """Plot 2D Pareto front projections."""
        obj_names = ['Duration', 'Cost', 'Quality', 'Resources', 'Weather']  # 5D Framework
        
        proj_results = [r for r in results if r['project'] == project_name and r['success']]
        algorithms = list(set(r['algorithm'] for r in proj_results))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for i, algo in enumerate(algorithms):
            algo_results = [r for r in proj_results if r['algorithm'] == algo]
            if algo_results and len(algo_results[0]['F']) > 0:
                # Take best run (highest HV)
                best = max(algo_results, key=lambda x: len(x['F']))
                F = best['F']
                
                ax.scatter(F[:, objectives[0]], F[:, objectives[1]],
                          c=[self.colors[i % len(self.colors)]], 
                          label=algo, alpha=0.6, s=30)
        
        ax.set_xlabel(f'Z{objectives[0]+1}: {obj_names[objectives[0]]}', fontsize=12)
        ax.set_ylabel(f'Z{objectives[1]+1}: {obj_names[objectives[1]]}', fontsize=12)
        ax.set_title(f'Pareto Front Projection - {project_name}', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / f'fig_pareto_{project_name.replace(" ", "_")}.png', 
                       dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_radar(self, df: pd.DataFrame, project_name: str = None,
                  save: bool = True) -> plt.Figure:
        """Plot radar chart for algorithm comparison."""
        metrics = ['HV', 'Spacing', 'Runtime']
        
        if project_name:
            df = df[df['Project'] == project_name]
        
        # Aggregate by algorithm
        agg = df.groupby('Algorithm')[metrics].mean()
        
        # Normalize metrics (0-1 scale)
        agg_norm = (agg - agg.min()) / (agg.max() - agg.min() + 1e-10)
        # Invert Spacing and Runtime (lower is better)
        agg_norm['Spacing'] = 1 - agg_norm['Spacing']
        agg_norm['Runtime'] = 1 - agg_norm['Runtime']
        
        algorithms = agg_norm.index.tolist()
        n_metrics = len(metrics)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        for i, algo in enumerate(algorithms):
            values = agg_norm.loc[algo].values.tolist()
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, 
                   color=self.colors[i % len(self.colors)], label=algo)
            ax.fill(angles, values, alpha=0.1, 
                   color=self.colors[i % len(self.colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_title(f'Algorithm Radar Chart{" - " + project_name if project_name else ""}',
                    fontsize=14, fontweight='bold', y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
        
        plt.tight_layout()
        
        if save:
            fname = f'fig_radar{"_" + project_name.replace(" ", "_") if project_name else ""}.png'
            fig.savefig(self.output_dir / fname, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_boxplots(self, df: pd.DataFrame, metric: str = 'HV',
                     save: bool = True) -> plt.Figure:
        """Plot box plots for metric distribution."""
        projects = df['Project'].unique()
        n_projects = len(projects)
        
        fig, axes = plt.subplots(1, max(n_projects, 1), figsize=(5 * max(n_projects, 1), 5))
        if n_projects == 1:
            axes = [axes]  # Make iterable
        
        for idx, project in enumerate(projects):
            ax = axes[idx]
            proj_df = df[df['Project'] == project]
            
            algorithms = proj_df['Algorithm'].unique()
            data = [proj_df[proj_df['Algorithm'] == algo][metric].dropna().values 
                   for algo in algorithms]
            
            # Filter out empty arrays
            valid_data = [d for d in data if len(d) > 0]
            valid_algos = [a for a, d in zip(algorithms, data) if len(d) > 0]
            
            if valid_data:
                bp = ax.boxplot(valid_data, labels=valid_algos, patch_artist=True)
                for i, patch in enumerate(bp['boxes']):
                    patch.set_facecolor(self.colors[i % len(self.colors)])
                    patch.set_alpha(0.7)
            
            ax.set_title(project, fontsize=12, fontweight='bold')
            ax.set_ylabel(metric if idx == 0 else '')
            ax.tick_params(axis='x', rotation=45)
        
        fig.suptitle(f'{metric} Distribution by Algorithm', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / f'fig_boxplot_{metric}.png', 
                       dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_heatmap(self, df: pd.DataFrame, title: str = 'Pairwise Comparison',
                    save: bool = True, filename: str = 'fig_heatmap.png') -> plt.Figure:
        """Plot heatmap for pairwise comparisons."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(df, annot=True, fmt='.3f', cmap='RdYlGn_r',
                   center=0.05, ax=ax, cbar_kws={'label': 'p-value'})
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_weather_concept(self, save: bool = True) -> plt.Figure:
        """Plot Weather Risk Index conceptual model for 5D framework."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Panel (a): Seasonal weather productivity factors
        ax = axes[0]
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Temperate climate productivity factors
        temperate = [0.6, 0.7, 0.85, 0.95, 1.0, 1.0, 1.0, 1.0, 0.95, 0.85, 0.75, 0.65]
        
        x = np.arange(len(months))
        bars = ax.bar(x, temperate, color='steelblue', edgecolor='black', linewidth=1, alpha=0.8)
        
        # Highlight best months
        for i, (bar, val) in enumerate(zip(bars, temperate)):
            if val >= 0.95:
                bar.set_color('#2ca02c')
            elif val < 0.75:
                bar.set_color('#d62728')
        
        ax.set_xticks(x)
        ax.set_xticklabels(months, rotation=45, ha='right')
        ax.set_ylabel('Productivity Factor', fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.set_title('(a) Seasonal Weather Productivity (Temperate Climate)', fontsize=12, fontweight='bold')
        ax.axhline(y=0.8, color='orange', linestyle='--', linewidth=1.5, label='Threshold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Panel (b): Weather sensitivity impact
        ax = axes[1]
        sensitivity_levels = np.linspace(0, 1, 100)
        
        # Risk multipliers for different weather conditions
        good_weather = 1 + 0.1 * sensitivity_levels
        moderate_weather = 1 + 0.5 * sensitivity_levels
        poor_weather = 1 + 1.5 * sensitivity_levels
        
        ax.plot(sensitivity_levels, good_weather, 'g-', linewidth=2.5, label='Good Weather')
        ax.plot(sensitivity_levels, moderate_weather, 'orange', linewidth=2.5, label='Moderate Weather')
        ax.plot(sensitivity_levels, poor_weather, 'r-', linewidth=2.5, label='Poor Weather')
        
        ax.fill_between(sensitivity_levels, good_weather, poor_weather, alpha=0.1, color='gray')
        
        ax.set_xlabel('Activity Weather Sensitivity', fontsize=11)
        ax.set_ylabel('Duration Risk Multiplier', fontsize=11)
        ax.set_title('(b) Weather Impact on Activity Duration', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 2.6)
        
        fig.suptitle('Weather Risk Index (Z5) - Conceptual Framework',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'fig_weather_concept.png', 
                       dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_pareto_matrix(self, results: List[Dict], project_name: str,
                          save: bool = True) -> plt.Figure:
        """
        Plot 5x5 Pareto front matrix showing all pairwise objective projections.
        This is a key figure for the 5D-MOPSP framework.
        """
        obj_names = ['Z1:Duration', 'Z2:Cost', 'Z3:Quality', 'Z4:Resources', 'Z5:Weather']
        n_obj = 5
        
        # Collect all solutions from all algorithms for this project
        proj_results = [r for r in results if r['project'] == project_name and r['success']]
        algorithms = list(set(r['algorithm'] for r in proj_results))
        
        fig, axes = plt.subplots(n_obj, n_obj, figsize=(20, 20))
        
        for i in range(n_obj):
            for j in range(n_obj):
                ax = axes[i, j]
                
                if i == j:
                    # Diagonal: histogram of objective values
                    for k, algo in enumerate(algorithms):
                        algo_results = [r for r in proj_results if r['algorithm'] == algo]
                        if algo_results:
                            all_vals = np.concatenate([r['F'][:, i] for r in algo_results if len(r['F']) > 0])
                            if len(all_vals) > 0:
                                ax.hist(all_vals, bins=20, alpha=0.5, 
                                       color=self.colors[k % len(self.colors)],
                                       density=True)
                    ax.set_xlabel(obj_names[i], fontsize=8)
                else:
                    # Off-diagonal: scatter plot
                    for k, algo in enumerate(algorithms):
                        algo_results = [r for r in proj_results if r['algorithm'] == algo]
                        if algo_results and len(algo_results[0]['F']) > 0:
                            best = max(algo_results, key=lambda x: len(x['F']))
                            F = best['F']
                            ax.scatter(F[:, j], F[:, i], 
                                      c=[self.colors[k % len(self.colors)]], 
                                      alpha=0.4, s=10)
                
                # Labels only on edges
                if i == n_obj - 1:
                    ax.set_xlabel(obj_names[j], fontsize=8)
                if j == 0:
                    ax.set_ylabel(obj_names[i], fontsize=8)
                    
                ax.tick_params(labelsize=6)
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=self.colors[k % len(self.colors)], 
                          markersize=8, label=algo) 
                          for k, algo in enumerate(algorithms)]
        fig.legend(handles=legend_elements, loc='upper right', 
                  bbox_to_anchor=(0.99, 0.99), fontsize=9)
        
        fig.suptitle(f'5D Pareto Front Matrix - {project_name}', 
                    fontsize=16, fontweight='bold', y=1.01)
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / f'fig_pareto_matrix_{project_name.replace(" ", "_")}.png',
                       dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_pareto_3d(self, results: List[Dict], project_name: str,
                      objectives: Tuple[int, int, int] = (0, 1, 6),
                      save: bool = True) -> plt.Figure:
        """Plot 3D Pareto front projection."""
        obj_names = ['Duration', 'Cost', 'Quality', 'Resources', 'Weather']  # 5D
        
        proj_results = [r for r in results if r['project'] == project_name and r['success']]
        algorithms = list(set(r['algorithm'] for r in proj_results))
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        for i, algo in enumerate(algorithms):
            algo_results = [r for r in proj_results if r['algorithm'] == algo]
            if algo_results and len(algo_results[0]['F']) > 0:
                best = max(algo_results, key=lambda x: len(x['F']))
                F = best['F']
                ax.scatter(F[:, objectives[0]], F[:, objectives[1]], F[:, objectives[2]],
                          c=[self.colors[i % len(self.colors)]], 
                          label=algo, alpha=0.6, s=30)
        
        ax.set_xlabel(f'Z{objectives[0]+1}: {obj_names[objectives[0]]}', fontsize=11)
        ax.set_ylabel(f'Z{objectives[1]+1}: {obj_names[objectives[1]]}', fontsize=11)
        ax.set_zlabel(f'Z{objectives[2]+1}: {obj_names[objectives[2]]}', fontsize=11)
        ax.set_title(f'3D Pareto Front - {project_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        
        if save:
            fig.savefig(self.output_dir / f'fig_pareto_3d_{project_name.replace(" ", "_")}.png',
                       dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_parallel_coordinates(self, results: List[Dict], project_name: str,
                                  n_solutions: int = 50, save: bool = True) -> plt.Figure:
        """Plot parallel coordinates for Pareto solutions."""
        obj_names = ['Z1:Duration', 'Z2:Cost', 'Z3:Quality', 'Z4:Resource', 'Z5:Weather']  # 5D
        
        proj_results = [r for r in results if r['project'] == project_name and r['success']]
        algorithms = list(set(r['algorithm'] for r in proj_results))
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for k, algo in enumerate(algorithms):
            algo_results = [r for r in proj_results if r['algorithm'] == algo]
            if algo_results:
                # Combine and sample solutions
                all_F = np.vstack([r['F'] for r in algo_results if len(r['F']) > 0])
                if len(all_F) > n_solutions:
                    indices = np.random.choice(len(all_F), n_solutions, replace=False)
                    all_F = all_F[indices]
                
                # Normalize for visualization
                F_norm = (all_F - all_F.min(axis=0)) / (all_F.max(axis=0) - all_F.min(axis=0) + 1e-10)
                # Invert quality (Z3) so higher is better visually
                F_norm[:, 2] = 1 - F_norm[:, 2]
                
                x = np.arange(5)  # 5D Framework
                for sol in F_norm:
                    ax.plot(x, sol, color=self.colors[k % len(self.colors)], 
                           alpha=0.3, linewidth=1)
        
        # Add algorithm legend with thicker lines
        for k, algo in enumerate(algorithms):
            ax.plot([], [], color=self.colors[k % len(self.colors)], 
                   linewidth=3, label=algo)
        
        ax.set_xticks(range(5))  # 5D Framework
        ax.set_xticklabels(obj_names, fontsize=10)
        ax.set_ylabel('Normalized Value (lower is better except Z3)', fontsize=11)
        ax.set_title(f'Parallel Coordinates - {project_name}', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / f'fig_parallel_coords_{project_name.replace(" ", "_")}.png',
                       dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_objective_correlation(self, results: List[Dict], project_name: str,
                                   save: bool = True) -> plt.Figure:
        """Plot correlation matrix between objectives."""
        obj_names = ['Z1:Duration', 'Z2:Cost', 'Z3:Quality', 'Z4:Resource', 'Z5:Weather']  # 5D
        
        proj_results = [r for r in results if r['project'] == project_name and r['success']]
        
        # Combine all solutions
        all_F = []
        for r in proj_results:
            if len(r['F']) > 0:
                all_F.append(r['F'])
        
        if not all_F:
            return None
        
        combined_F = np.vstack(all_F)
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(combined_F.T)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, vmin=-1, vmax=1, ax=ax,
                   xticklabels=obj_names, yticklabels=obj_names,
                   cbar_kws={'label': 'Correlation'})
        
        ax.set_title(f'Objective Correlation Matrix - {project_name}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / f'fig_correlation_{project_name.replace(" ", "_")}.png',
                       dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_algorithm_ranking(self, df: pd.DataFrame, metric: str = 'HV',
                               save: bool = True) -> plt.Figure:
        """Plot algorithm ranking comparison across projects."""
        projects = df['Project'].unique()
        algorithms = df['Algorithm'].unique()
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(projects))
        width = 0.8 / max(len(algorithms), 1)
        
        for i, algo in enumerate(algorithms):
            means = []
            stds = []
            for proj in projects:
                data = df[(df['Project'] == proj) & (df['Algorithm'] == algo)][metric].dropna()
                if len(data) > 0:
                    means.append(data.mean())
                    stds.append(data.std() if len(data) > 1 else 0)
                else:
                    means.append(0)
                    stds.append(0)
            
            offset = (i - len(algorithms)/2 + 0.5) * width
            bars = ax.bar(x + offset, means, width, yerr=stds,
                         label=algo, color=self.colors[i % len(self.colors)],
                         alpha=0.8, capsize=2)
        
        ax.set_xlabel('Project', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'Algorithm Comparison by {metric}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(projects, fontsize=10)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / f'fig_algorithm_ranking_{metric}.png',
                       dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_runtime_comparison(self, df: pd.DataFrame, save: bool = True) -> plt.Figure:
        """Plot runtime comparison across algorithms and projects."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Box plot by algorithm
        ax = axes[0]
        algorithms = df['Algorithm'].unique()
        data = [df[df['Algorithm'] == algo]['Runtime'].dropna().values for algo in algorithms]
        
        # Filter out empty arrays
        valid_data = [d for d in data if len(d) > 0]
        valid_algos = [a for a, d in zip(algorithms, data) if len(d) > 0]
        
        if valid_data:
            bp = ax.boxplot(valid_data, labels=valid_algos, patch_artist=True)
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(self.colors[i % len(self.colors)])
                patch.set_alpha(0.7)
        ax.set_ylabel('Runtime (seconds)', fontsize=11)
        ax.set_title('Runtime Distribution by Algorithm', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Right: Bar chart mean runtime by project
        ax = axes[1]
        projects = df['Project'].unique()
        for i, algo in enumerate(algorithms):
            means = []
            for p in projects:
                runtime_data = df[(df['Project'] == p) & (df['Algorithm'] == algo)]['Runtime'].dropna()
                means.append(runtime_data.mean() if len(runtime_data) > 0 else 0)
            x = np.arange(len(projects))
            width = 0.8 / max(len(algorithms), 1)
            offset = (i - len(algorithms)/2 + 0.5) * width
            ax.bar(x + offset, means, width, 
                  color=self.colors[i % len(self.colors)], alpha=0.8)
        
        ax.set_xlabel('Project', fontsize=11)
        ax.set_ylabel('Mean Runtime (seconds)', fontsize=11)
        ax.set_title('Mean Runtime by Project', fontsize=12, fontweight='bold')
        ax.set_xticks(np.arange(len(projects)))
        ax.set_xticklabels(projects, fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'fig_runtime_comparison.png',
                       dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_convergence_curves(self, df: pd.DataFrame, save: bool = True) -> plt.Figure:
        """Plot multi-metric convergence comparison."""
        metrics = ['HV', 'IGD', 'Spacing']
        projects = df['Project'].unique()
        n_projects = len(projects)
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(n_metrics, max(n_projects, 1), 
                                figsize=(5 * max(n_projects, 1), 4 * n_metrics))
        
        # Handle different array shapes
        if n_projects == 1 and n_metrics == 1:
            axes = np.array([[axes]])
        elif n_projects == 1:
            axes = axes.reshape(-1, 1)
        elif n_metrics == 1:
            axes = axes.reshape(1, -1)
        
        for i, metric in enumerate(metrics):
            for j, project in enumerate(projects):
                ax = axes[i, j]
                proj_df = df[df['Project'] == project]
                
                algorithms = proj_df['Algorithm'].unique()
                data = [proj_df[proj_df['Algorithm'] == algo][metric].dropna().values 
                       for algo in algorithms]
                
                # Filter out empty arrays
                valid_data = [d for d in data if len(d) > 0]
                valid_algos = [a for a, d in zip(algorithms, data) if len(d) > 0]
                
                if valid_data:
                    bp = ax.boxplot(valid_data, patch_artist=True)
                    for k, patch in enumerate(bp['boxes']):
                        patch.set_facecolor(self.colors[k % len(self.colors)])
                        patch.set_alpha(0.7)
                    
                    if i == n_metrics - 1:
                        ax.set_xticklabels(valid_algos, rotation=45, ha='right', fontsize=7)
                    else:
                        ax.set_xticklabels([])
                
                if i == 0:
                    ax.set_title(project, fontsize=11, fontweight='bold')
                if j == 0:
                    ax.set_ylabel(metric, fontsize=10)
        
        fig.suptitle('Multi-Metric Performance Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'fig_multi_metric_comparison.png',
                       dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_solution_distribution(self, results: List[Dict], project_name: str,
                                   save: bool = True) -> plt.Figure:
        """Plot solution count distribution across algorithms."""
        proj_results = [r for r in results if r['project'] == project_name and r['success']]
        
        algo_solutions = {}
        for r in proj_results:
            algo = r['algorithm']
            if algo not in algo_solutions:
                algo_solutions[algo] = []
            algo_solutions[algo].append(r['n_solutions'])
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Box plot
        ax = axes[0]
        algos = list(algo_solutions.keys())
        data = [algo_solutions[a] for a in algos]
        bp = ax.boxplot(data, labels=algos, patch_artist=True)
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(self.colors[i % len(self.colors)])
            patch.set_alpha(0.7)
        ax.set_ylabel('Number of Pareto Solutions', fontsize=11)
        ax.set_title('Solution Count Distribution', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Right: Mean with error bars
        ax = axes[1]
        means = [np.mean(algo_solutions[a]) for a in algos]
        stds = [np.std(algo_solutions[a]) for a in algos]
        x = np.arange(len(algos))
        bars = ax.bar(x, means, yerr=stds, capsize=3,
                     color=[self.colors[i % len(self.colors)] for i in range(len(algos))],
                     alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(algos, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Mean Pareto Solutions', fontsize=11)
        ax.set_title('Average Solution Count', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(f'Pareto Solution Distribution - {project_name}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / f'fig_solution_dist_{project_name.replace(" ", "_")}.png',
                       dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_topsis_sensitivity(self, results: List[Dict], project_name: str,
                                save: bool = True) -> plt.Figure:
        """Plot TOPSIS sensitivity analysis with different weight scenarios."""
        proj_results = [r for r in results if r['project'] == project_name and r['success']]
        
        # Combine all solutions
        all_F = []
        all_algos = []
        for r in proj_results:
            if len(r['F']) > 0:
                all_F.append(r['F'])
                all_algos.extend([r['algorithm']] * len(r['F']))
        
        if not all_F:
            return None
        
        combined_F = np.vstack(all_F)
        
        # Define weight scenarios (5D: Duration, Cost, Quality, Resource, WeatherRisk)
        scenarios = {
            'Equal': [1, 1, 1, 1, 1],
            'Time-Cost Focus': [1, 2, 5, 4, 3],
            'Quality Focus': [3, 4, 1, 5, 2],
            'WeatherRisk Focus': [2, 3, 4, 5, 1],
            'Resource Focus': [4, 5, 3, 1, 2],
        }
        
        mcdm = OPATOPSIS()
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (scenario_name, priorities) in enumerate(scenarios.items()):
            if idx >= len(axes):
                break
            ax = axes[idx]
            
            mcdm.set_priority(priorities)
            cc, best_idx = mcdm.topsis_rank(combined_F)
            
            # Plot top 10 solutions
            top_indices = np.argsort(-cc)[:10]
            top_cc = cc[top_indices]
            top_algos = [all_algos[i] for i in top_indices]
            
            colors_list = [self.colors[list(set(all_algos)).index(a) % len(self.colors)] 
                          for a in top_algos]
            
            ax.barh(range(10), top_cc, color=colors_list, alpha=0.8)
            ax.set_yticks(range(10))
            ax.set_yticklabels([f'{a} (#{i+1})' for i, a in enumerate(top_algos)], fontsize=8)
            ax.set_xlabel('TOPSIS CC', fontsize=10)
            ax.set_title(scenario_name, fontsize=11, fontweight='bold')
            ax.invert_yaxis()
        
        # Legend in last subplot
        ax = axes[-1]
        unique_algos = list(set(all_algos))
        for i, algo in enumerate(unique_algos):
            ax.barh(i, 1, color=self.colors[i % len(self.colors)], alpha=0.8, label=algo)
        ax.set_xlim(0, 1)
        ax.legend(loc='center', fontsize=9)
        ax.set_title('Algorithm Legend', fontsize=11, fontweight='bold')
        ax.axis('off')
        
        fig.suptitle(f'TOPSIS Sensitivity Analysis - {project_name}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / f'fig_topsis_sensitivity_{project_name.replace(" ", "_")}.png',
                       dpi=self.dpi, bbox_inches='tight')
        
        return fig


# =============================================================================
# PART 12: TABLE GENERATION
# =============================================================================

class TableGenerator:
    """Generate publication-ready tables."""
    
    def __init__(self, output_dir: str = 'results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def project_characteristics(self, projects: List[Project]) -> pd.DataFrame:
        """Generate Table 3: Project Dataset Characteristics."""
        records = []
        for i, proj in enumerate(projects):
            n_relationships = sum(len(act.predecessors) for act in proj.activities)
            records.append({
                'Project': f'Project {chr(65+i)}',
                'Type': proj.project_type,
                'No. of Activities': proj.n_activities,
                'Relationships': n_relationships,
                'Search Space Size': f'~{proj.search_space_size:.2e}'
            })
        
        df = pd.DataFrame(records)
        df.to_csv(self.output_dir / 'table_project_characteristics.csv', index=False)
        return df
    
    def performance_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate Table 4: Algorithm Performance Metrics."""
        metrics = ['HV', 'IGD', 'GD', 'Spacing', 'Runtime']
        
        # Group by project and algorithm
        grouped = df.groupby(['Project', 'Algorithm'])
        
        records = []
        for (project, algo), group in grouped:
            record = {'Project': project, 'Algorithm': algo}
            for metric in metrics:
                values = group[metric].dropna()
                if len(values) > 0:
                    record[f'{metric}_mean'] = values.mean()
                    record[f'{metric}_std'] = values.std()
                else:
                    record[f'{metric}_mean'] = np.nan
                    record[f'{metric}_std'] = np.nan
            records.append(record)
        
        result = pd.DataFrame(records)
        result.to_csv(self.output_dir / 'table_performance_metrics.csv', index=False)
        return result
    
    def friedman_rankings(self, df: pd.DataFrame, metrics: List[str] = None) -> pd.DataFrame:
        """Generate Table 5: Friedman Test Rankings."""
        if metrics is None:
            metrics = ['HV', 'IGD', 'Spacing']
        
        records = []
        for metric in metrics:
            result = StatisticalAnalysis.friedman_test(df, metric=metric)
            if 'error' not in result:
                for algo, rank in result['mean_ranks'].items():
                    records.append({
                        'Metric': metric,
                        'Algorithm': algo,
                        'Mean Rank': rank,
                        'p-value': result['p_value'],
                        'Significant': result['significant']
                    })
        
        result_df = pd.DataFrame(records)
        result_df.to_csv(self.output_dir / 'table_friedman_rankings.csv', index=False)
        return result_df
    
    def best_solutions(self, results: List[Dict], mcdm: OPATOPSIS) -> pd.DataFrame:
        """Generate Table 7: Best Solutions per Project."""
        records = []
        
        projects = list(set(r['project'] for r in results))
        
        for project in projects:
            proj_results = [r for r in results if r['project'] == project and r['success']]
            
            # Find best overall solution using TOPSIS
            all_F = []
            all_X = []
            all_algos = []
            
            for r in proj_results:
                if len(r['F']) > 0:
                    all_F.append(r['F'])
                    all_X.append(r['X'])
                    all_algos.extend([r['algorithm']] * len(r['F']))
            
            if all_F:
                combined_F = np.vstack(all_F)
                combined_X = np.vstack(all_X)
                
                cc, best_idx = mcdm.topsis_rank(combined_F)
                best_F = combined_F[best_idx]
                best_X = combined_X[best_idx]
                best_algo = all_algos[best_idx]
                
                # Build record with actual number of objectives
                record = {
                    'Project': project,
                    'Best Algorithm': best_algo,
                    'TOPSIS CC': cc[best_idx],
                }
                
                # 5D objectives: Duration, Cost, Quality, Resources, Weather Risk
                obj_names = ['Z1_Duration', 'Z2_Cost', 'Z3_Quality', 'Z4_Resource', 'Z5_WeatherRisk']
                for i, name in enumerate(obj_names):
                    if i < len(best_F):
                        # Quality (index 2) is negated for maximization
                        if i == 2:
                            record[name] = -best_F[i]
                        else:
                            record[name] = best_F[i]
                
                record['Solution'] = str(list(best_X.astype(int)))
                records.append(record)
        
        result_df = pd.DataFrame(records)
        result_df.to_csv(self.output_dir / 'table_best_solutions.csv', index=False)
        return result_df
    
    def algorithm_ranking_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate overall algorithm ranking summary."""
        algorithms = df['Algorithm'].unique()
        
        records = []
        for algo in algorithms:
            algo_df = df[df['Algorithm'] == algo]
            records.append({
                'Algorithm': algo,
                'Mean HV': algo_df['HV'].mean(),
                'Mean IGD': algo_df['IGD'].mean(),
                'Mean Spacing': algo_df['Spacing'].mean(),
                'Mean Runtime': algo_df['Runtime'].mean(),
                'Success Rate': algo_df['Success'].mean() * 100
            })
        
        result = pd.DataFrame(records)
        result['HV Rank'] = result['Mean HV'].rank(ascending=False)
        result['IGD Rank'] = result['Mean IGD'].rank(ascending=True)
        result['Overall Score'] = (result['HV Rank'] + result['IGD Rank']) / 2
        result = result.sort_values('Overall Score')
        
        result.to_csv(self.output_dir / 'table_algorithm_rankings.csv', index=False)
        return result
    
    def objective_statistics(self, results: List[Dict]) -> pd.DataFrame:
        """Generate Table: Objective Value Statistics per Project."""
        obj_names = ['Z1_Duration', 'Z2_Cost', 'Z3_Quality', 'Z4_Resource', 'Z5_WeatherRisk']
        
        records = []
        projects = list(set(r['project'] for r in results))
        
        for project in projects:
            proj_results = [r for r in results if r['project'] == project and r['success']]
            
            all_F = []
            for r in proj_results:
                if len(r['F']) > 0:
                    all_F.append(r['F'])
            
            if all_F:
                combined_F = np.vstack(all_F)
                # Convert Z3 back to positive
                combined_F[:, 2] = -combined_F[:, 2]
                
                for i, obj in enumerate(obj_names):
                    records.append({
                        'Project': project,
                        'Objective': obj,
                        'Min': combined_F[:, i].min(),
                        'Max': combined_F[:, i].max(),
                        'Mean': combined_F[:, i].mean(),
                        'Std': combined_F[:, i].std(),
                        'Median': np.median(combined_F[:, i])
                    })
        
        result = pd.DataFrame(records)
        result.to_csv(self.output_dir / 'table_objective_statistics.csv', index=False)
        return result
    
    def baseline_comparison(self, projects: List[Project], results: List[Dict],
                           mcdm: OPATOPSIS) -> pd.DataFrame:
        """Generate Table: Comparison with Baseline (all fastest methods)."""
        records = []
        
        for proj in projects:
            # Calculate baseline (Method 0 for all activities - usually fastest)
            baseline_solution = np.zeros(proj.n_activities)
            calc = ObjectiveCalculator(proj)
            baseline_F = calc.evaluate(baseline_solution)
            baseline_F[2] = -baseline_F[2]  # Convert quality back
            
            # Find best optimized solution
            proj_results = [r for r in results if r['project'] == proj.name and r['success']]
            all_F = []
            all_X = []
            for r in proj_results:
                if len(r['F']) > 0:
                    all_F.append(r['F'])
                    all_X.append(r['X'])
            
            if all_F:
                combined_F = np.vstack(all_F)
                combined_X = np.vstack(all_X)
                cc, best_idx = mcdm.topsis_rank(combined_F)
                best_F = combined_F[best_idx]
                best_F[2] = -best_F[2]  # Convert quality back
                
                obj_names = ['Duration', 'Cost', 'Quality', 'Resource', 'WeatherRisk']
                directions = ['min', 'min', 'max', 'min', 'min']
                
                for i, (obj, direction) in enumerate(zip(obj_names, directions)):
                    improvement = ((baseline_F[i] - best_F[i]) / baseline_F[i]) * 100
                    if direction == 'max':
                        improvement = -improvement  # Invert for maximization
                    
                    records.append({
                        'Project': proj.name,
                        'Objective': obj,
                        'Baseline': baseline_F[i],
                        'Optimized': best_F[i],
                        'Improvement (%)': improvement,
                        'Direction': direction
                    })
        
        result = pd.DataFrame(records)
        result.to_csv(self.output_dir / 'table_baseline_comparison.csv', index=False)
        return result
    
    def topsis_rankings_by_scenario(self, results: List[Dict]) -> pd.DataFrame:
        """Generate Table: TOPSIS Rankings under Different Weight Scenarios."""
        scenarios = {
            'Equal': [1, 1, 1, 1, 1],
            'Time-Cost': [1, 2, 5, 4, 3],
            'Quality': [3, 4, 1, 5, 2],
            'WeatherRisk': [2, 3, 4, 5, 1],
            'Resource': [4, 5, 3, 1, 2],
        }
        
        records = []
        projects = list(set(r['project'] for r in results))
        mcdm = OPATOPSIS()
        
        for project in projects:
            proj_results = [r for r in results if r['project'] == project and r['success']]
            
            all_F = []
            all_algos = []
            for r in proj_results:
                if len(r['F']) > 0:
                    all_F.append(r['F'])
                    all_algos.extend([r['algorithm']] * len(r['F']))
            
            if all_F:
                combined_F = np.vstack(all_F)
                
                for scenario_name, priorities in scenarios.items():
                    mcdm.set_priority(priorities)
                    cc, best_idx = mcdm.topsis_rank(combined_F)
                    best_algo = all_algos[best_idx]
                    
                    records.append({
                        'Project': project,
                        'Scenario': scenario_name,
                        'Best Algorithm': best_algo,
                        'TOPSIS CC': cc[best_idx],
                        'Priority': str(priorities)
                    })
        
        result = pd.DataFrame(records)
        result.to_csv(self.output_dir / 'table_topsis_scenarios.csv', index=False)
        return result
    
    def activity_method_frequency(self, results: List[Dict], projects: List[Project]) -> pd.DataFrame:
        """Generate Table: Most Frequently Selected Methods per Activity."""
        records = []
        
        for proj in projects:
            proj_results = [r for r in results if r['project'] == proj.name and r['success']]
            
            all_X = []
            for r in proj_results:
                if len(r['X']) > 0:
                    all_X.append(r['X'])
            
            if all_X:
                combined_X = np.vstack(all_X)
                
                for i, act in enumerate(proj.activities):
                    methods = combined_X[:, i].astype(int)
                    unique, counts = np.unique(methods, return_counts=True)
                    most_common_idx = unique[np.argmax(counts)]
                    most_common_count = counts[np.argmax(counts)]
                    
                    records.append({
                        'Project': proj.name,
                        'Activity': act.name,
                        'Most Selected Method': int(most_common_idx),
                        'Selection Frequency (%)': (most_common_count / len(methods)) * 100,
                        'Total Selections': len(methods)
                    })
        
        result = pd.DataFrame(records)
        result.to_csv(self.output_dir / 'table_method_frequency.csv', index=False)
        return result
    
    def pareto_dominance_summary(self, results: List[Dict]) -> pd.DataFrame:
        """Generate Table: Pareto Dominance Analysis Between Algorithms."""
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
        
        records = []
        projects = list(set(r['project'] for r in results))
        
        for project in projects:
            proj_results = [r for r in results if r['project'] == project and r['success']]
            algorithms = list(set(r['algorithm'] for r in proj_results))
            
            # Get best solution from each algorithm
            algo_best = {}
            for algo in algorithms:
                algo_results = [r for r in proj_results if r['algorithm'] == algo]
                if algo_results:
                    all_F = np.vstack([r['F'] for r in algo_results if len(r['F']) > 0])
                    nds = NonDominatedSorting()
                    fronts = nds.do(all_F)
                    algo_best[algo] = all_F[fronts[0]]
            
            # Count dominance relationships
            for algo in algorithms:
                dominated_by = 0
                dominates = 0
                for other_algo in algorithms:
                    if algo != other_algo and algo in algo_best and other_algo in algo_best:
                        # Simple dominance check (any solution dominates any other)
                        for sol1 in algo_best[algo]:
                            for sol2 in algo_best[other_algo]:
                                if np.all(sol1 <= sol2) and np.any(sol1 < sol2):
                                    dominates += 1
                                if np.all(sol2 <= sol1) and np.any(sol2 < sol1):
                                    dominated_by += 1
                
                records.append({
                    'Project': project,
                    'Algorithm': algo,
                    'Pareto Solutions': len(algo_best.get(algo, [])),
                    'Dominates Others': dominates,
                    'Dominated By Others': dominated_by
                })
        
        result = pd.DataFrame(records)
        result.to_csv(self.output_dir / 'table_dominance_analysis.csv', index=False)
        return result
    
    def latex_performance_table(self, df: pd.DataFrame) -> str:
        """Generate LaTeX formatted performance table."""
        metrics = ['HV', 'IGD', 'Spacing', 'Runtime']
        
        # Aggregate
        agg = df.groupby(['Project', 'Algorithm']).agg({
            m: ['mean', 'std'] for m in metrics
        }).round(4)
        
        # Generate LaTeX
        latex = agg.to_latex(
            caption='Algorithm Performance Metrics (Mean ± Std)',
            label='tab:performance',
            escape=False
        )
        
        with open(self.output_dir / 'table_performance.tex', 'w') as f:
            f.write(latex)
        
        return latex
    
    def percentage_improvement_table(self, projects: List[Project], results: List[Dict]) -> pd.DataFrame:
        """
        Generate Table: Percentage Improvement vs Baseline.
        Compares optimized solutions against baseline (fastest/cheapest methods).
        """
        obj_names = ['Duration (Z1)', 'Cost (Z2)', 'Quality (Z3)', 'Resource (Z4)', 'WeatherRisk (Z5)']
        records = []
        
        for proj in projects:
            # Calculate baseline (all fastest methods = method index 0)
            calculator = ObjectiveCalculator(proj)
            baseline_x = np.zeros(len(proj.activities), dtype=int)
            baseline_F = calculator.evaluate(baseline_x)
            
            # Get all optimized solutions
            proj_results = [r for r in results if r['project'] == proj.name and r['success']]
            
            for algo in list(set(r['algorithm'] for r in proj_results)):
                algo_results = [r for r in proj_results if r['algorithm'] == algo]
                if not algo_results:
                    continue
                    
                # Get best solution (by TOPSIS or first Pareto)
                best_F = None
                best_improvement = float('-inf')
                
                for r in algo_results:
                    if len(r['F']) > 0:
                        # Use mean of Pareto front
                        mean_F = np.mean(r['F'], axis=0)
                        if best_F is None:
                            best_F = mean_F
                        else:
                            # Compare sum of improvements
                            curr_imp = np.sum((baseline_F - mean_F) / (np.abs(baseline_F) + 1e-10))
                            if curr_imp > best_improvement:
                                best_improvement = curr_imp
                                best_F = mean_F
                
                if best_F is not None:
                    record = {'Project': proj.name, 'Algorithm': algo}
                    
                    for i, obj_name in enumerate(obj_names):
                        baseline_val = baseline_F[i]
                        opt_val = best_F[i]
                        
                        # Calculate improvement (positive % means better for ALL objectives)
                        # Since Quality is stored as negative, minimizing the negative value improves quality
                        # Formula: (Baseline - Optimized) / |Baseline|
                        improvement = ((baseline_val - opt_val) / (abs(baseline_val) + 1e-10)) * 100
                        
                        record[f'{obj_name} Improvement (%)'] = round(improvement, 2)
                        
                        # For display, show positive values for Quality
                        if i == 2:
                            record[f'{obj_name} Baseline'] = round(abs(baseline_val), 2)
                            record[f'{obj_name} Optimized'] = round(abs(opt_val), 2)
                        else:
                            record[f'{obj_name} Baseline'] = round(baseline_val, 2)
                            record[f'{obj_name} Optimized'] = round(opt_val, 2)
                    
                    records.append(record)
        
        result = pd.DataFrame(records)
        result.to_csv(self.output_dir / 'table_percentage_improvement.csv', index=False)
        return result
    
    def improvement_summary_by_algorithm(self, projects: List[Project], results: List[Dict]) -> pd.DataFrame:
        """Generate Table: Average % Improvement by Algorithm across all objectives."""
        obj_names = ['Duration', 'Cost', 'Quality', 'Resource', 'WeatherRisk']
        records = []
        
        algorithms = list(set(r['algorithm'] for r in results))
        
        for algo in algorithms:
            algo_improvements = {obj: [] for obj in obj_names}
            
            for proj in projects:
                calculator = ObjectiveCalculator(proj)
                baseline_x = np.zeros(len(proj.activities), dtype=int)
                baseline_F = calculator.evaluate(baseline_x)
                
                proj_results = [r for r in results 
                               if r['project'] == proj.name and r['algorithm'] == algo and r['success']]
                
                for r in proj_results:
                    if len(r['F']) > 0:
                        for sol_F in r['F']:
                            for i, obj in enumerate(obj_names):
                                # Unified improvement formula (works for minimized and negated-maximized)
                                imp = ((baseline_F[i] - sol_F[i]) / (abs(baseline_F[i]) + 1e-10)) * 100
                                algo_improvements[obj].append(imp)
            
            record = {'Algorithm': algo}
            total_avg = []
            for obj in obj_names:
                if algo_improvements[obj]:
                    avg = np.mean(algo_improvements[obj])
                    record[f'{obj} Avg Imp (%)'] = round(avg, 2)
                    total_avg.append(avg)
                else:
                    record[f'{obj} Avg Imp (%)'] = 0
            
            record['Overall Avg (%)'] = round(np.mean(total_avg), 2) if total_avg else 0
            records.append(record)
        
        result = pd.DataFrame(records)
        result = result.sort_values('Overall Avg (%)', ascending=False)
        result.to_csv(self.output_dir / 'table_improvement_by_algorithm.csv', index=False)
        return result
    
    def algorithm_pairwise_wins(self, df: pd.DataFrame, metric: str = 'HV') -> pd.DataFrame:
        """Generate Table: Pairwise Win/Tie/Loss matrix between algorithms."""
        algorithms = df['Algorithm'].unique()
        projects = df['Project'].unique()
        
        wins = pd.DataFrame(0, index=algorithms, columns=algorithms)
        
        for proj in projects:
            proj_df = df[df['Project'] == proj]
            
            for algo1 in algorithms:
                for algo2 in algorithms:
                    if algo1 == algo2:
                        continue
                    
                    mean1 = proj_df[proj_df['Algorithm'] == algo1][metric].mean()
                    mean2 = proj_df[proj_df['Algorithm'] == algo2][metric].mean()
                    
                    # For HV, higher is better
                    if metric in ['HV']:
                        if mean1 > mean2:
                            wins.loc[algo1, algo2] += 1
                    else:  # For IGD, Spacing, lower is better
                        if mean1 < mean2:
                            wins.loc[algo1, algo2] += 1
        
        wins.to_csv(self.output_dir / f'table_pairwise_wins_{metric}.csv')
        return wins
    
    def best_solution_characteristics(self, results: List[Dict], projects: List[Project], 
                                      mcdm: 'OPATOPSIS') -> pd.DataFrame:
        """Generate Table: Characteristics of the best solution for each project."""
        obj_names = ['Duration', 'Cost', 'Quality', 'Resource', 'WeatherRisk']  # 5D Framework
        records = []
        
        for proj in projects:
            proj_results = [r for r in results if r['project'] == proj.name and r['success']]
            
            # Combine all solutions
            all_F = []
            all_X = []
            all_algos = []
            
            for r in proj_results:
                if len(r['F']) > 0:
                    for i, f in enumerate(r['F']):
                        all_F.append(f)
                        all_X.append(r['X'][i])
                        all_algos.append(r['algorithm'])
            
            if all_F:
                all_F = np.array(all_F)
                
                # Find best by TOPSIS
                cc, best_idx = mcdm.topsis_rank(all_F)
                best_F = all_F[best_idx]
                best_X = all_X[best_idx]
                best_algo = all_algos[best_idx]
                
                record = {
                    'Project': proj.name,
                    'Best Algorithm': best_algo,
                    'TOPSIS Score': round(cc[best_idx], 4)
                }
                
                for i, obj in enumerate(obj_names):
                    if i == 2:  # Quality
                        record[obj] = round(abs(best_F[i]), 2)
                    else:
                        record[obj] = round(best_F[i], 2)
                
                # Method selection summary
                method_counts = np.bincount(best_X.astype(int), minlength=3)
                record['Method 0 Count'] = method_counts[0] if len(method_counts) > 0 else 0
                record['Method 1 Count'] = method_counts[1] if len(method_counts) > 1 else 0
                record['Method 2 Count'] = method_counts[2] if len(method_counts) > 2 else 0
                
                records.append(record)
        
        result = pd.DataFrame(records)
        result.to_csv(self.output_dir / 'table_best_solution_characteristics.csv', index=False)
        return result
    
    def computational_efficiency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate Table: Computational Efficiency Analysis."""
        records = []
        
        for algo in df['Algorithm'].unique():
            algo_df = df[df['Algorithm'] == algo]
            
            record = {
                'Algorithm': algo,
                'Mean Runtime (s)': round(algo_df['Runtime'].mean(), 2),
                'Std Runtime (s)': round(algo_df['Runtime'].std(), 2),
                'Min Runtime (s)': round(algo_df['Runtime'].min(), 2),
                'Max Runtime (s)': round(algo_df['Runtime'].max(), 2),
                'Mean Solutions': round(algo_df['N_Solutions'].mean(), 1),
                'Solutions/Second': round(algo_df['N_Solutions'].sum() / algo_df['Runtime'].sum(), 2),
                'Mean HV': f"{algo_df['HV'].mean():.2e}",
                'HV per Second': f"{algo_df['HV'].mean() / algo_df['Runtime'].mean():.2e}"
            }
            records.append(record)
        
        result = pd.DataFrame(records)
        result = result.sort_values('Mean Runtime (s)')
        result.to_csv(self.output_dir / 'table_computational_efficiency.csv', index=False)
        return result
    
    def solution_diversity_analysis(self, results: List[Dict]) -> pd.DataFrame:
        """Generate Table: Solution Diversity Analysis per Algorithm."""
        records = []
        
        projects = list(set(r['project'] for r in results))
        algorithms = list(set(r['algorithm'] for r in results))
        
        for algo in algorithms:
            algo_record = {'Algorithm': algo}
            
            total_solutions = 0
            total_unique = 0
            total_spread = []
            
            for proj in projects:
                proj_results = [r for r in results 
                               if r['project'] == proj and r['algorithm'] == algo and r['success']]
                
                all_F = []
                for r in proj_results:
                    if len(r['F']) > 0:
                        all_F.extend(r['F'].tolist())
                
                if all_F:
                    all_F = np.array(all_F)
                    total_solutions += len(all_F)
                    
                    # Count unique solutions (within tolerance)
                    unique_count = len(np.unique(np.round(all_F, 4), axis=0))
                    total_unique += unique_count
                    
                    # Calculate spread (range for each objective)
                    spreads = np.ptp(all_F, axis=0)  # Peak to peak (max - min)
                    total_spread.append(np.mean(spreads))
            
            algo_record['Total Solutions'] = total_solutions
            algo_record['Unique Solutions'] = total_unique
            algo_record['Uniqueness (%)'] = round(100 * total_unique / max(total_solutions, 1), 1)
            algo_record['Mean Spread'] = round(np.mean(total_spread), 2) if total_spread else 0
            
            records.append(algo_record)
        
        result = pd.DataFrame(records)
        result = result.sort_values('Uniqueness (%)', ascending=False)
        result.to_csv(self.output_dir / 'table_solution_diversity.csv', index=False)
        return result
    
    def objective_extremes(self, results: List[Dict]) -> pd.DataFrame:
        """Generate Table: Best and Worst values found for each objective."""
        obj_names = ['Duration', 'Cost', 'Quality', 'Resource', 'WeatherRisk']  # 5D Framework
        records = []
        
        projects = list(set(r['project'] for r in results))
        
        for proj in projects:
            proj_results = [r for r in results if r['project'] == proj and r['success']]
            
            all_F = []
            for r in proj_results:
                if len(r['F']) > 0:
                    all_F.extend(r['F'].tolist())
            
            if all_F:
                all_F = np.array(all_F)
                
                for i, obj in enumerate(obj_names):
                    record = {
                        'Project': proj,
                        'Objective': obj,
                        # For Quality (i=2), min stored value = best real value (since stored is negative)
                        # We want to display positive values
                        'Best Value': round(abs(np.min(all_F[:, i])), 2) if i == 2 else round(np.min(all_F[:, i]), 2),
                        'Worst Value': round(abs(np.max(all_F[:, i])), 2) if i == 2 else round(np.max(all_F[:, i]), 2),
                        'Mean Value': round(np.mean(all_F[:, i]), 2),
                        'Std Value': round(np.std(all_F[:, i]), 2),
                        'Range': round(np.ptp(all_F[:, i]), 2)
                    }
                    records.append(record)
        
        result = pd.DataFrame(records)
        result.to_csv(self.output_dir / 'table_objective_extremes.csv', index=False)
        return result


# =============================================================================
# PART 13: MAIN EXECUTION
# =============================================================================

def run_test_mode():
    """Run quick test with reduced parameters."""
    print("=" * 60)
    print("RUNNING IN TEST MODE (reduced parameters)")
    print("=" * 60)
    
    # Create projects
    projects = [create_highway_project()]  # Only one project for testing
    
    # Use subset of algorithms
    algo_names = ["NSGA-II", "NSGA-III", "AGE-MOEA"]
    
    # Run with reduced parameters
    results = run_parallel_optimization(
        projects=projects,
        algo_names=algo_names,
        n_runs=3,
        n_jobs=1,  # Sequential for testing
        pop_size=50,
        n_gen=50
    )
    
    return results, projects


def run_full_experiment(n_runs: int = 30, n_jobs: int = -1,
                       pop_size: int = 100, n_gen: int = 200):
    """
    Run full experimental design with 5D Fuzzy Weather-Sensitive Framework.
    
    Uses NSGA-III algorithm to optimize 3 infrastructure projects with
    varying weather sensitivities and complex activity dependencies:
    - Metro Station: 12 activities, mixed indoor/outdoor
    - Highway Interchange: 10 activities, parallel ramps with convergence
    - Pipeline Network: 8 activities, parallel excavation branches
    """
    print("=" * 60)
    print("5D FUZZY WEATHER-SENSITIVE OPTIMIZATION FRAMEWORK")
    print("Infrastructure Projects with Complex Dependencies")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Algorithm: NSGA-III (Energy-based ref dirs)")
    print(f"  Population Size: {pop_size}")
    print(f"  Generations: {n_gen}")
    print(f"  Runs per config: {n_runs}")
    print(f"  Parallel jobs: {n_jobs}")
    print(f"  Objectives: Z1-Z5 (Duration, Cost, Quality, Resources, Weather)")
    print("=" * 60)
    
    # Create all 3 infrastructure projects
    projects = [
        create_metro_project(),               # 12 activities, sequential + parallel
        create_highway_interchange_project(), # 10 activities, complex convergence
        create_pipeline_project()             # 8 activities, parallel branches
    ]
    
    # Print project info
    for proj in projects:
        print(f"\n{proj.name}:")
        print(f"  Activities: {proj.n_activities}")
        print(f"  Search Space: {proj.search_space_size:,.0f}")
    
    # Run optimization
    print("\n" + "=" * 60)
    start_time = time.time()
    
    results = run_parallel_optimization(
        projects=projects,
        algo_names=ALGORITHM_NAMES,  # Only NSGA-III
        n_runs=n_runs,
        n_jobs=n_jobs,
        pop_size=pop_size,
        n_gen=n_gen
    )
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time/60:.2f} minutes")
    
    return results, projects


def generate_outputs(results: List[Dict], projects: List[Project], output_dir: str = 'results'):
    """Generate all tables and figures."""
    print("\n" + "=" * 60)
    print("GENERATING OUTPUTS")
    print("=" * 60)
    
    # Aggregate results
    print("Aggregating results...")
    df = aggregate_results(results)
    
    # Initialize generators
    viz = Visualizer(output_dir)
    tables = TableGenerator(output_dir)
    mcdm = OPATOPSIS()
    
    # ===========================================
    # TABLES
    # ===========================================
    print("Generating tables...")
    
    # Core tables
    tables.project_characteristics(projects)
    tables.performance_metrics(df)
    tables.friedman_rankings(df)
    tables.best_solutions(results, mcdm)
    tables.algorithm_ranking_summary(df)
    
    # Additional tables
    tables.objective_statistics(results)
    tables.baseline_comparison(projects, results, mcdm)
    tables.topsis_rankings_by_scenario(results)
    tables.activity_method_frequency(results, projects)
    tables.pareto_dominance_summary(results)
    tables.latex_performance_table(df)
    
    # NEW: Percentage improvement tables
    print("  Generating improvement analysis tables...")
    tables.percentage_improvement_table(projects, results)
    tables.improvement_summary_by_algorithm(projects, results)
    tables.algorithm_pairwise_wins(df, 'HV')
    tables.algorithm_pairwise_wins(df, 'IGD')
    tables.best_solution_characteristics(results, projects, mcdm)
    tables.computational_efficiency(df)
    tables.solution_diversity_analysis(results)
    tables.objective_extremes(results)
    
    # Wilcoxon pairwise
    wilcoxon_df = StatisticalAnalysis.wilcoxon_pairwise(df, metric='HV')
    wilcoxon_df.to_csv(Path(output_dir) / 'table_wilcoxon_pairwise.csv')
    
    # ===========================================
    # FIGURES
    # ===========================================
    print("Generating figures...")
    
    # Conceptual figures
    viz.plot_weather_concept()
    
    # Performance comparison figures
    viz.plot_convergence(results)
    viz.plot_boxplots(df, 'HV')
    viz.plot_boxplots(df, 'Spacing')
    viz.plot_boxplots(df, 'IGD')
    viz.plot_algorithm_ranking(df, 'HV')
    viz.plot_algorithm_ranking(df, 'IGD')
    viz.plot_runtime_comparison(df)
    viz.plot_convergence_curves(df)
    
    # Statistical figures
    viz.plot_heatmap(wilcoxon_df, title='Wilcoxon Pairwise p-values (HV)',
                    filename='fig_wilcoxon_hv.png')
    
    # Per-project figures
    for proj in projects:
        print(f"  Generating figures for {proj.name}...")
        
        # Pareto front visualizations (5D objectives)
        viz.plot_pareto_fronts(results, proj.name, objectives=(0, 1))  # Duration vs Cost
        viz.plot_pareto_fronts(results, proj.name, objectives=(0, 4))  # Duration vs Weather
        viz.plot_pareto_fronts(results, proj.name, objectives=(1, 2))  # Cost vs Quality
        viz.plot_pareto_fronts(results, proj.name, objectives=(3, 4))  # Resources vs Weather
        
        # 5D Pareto Matrix (key figure for manuscript)
        viz.plot_pareto_matrix(results, proj.name)
        
        # 3D Pareto plots (5D combinations)
        viz.plot_pareto_3d(results, proj.name, objectives=(0, 1, 4))  # Duration-Cost-Weather
        viz.plot_pareto_3d(results, proj.name, objectives=(0, 2, 4))  # Duration-Quality-Weather
        
        # Parallel coordinates
        viz.plot_parallel_coordinates(results, proj.name)
        
        # Objective correlation
        viz.plot_objective_correlation(results, proj.name)
        
        # Algorithm radar chart
        viz.plot_radar(df, proj.name)
        
        # Solution distribution
        viz.plot_solution_distribution(results, proj.name)
        
        # TOPSIS sensitivity analysis
        viz.plot_topsis_sensitivity(results, proj.name)
    
    # ===========================================
    # RAW DATA EXPORT
    # ===========================================
    print("Saving raw data...")
    df.to_csv(Path(output_dir) / 'results_all_runs.csv', index=False)
    
    # Save Pareto fronts
    pareto_data = {}
    for r in results:
        if r['success'] and len(r['F']) > 0:
            key = f"{r['project']}_{r['algorithm']}_{r['seed']}"
            pareto_data[key] = {
                'F': r['F'].tolist(),
                'X': r['X'].tolist() if hasattr(r['X'], 'tolist') else r['X']
            }
    
    with open(Path(output_dir) / 'pareto_fronts.json', 'w') as f:
        json.dump(pareto_data, f, indent=2)
    
    # Print summary of outputs
    print(f"\n{'='*60}")
    print("OUTPUT SUMMARY")
    print(f"{'='*60}")
    print(f"Tables generated: 20")
    print(f"Figures generated: {8 + len(projects) * 11}")
    print(f"Data files: 2 (CSV + JSON)")
    print(f"\nAll outputs saved to: {output_dir}/")
    print("=" * 60)
    
    return df


def main(test_mode: bool = False, parallel: bool = True, n_jobs: int = -1,
         n_runs: int = 30, pop_size: int = 100, n_gen: int = 200,
         output_dir: str = 'results'):
    """
    Main entry point. Can be called directly with parameters or via command line.
    
    Args:
        test_mode: Run quick test with reduced parameters
        parallel: Enable parallel execution
        n_jobs: Number of parallel jobs (-1 for all cores)
        n_runs: Number of independent runs per configuration
        pop_size: Population size
        n_gen: Number of generations
        output_dir: Output directory for results
        
    Returns:
        Tuple of (results list, aggregated DataFrame)
    """
    # Update config
    CONFIG['n_jobs'] = n_jobs if parallel else 1
    CONFIG['n_runs'] = n_runs
    CONFIG['pop_size'] = pop_size
    CONFIG['n_gen'] = n_gen
    CONFIG['output_dir'] = output_dir
    
    # Run experiment
    if test_mode:
        results, projects = run_test_mode()
    else:
        results, projects = run_full_experiment(
            n_runs=n_runs,
            n_jobs=CONFIG['n_jobs'],
            pop_size=pop_size,
            n_gen=n_gen
        )
    
    # Generate outputs
    df = generate_outputs(results, projects, output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total runs completed: {len(results)}")
    print(f"Successful runs: {sum(1 for r in results if r['success'])}")
    
    # Best algorithm per project
    print("\nBest Algorithm per Project (by mean HV):")
    for proj in df['Project'].unique():
        proj_df = df[df['Project'] == proj]
        best = proj_df.groupby('Algorithm')['HV'].mean().idxmax()
        best_hv = proj_df.groupby('Algorithm')['HV'].mean().max()
        print(f"  {proj}: {best} (HV = {best_hv:.4e})")
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    
    return results, df


def run_cli():
    """Command-line interface entry point."""
    import argparse
    import sys
    
    # Check if running in Jupyter/Colab
    if 'ipykernel' in sys.modules or 'google.colab' in sys.modules:
        print("Detected Jupyter/Colab environment.")
        print("Use main() function directly with parameters instead of CLI.")
        print("\nExample usage:")
        print("  results, df = main(test_mode=True)")
        print("  results, df = main(n_runs=30, parallel=True, n_jobs=-1)")
        return None, None
    
    parser = argparse.ArgumentParser(description='7D-MOPSP Optimization Framework')
    parser.add_argument('--test-mode', action='store_true', 
                       help='Run quick test with reduced parameters')
    parser.add_argument('--parallel', action='store_true',
                       help='Enable parallel execution')
    parser.add_argument('--n-jobs', type=int, default=-1,
                       help='Number of parallel jobs (-1 for all cores)')
    parser.add_argument('--n-runs', type=int, default=30,
                       help='Number of independent runs per configuration')
    parser.add_argument('--pop-size', type=int, default=100,
                       help='Population size')
    parser.add_argument('--n-gen', type=int, default=200,
                       help='Number of generations')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    return main(
        test_mode=args.test_mode,
        parallel=args.parallel,
        n_jobs=args.n_jobs,
        n_runs=args.n_runs,
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        output_dir=args.output_dir
    )


# =============================================================================
# CONVENIENCE FUNCTIONS FOR INTERACTIVE USE
# =============================================================================

def quick_test():
    """Run a quick test with minimal parameters for verification."""
    return main(test_mode=True)


def run_experiment(n_runs: int = 30, parallel: bool = True, n_jobs: int = -1,
                  pop_size: int = 100, n_gen: int = 200):
    """
    Run the full experiment with specified parameters.
    
    Convenience function for interactive use in Jupyter/Colab.
    """
    return main(
        test_mode=False,
        parallel=parallel,
        n_jobs=n_jobs,
        n_runs=n_runs,
        pop_size=pop_size,
        n_gen=n_gen
    )


if __name__ == "__main__":
    run_cli()

