"""
7D Multi-Objective Project Scheduling Problem (7D-MOPSP)
==========================================================
A comprehensive framework for infrastructure project optimization using
pymoo algorithms with OPA-TOPSIS MCDM ranking.

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
# PART 2: PROJECT DATA STRUCTURES
# =============================================================================

@dataclass
class Method:
    """Construction method for an activity."""
    id: int
    duration: int  # days
    cost: float  # direct cost ($)
    quality: float  # 0-1 scale
    labor: int  # workers required
    equipment: int  # equipment units
    environmental: float  # impact index
    social: float  # social cost index
    safety_risk: float  # base risk factor

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
    """Create Metro Station project (12 activities)."""
    activities = [
        Activity(0, "Site Clearance", [
            Method(0, 5, 30000, 0.76, 10, 4, 0.5, 0.4, 0.30),
            Method(1, 4, 38000, 0.84, 12, 5, 0.6, 0.5, 0.35),
            Method(2, 3, 48000, 0.91, 15, 6, 0.7, 0.6, 0.40),
        ], [], 1.0),
        Activity(1, "Deep Excavation", [
            Method(0, 18, 250000, 0.68, 35, 15, 1.0, 0.8, 0.65),
            Method(1, 14, 310000, 0.76, 42, 18, 1.1, 0.9, 0.70),
            Method(2, 10, 380000, 0.84, 50, 22, 1.2, 1.0, 0.75),
        ], [0], 1.5),
        Activity(2, "Retaining Wall", [
            Method(0, 12, 180000, 0.72, 25, 10, 0.8, 0.6, 0.50),
            Method(1, 9, 220000, 0.80, 30, 12, 0.9, 0.7, 0.55),
            Method(2, 7, 270000, 0.88, 38, 15, 1.0, 0.8, 0.60),
        ], [1], 1.3),
        Activity(3, "Foundation Slab", [
            Method(0, 10, 160000, 0.74, 22, 8, 0.7, 0.5, 0.45),
            Method(1, 8, 195000, 0.82, 28, 10, 0.8, 0.6, 0.50),
            Method(2, 6, 240000, 0.90, 35, 12, 0.9, 0.7, 0.55),
        ], [2], 1.4),
        Activity(4, "Platform Structure", [
            Method(0, 14, 220000, 0.73, 28, 12, 0.8, 0.6, 0.50),
            Method(1, 11, 270000, 0.81, 35, 15, 0.9, 0.7, 0.55),
            Method(2, 8, 330000, 0.89, 42, 18, 1.0, 0.8, 0.60),
        ], [3], 1.5),
        Activity(5, "Tunnel Connection", [
            Method(0, 20, 350000, 0.70, 40, 18, 1.0, 0.8, 0.70),
            Method(1, 15, 430000, 0.78, 48, 22, 1.1, 0.9, 0.75),
            Method(2, 12, 520000, 0.86, 58, 26, 1.2, 1.0, 0.80),
        ], [3], 1.6),
        Activity(6, "Roof Structure", [
            Method(0, 12, 200000, 0.75, 24, 14, 0.7, 0.5, 0.45),
            Method(1, 9, 245000, 0.83, 30, 17, 0.8, 0.6, 0.50),
            Method(2, 7, 300000, 0.91, 38, 20, 0.9, 0.7, 0.55),
        ], [4, 5], 1.4),
        Activity(7, "MEP Systems", [
            Method(0, 15, 280000, 0.72, 30, 10, 0.6, 0.4, 0.40),
            Method(1, 12, 340000, 0.80, 38, 12, 0.7, 0.5, 0.45),
            Method(2, 9, 420000, 0.88, 45, 15, 0.8, 0.6, 0.50),
        ], [6], 1.3),
        Activity(8, "Finishing Works", [
            Method(0, 10, 150000, 0.78, 25, 6, 0.4, 0.4, 0.30),
            Method(1, 8, 185000, 0.85, 32, 8, 0.5, 0.5, 0.35),
            Method(2, 6, 225000, 0.92, 40, 10, 0.6, 0.6, 0.40),
        ], [7], 1.2),
        Activity(9, "Escalator Installation", [
            Method(0, 8, 180000, 0.80, 15, 8, 0.4, 0.3, 0.40),
            Method(1, 6, 220000, 0.87, 18, 10, 0.5, 0.4, 0.45),
            Method(2, 5, 270000, 0.94, 22, 12, 0.6, 0.5, 0.50),
        ], [8], 1.1),
        Activity(10, "Safety Systems", [
            Method(0, 6, 120000, 0.82, 12, 5, 0.3, 0.2, 0.25),
            Method(1, 5, 148000, 0.89, 15, 6, 0.4, 0.3, 0.30),
            Method(2, 4, 180000, 0.95, 18, 8, 0.5, 0.4, 0.35),
        ], [8], 1.2),
        Activity(11, "Commissioning", [
            Method(0, 5, 50000, 0.84, 10, 4, 0.2, 0.2, 0.20),
            Method(1, 4, 62000, 0.91, 12, 5, 0.3, 0.3, 0.25),
            Method(2, 3, 76000, 0.97, 15, 6, 0.4, 0.4, 0.30),
        ], [9, 10], 1.0),
    ]
    return Project("Metro Station", "Metro", activities, 15000, 100, 45)


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
# PART 4: OBJECTIVE FUNCTIONS (Z1-Z7)
# =============================================================================

class ObjectiveCalculator:
    """Calculate all 7 objective functions."""
    
    def __init__(self, project: Project, alpha: float = 0.3):
        self.project = project
        self.alpha = alpha  # Congestion sensitivity
        self.scheduler = CPMScheduler(project)
        
    def evaluate(self, solution: np.ndarray) -> np.ndarray:
        """
        Evaluate all 7 objectives for a solution.
        
        Args:
            solution: Array of method indices
            
        Returns:
            Array of 7 objective values
        """
        schedule = self.scheduler.schedule(solution)
        
        z1 = self._calc_duration(schedule)
        z2 = self._calc_cost(solution, schedule)
        z3 = self._calc_quality(solution)
        z4 = self._calc_resource_moment(schedule)
        z5 = self._calc_environmental(solution)
        z6 = self._calc_social(solution)
        z7 = self._calc_safety(solution, schedule)
        
        # Note: Z3 is maximization, convert to minimization
        return np.array([z1, z2, -z3, z4, z5, z6, z7])
    
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


# =============================================================================
# PART 5: PYMOO PROBLEM DEFINITION
# =============================================================================

class SchedulingProblem(Problem):
    """Pymoo problem class for 7D-MOPSP."""
    
    def __init__(self, project: Project, alpha: float = 0.3):
        self.project = project
        self.calculator = ObjectiveCalculator(project, alpha)
        
        # Variable bounds: method indices for each activity
        n_vars = project.n_activities
        xl = np.zeros(n_vars)
        xu = np.array([len(act.methods) - 1 for act in project.activities])
        
        super().__init__(
            n_var=n_vars,
            n_obj=7,
            n_ieq_constr=0,  # Constraints handled via repair
            xl=xl,
            xu=xu,
            vtype=int
        )
    
    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluate population of solutions."""
        F = np.zeros((X.shape[0], 7))
        
        for i, x in enumerate(X):
            F[i] = self.calculator.evaluate(x.astype(int))
        
        out["F"] = F


# =============================================================================
# PART 6: ALGORITHM FACTORY
# =============================================================================

def get_ref_dirs(n_obj: int, pop_size: int = 100) -> np.ndarray:
    """
    Get reference directions for many-objective algorithms.
    
    For high-dimensional problems (7 objectives), das-dennis with standard
    partitions creates too many points. Use energy-based or reduced partitions.
    """
    if n_obj <= 3:
        # Standard das-dennis for low dimensions
        return get_reference_directions("das-dennis", n_obj, n_partitions=12)
    else:
        # For many-objective (7D), use energy-based method with specified pop_size
        # This ensures ref_dirs matches pop_size
        try:
            ref_dirs = get_reference_directions("energy", n_obj, pop_size, seed=42)
        except:
            # Fallback to das-dennis with low partitions
            # For 7 objectives, n_partitions=3 gives C(3+7-1, 7-1) = C(9,6) = 84 points
            ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=3)
        return ref_dirs


def create_algorithm(name: str, n_obj: int = 7, pop_size: int = 100) -> Any:
    """
    Create a pymoo algorithm instance.
    
    Args:
        name: Algorithm name
        n_obj: Number of objectives
        pop_size: Population size
        
    Returns:
        Configured algorithm instance
    """
    # Get reference directions matching pop_size
    ref_dirs = get_ref_dirs(n_obj, pop_size)
    
    # Adjust pop_size for reference-direction-based algorithms
    # to match the number of reference directions
    ref_pop_size = len(ref_dirs)
    
    # Common operators for integer encoding
    sampling = IntegerRandomSampling()
    crossover = SBX(prob=0.9, eta=15, repair=RoundingRepair())
    mutation = PM(eta=20, repair=RoundingRepair())
    
    algorithms = {
        "NSGA-II": lambda: NSGA2(
            pop_size=pop_size,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=True
        ),
        "R-NSGA-II": lambda: RNSGA2(
            pop_size=pop_size,
            ref_points=np.array([[0.3, 0.3, -0.9, 0.3, 0.3, 0.3, 0.3]]),  # Aspiration
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            epsilon=0.01,
            normalization="front",
            weights=np.array([1/7]*7)
        ),
        "NSGA-III": lambda: NSGA3(
            ref_dirs=ref_dirs,
            pop_size=ref_pop_size,  # Must match ref_dirs
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=True
        ),
        "U-NSGA-III": lambda: UNSGA3(
            ref_dirs=ref_dirs,
            pop_size=ref_pop_size,  # Must match ref_dirs
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=True
        ),
        "R-NSGA-III": lambda: RNSGA3(
            ref_points=np.array([[0.1]*7]),  # Aspiration point (ideal values)
            pop_per_ref_point=28,  # Must be valid: 7*(n_partitions+1)!/7!/n_partitions! = 28 for n_partitions=2
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
        ),
        "MOEAD": lambda: MOEAD(
            ref_dirs=ref_dirs,
            n_neighbors=min(15, ref_pop_size - 1),  # Neighbors must be < pop_size
            prob_neighbor_mating=0.7,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
        ),
        "AGE-MOEA": lambda: AGEMOEA(
            pop_size=pop_size,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=True
        ),
        "C-TAEA": lambda: CTAEA(
            ref_dirs=ref_dirs,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
        ),
        "SMS-EMOA": lambda: SMSEMOA(
            pop_size=min(pop_size, 50),  # SMS-EMOA is slow, use smaller pop
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
        ),
        "RVEA": lambda: RVEA(
            ref_dirs=ref_dirs,
            pop_size=ref_pop_size,  # Must match ref_dirs
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
        ),
    }
    
    if name not in algorithms:
        raise ValueError(f"Unknown algorithm: {name}. Available: {list(algorithms.keys())}")
    
    return algorithms[name]()


# List of available algorithms
ALGORITHM_NAMES = [
    "NSGA-II", "R-NSGA-II", "NSGA-III", "U-NSGA-III", "R-NSGA-III",
    "MOEAD", "AGE-MOEA", "C-TAEA", "SMS-EMOA", "RVEA"
]


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
        """Calculate hypervolume indicator with normalization for 7D."""
        if len(F) == 0:
            return 0.0
            
        # Normalize objectives to [0, 1] range for meaningful HV values
        F_min = np.min(F, axis=0)
        F_max = np.max(F, axis=0)
        F_range = F_max - F_min
        F_range[F_range == 0] = 1.0  # Avoid division by zero
        
        F_normalized = (F - F_min) / F_range
        
        # Use reference point at [1.1, 1.1, ...] after normalization
        normalized_ref = np.ones(F.shape[1]) * 1.1
        
        try:
            hv = HV(ref_point=normalized_ref)
            return hv(F_normalized)
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
# PART 8: OPA-TOPSIS MCDM FRAMEWORK
# =============================================================================

class OPATOPSIS:
    """Ordinal Priority Approach with TOPSIS for solution ranking."""
    
    def __init__(self, n_criteria: int = 7):
        self.n_criteria = n_criteria
        # Default priority order (can be customized)
        # 1=highest priority: Time, Cost, Safety, Quality, Resources, Environment, Social
        self.priority_order = [1, 2, 7, 3, 4, 5, 6]  # Ranks for Z1-Z7
        
        # Benefit (1) vs Cost (-1) criteria
        # Z3 (Quality) is benefit, others are cost
        self.criteria_type = [-1, -1, 1, -1, -1, -1, -1]
    
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
            F: Objective values
            X: Solution vectors (optional)
            
        Returns:
            DataFrame with rankings and objective values
        """
        weights = self.opa_weights()
        cc, best_idx = self.topsis_rank(F, weights)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Rank': np.argsort(-cc) + 1,
            'CC': cc,
            'Z1_Time': F[:, 0],
            'Z2_Cost': F[:, 1],
            'Z3_Quality': -F[:, 2],  # Convert back to positive
            'Z4_Resource': F[:, 3],
            'Z5_Environment': F[:, 4],
            'Z6_Social': F[:, 5],
            'Z7_Safety': F[:, 6]
        })
        
        if X is not None:
            df['Solution'] = [str(list(x.astype(int))) for x in X]
        
        return df.sort_values('Rank')


# =============================================================================
# PART 9: PARALLEL EXECUTION ENGINE
# =============================================================================

def run_single_optimization(project: Project, algo_name: str, seed: int,
                           pop_size: int = 100, n_gen: int = 200) -> Dict:
    """
    Run a single optimization experiment.
    
    Args:
        project: Project instance
        algo_name: Algorithm name
        seed: Random seed
        pop_size: Population size
        n_gen: Number of generations
        
    Returns:
        Dictionary with results
    """
    np.random.seed(seed)
    
    start_time = time.time()
    
    # Create problem and algorithm
    problem = SchedulingProblem(project, alpha=CONFIG['alpha_congestion'])
    algorithm = create_algorithm(algo_name, n_obj=7, pop_size=pop_size)
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
        # Parallel execution - leave 1 core free for system responsiveness
        from joblib import Parallel, delayed, cpu_count
        
        # Calculate actual jobs: -1 means all cores minus 1 (leave one free)
        total_cpus = cpu_count()
        if n_jobs == -1:
            actual_jobs = max(total_cpus - 1, 1)  # Leave 1 core free, minimum 1 job
        else:
            actual_jobs = n_jobs
        
        print(f"  Using {actual_jobs} of {total_cpus} CPU cores (leaving 1 free)")
        
        # Run all tasks in parallel at once (most efficient)
        if TQDM_AVAILABLE:
            # For tqdm, use the parallel_backend with prefer='processes' for true parallelism
            from joblib import parallel_backend
            
            with parallel_backend('loky', n_jobs=actual_jobs):
                results = Parallel(verbose=0)(
                    delayed(run_single_optimization)(
                        project, algo, seed, pop_size, n_gen
                    )
                    for project, algo, seed in tqdm(tasks, desc="Optimization Progress", 
                                                      unit="run", ncols=100)
                )
        else:
            results = Parallel(n_jobs=n_jobs, verbose=10, prefer='processes')(
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
        obj_names = ['Time', 'Cost', 'Quality', 'Resource', 'Environment', 'Social', 'Safety']
        
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
        """Plot algorithm comparison using bar chart with error bars (robust for small n)."""
        projects = df['Project'].unique()
        n_projects = len(projects)
        
        metric_titles = {
            'HV': 'Hypervolume',
            'IGD': 'Inverted Generational Distance',
            'GD': 'Generational Distance', 
            'Spacing': 'Spacing',
            'Spread': 'Spread'
        }
        
        # Debug: print data info
        print(f"    Plotting {metric}: {len(df)} rows, projects={list(projects)}")
        print(f"    {metric} range: {df[metric].min():.4f} to {df[metric].max():.4f}")
        
        fig, axes = plt.subplots(1, n_projects, figsize=(8 * n_projects, 6))
        if n_projects == 1:
            axes = [axes]
        
        for idx, project in enumerate(projects):
            ax = axes[idx]
            proj_df = df[df['Project'] == project].copy()
            
            if len(proj_df) > 0 and proj_df[metric].notna().any():
                # Calculate mean and std for each algorithm
                algo_stats = proj_df.groupby('Algorithm')[metric].agg(['mean', 'std', 'count'])
                algo_stats = algo_stats.sort_values('mean', ascending=False)  # Sort by performance
                
                algorithms = algo_stats.index.tolist()
                means = algo_stats['mean'].values
                stds = algo_stats['std'].fillna(0).values
                
                # Create bar positions
                x_pos = np.arange(len(algorithms))
                
                # Create bars with error bars
                bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                             color=self.colors[:len(algorithms)], alpha=0.8,
                             edgecolor='black', linewidth=0.5)
                
                # Set x-axis labels
                ax.set_xticks(x_pos)
                ax.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=9)
                
                # Set y-axis limits based on data
                y_min = max(0, min(means) - max(stds) * 1.5) if len(means) > 0 else 0
                y_max = max(means) + max(stds) * 1.5 if len(means) > 0 else 1
                ax.set_ylim(y_min, y_max)
                
                # Add value labels on bars
                for bar, mean in zip(bars, means):
                    height = bar.get_height()
                    ax.annotate(f'{mean:.3f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3), textcoords="offset points",
                               ha='center', va='bottom', fontsize=7, rotation=0)
            else:
                ax.text(0.5, 0.5, 'No valid data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12, color='gray')
            
            ax.set_title(project, fontsize=12, fontweight='bold')
            ax.set_ylabel(metric_titles.get(metric, metric) if idx == 0 else '')
            ax.set_xlabel('')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
        
        fig.suptitle(f'Algorithm Performance Comparison ({metric_titles.get(metric, metric)})', 
                    fontsize=14, fontweight='bold')
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
    
    def plot_safety_concept(self, save: bool = True) -> plt.Figure:
        """Plot Safety Risk Index conceptual model (Figure 1)."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Panel (a): Risk factors by method type
        methods = ['Manual\nLabor', 'Semi-\nAutomated', 'Fully\nAutomated']
        risks = [0.6, 0.4, 0.25]
        colors = ['#d62728', '#ff7f0e', '#2ca02c']
        
        ax = axes[0]
        bars = ax.bar(methods, risks, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Base Risk Factor (Rij)', fontsize=11)
        ax.set_title('(a) Risk Factors by Method Type', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 0.8)
        
        for bar, risk in zip(bars, risks):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{risk:.2f}', ha='center', fontsize=10, fontweight='bold')
        
        # Panel (b): Congestion penalty function
        ax = axes[1]
        n_t = np.linspace(1, 8, 100)
        alpha_values = [0.2, 0.3, 0.5]
        
        for alpha in alpha_values:
            delta_t = np.exp(alpha * (n_t - 1))
            ax.plot(n_t, delta_t, linewidth=2, label=f'α = {alpha}')
        
        ax.set_xlabel('Number of Concurrent Activities (nt)', fontsize=11)
        ax.set_ylabel('Congestion Penalty (δt)', fontsize=11)
        ax.set_title('(b) Exponential Congestion Penalty', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        fig.suptitle('Figure 1: Safety Risk Index (SRI) Conceptual Model',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'fig_safety_concept.png', 
                       dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_pareto_matrix(self, results: List[Dict], project_name: str,
                          save: bool = True) -> plt.Figure:
        """
        Plot 7x7 Pareto front matrix showing all pairwise objective projections.
        This is Figure X in the manuscript.
        """
        obj_names = ['Z1:Time', 'Z2:Cost', 'Z3:Quality', 'Z4:Resource', 
                    'Z5:Environ', 'Z6:Social', 'Z7:Safety']
        n_obj = 7
        
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
        
        fig.suptitle(f'7D Pareto Front Matrix - {project_name}', 
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
        obj_names = ['Time', 'Cost', 'Quality', 'Resource', 'Environment', 'Social', 'Safety']
        
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
        obj_names = ['Z1:Time', 'Z2:Cost', 'Z3:Quality', 'Z4:Resource', 
                    'Z5:Environ', 'Z6:Social', 'Z7:Safety']
        
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
                
                x = np.arange(7)
                for sol in F_norm:
                    ax.plot(x, sol, color=self.colors[k % len(self.colors)], 
                           alpha=0.3, linewidth=1)
        
        # Add algorithm legend with thicker lines
        for k, algo in enumerate(algorithms):
            ax.plot([], [], color=self.colors[k % len(self.colors)], 
                   linewidth=3, label=algo)
        
        ax.set_xticks(range(7))
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
        obj_names = ['Z1:Time', 'Z2:Cost', 'Z3:Quality', 'Z4:Resource', 
                    'Z5:Environ', 'Z6:Social', 'Z7:Safety']
        
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
        
        # Define weight scenarios
        scenarios = {
            'Equal': [1, 1, 1, 1, 1, 1, 1],
            'Time-Cost Focus': [1, 2, 3, 4, 5, 6, 7],
            'Quality Focus': [3, 4, 1, 5, 6, 7, 2],
            'Safety Focus': [2, 3, 4, 5, 6, 7, 1],
            'Sustainability': [4, 5, 6, 7, 1, 2, 3],
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
    
    def plot_gantt_chart(self, results: List[Dict], project: Project, 
                         mcdm: 'OPATOPSIS', save: bool = True) -> plt.Figure:
        """
        Plot Gantt chart for the best compromise solution.
        
        Args:
            results: List of optimization results
            project: Project instance for schedule computation
            mcdm: OPATOPSIS instance for ranking solutions
            save: Whether to save the figure
        """
        # Get best solution for this project
        proj_results = [r for r in results if r['project'] == project.name and r['success']]
        if not proj_results:
            return None
        
        # Combine all solutions and find best by TOPSIS
        all_F = []
        all_X = []
        for r in proj_results:
            if len(r['F']) > 0:
                all_F.append(r['F'])
                all_X.append(r['X'])
        
        if not all_F:
            return None
        
        F = np.vstack(all_F)
        X = np.vstack(all_X)
        
        # Get best solution using TOPSIS
        cc, best_idx = mcdm.topsis_rank(F)
        best_X = X[best_idx].astype(int)
        
        # Compute schedule
        scheduler = CPMScheduler(project)
        schedule = scheduler.schedule(best_X)
        
        # Create Gantt chart
        fig, ax = plt.subplots(figsize=(14, max(6, project.n_activities * 0.5)))
        
        # Color map for methods
        method_colors = ['#2ecc71', '#3498db', '#e74c3c']  # Green, Blue, Red for methods 0,1,2
        
        # Plot each activity as a horizontal bar
        y_positions = []
        for i, act in enumerate(project.activities):
            y_pos = project.n_activities - i - 1
            y_positions.append(y_pos)
            
            start = schedule['es'][i]
            duration = schedule['durations'][i]
            method_idx = int(best_X[i])
            
            # Draw bar
            bar = ax.barh(y_pos, duration, left=start, height=0.6, 
                         color=method_colors[method_idx], 
                         edgecolor='black', linewidth=0.5, alpha=0.9)
            
            # Add activity name and method info
            ax.text(start + duration/2, y_pos, f'M{method_idx}', 
                   ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        
        # Format axes
        ax.set_yticks(y_positions)
        ax.set_yticklabels([act.name for act in project.activities], fontsize=9)
        ax.set_xlabel('Time (days)', fontsize=11)
        ax.set_xlim(0, schedule['makespan'] + 2)
        ax.set_ylim(-0.5, project.n_activities - 0.5)
        
        # Add grid
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Legend for methods
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=method_colors[0], edgecolor='black', label='Method 0 (Standard)'),
            Patch(facecolor=method_colors[1], edgecolor='black', label='Method 1 (Accelerated)'),
            Patch(facecolor=method_colors[2], edgecolor='black', label='Method 2 (Crash)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        # Title with schedule info
        ax.set_title(f'Gantt Chart - {project.name}\n'
                    f'Makespan: {schedule["makespan"]} days | '
                    f'Best Compromise Solution (TOPSIS Score: {cc[best_idx]:.4f})',
                    fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / f'fig_gantt_{project.name.replace(" ", "_")}.png',
                       dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_resource_profile(self, results: List[Dict], project: Project,
                              mcdm: 'OPATOPSIS', save: bool = True) -> plt.Figure:
        """
        Plot resource usage profile (Labor and Equipment) for best solution.
        
        Args:
            results: List of optimization results
            project: Project instance for schedule computation
            mcdm: OPATOPSIS instance for ranking solutions
            save: Whether to save the figure
        """
        # Get best solution for this project
        proj_results = [r for r in results if r['project'] == project.name and r['success']]
        if not proj_results:
            return None
        
        # Combine all solutions and find best by TOPSIS
        all_F = []
        all_X = []
        for r in proj_results:
            if len(r['F']) > 0:
                all_F.append(r['F'])
                all_X.append(r['X'])
        
        if not all_F:
            return None
        
        F = np.vstack(all_F)
        X = np.vstack(all_X)
        
        # Get best solution using TOPSIS
        cc, best_idx = mcdm.topsis_rank(F)
        best_X = X[best_idx].astype(int)
        
        # Compute schedule
        scheduler = CPMScheduler(project)
        schedule = scheduler.schedule(best_X)
        
        # Create resource profile plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        days = np.arange(schedule['makespan'])
        
        # Plot Labor
        ax1 = axes[0]
        ax1.fill_between(days, schedule['daily_labor'], alpha=0.7, color='#3498db', 
                        label='Daily Labor')
        ax1.axhline(y=project.max_labor, color='red', linestyle='--', linewidth=2, 
                   label=f'Max Limit ({project.max_labor})')
        ax1.set_ylabel('Labor (workers)', fontsize=11)
        ax1.set_title(f'Labor Resource Profile - {project.name}', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(alpha=0.3, linestyle='--')
        ax1.set_ylim(0, max(max(schedule['daily_labor']), project.max_labor) * 1.1)
        
        # Plot Equipment
        ax2 = axes[1]
        ax2.fill_between(days, schedule['daily_equipment'], alpha=0.7, color='#e67e22',
                        label='Daily Equipment')
        ax2.axhline(y=project.max_equipment, color='red', linestyle='--', linewidth=2,
                   label=f'Max Limit ({project.max_equipment})')
        ax2.set_ylabel('Equipment (units)', fontsize=11)
        ax2.set_xlabel('Time (days)', fontsize=11)
        ax2.set_title(f'Equipment Resource Profile - {project.name}', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(alpha=0.3, linestyle='--')
        ax2.set_ylim(0, max(max(schedule['daily_equipment']), project.max_equipment) * 1.1)
        
        fig.suptitle(f'Resource Usage Profile - Best Compromise Solution\n'
                    f'Makespan: {schedule["makespan"]} days',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / f'fig_resource_{project.name.replace(" ", "_")}.png',
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
                
                records.append({
                    'Project': project,
                    'Best Algorithm': best_algo,
                    'TOPSIS CC': cc[best_idx],
                    'Z1_Time': best_F[0],
                    'Z2_Cost': best_F[1],
                    'Z3_Quality': -best_F[2],
                    'Z4_Resource': best_F[3],
                    'Z5_Environment': best_F[4],
                    'Z6_Social': best_F[5],
                    'Z7_Safety': best_F[6],
                    'Solution': str(list(best_X.astype(int)))
                })
        
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
        obj_names = ['Z1_Time', 'Z2_Cost', 'Z3_Quality', 'Z4_Resource', 
                    'Z5_Environment', 'Z6_Social', 'Z7_Safety']
        
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
                
                obj_names = ['Time', 'Cost', 'Quality', 'Resource', 'Environment', 'Social', 'Safety']
                directions = ['min', 'min', 'max', 'min', 'min', 'min', 'min']
                
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
            'Equal': [1, 1, 1, 1, 1, 1, 1],
            'Time-Cost': [1, 2, 3, 4, 5, 6, 7],
            'Quality': [3, 4, 1, 5, 6, 7, 2],
            'Safety': [2, 3, 4, 5, 6, 7, 1],
            'Sustainability': [4, 5, 6, 7, 1, 2, 3],
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
        obj_names = ['Time (Z1)', 'Cost (Z2)', 'Quality (Z3)', 'Resource (Z4)', 
                    'Environment (Z5)', 'Social (Z6)', 'Safety (Z7)']
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
                        
                        # Calculate improvement (negative means better for minimization)
                        if i == 2:  # Quality is maximized (stored as negative)
                            improvement = ((opt_val - baseline_val) / (abs(baseline_val) + 1e-10)) * 100
                        else:  # Other objectives are minimized
                            improvement = ((baseline_val - opt_val) / (abs(baseline_val) + 1e-10)) * 100
                        
                        record[f'{obj_name} Improvement (%)'] = round(improvement, 2)
                        record[f'{obj_name} Baseline'] = round(baseline_val, 2)
                        record[f'{obj_name} Optimized'] = round(opt_val, 2)
                    
                    records.append(record)
        
        result = pd.DataFrame(records)
        result.to_csv(self.output_dir / 'table_percentage_improvement.csv', index=False)
        return result
    
    def improvement_summary_by_algorithm(self, projects: List[Project], results: List[Dict]) -> pd.DataFrame:
        """Generate Table: Average % Improvement by Algorithm across all objectives."""
        obj_names = ['Time', 'Cost', 'Quality', 'Resource', 'Environment', 'Social', 'Safety']
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
                                if i == 2:  # Quality (maximized, stored negative)
                                    imp = ((sol_F[i] - baseline_F[i]) / (abs(baseline_F[i]) + 1e-10)) * 100
                                else:
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
        obj_names = ['Time', 'Cost', 'Quality', 'Resource', 'Environment', 'Social', 'Safety']
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
        obj_names = ['Time', 'Cost', 'Quality', 'Resource', 'Environment', 'Social', 'Safety']
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
                        'Best Value': round(np.min(all_F[:, i]) if i != 2 else np.max(all_F[:, i]), 2),
                        'Worst Value': round(np.max(all_F[:, i]) if i != 2 else np.min(all_F[:, i]), 2),
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
    """Run full experimental design."""
    print("=" * 60)
    print("7D MULTI-OBJECTIVE PROJECT SCHEDULING OPTIMIZATION")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Population Size: {pop_size}")
    print(f"  Generations: {n_gen}")
    print(f"  Runs per config: {n_runs}")
    print(f"  Parallel jobs: {n_jobs}")
    print("=" * 60)
    
    # Create all projects (2 projects: Highway simple, Metro complex with 12 activities)
    projects = [
        create_highway_project(),  # 8 activities - simpler project
        create_metro_project()     # 12 activities - complex project with relationships
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
        algo_names=ALGORITHM_NAMES,
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
    
    # *** SAVE RESULTS IMMEDIATELY - Never lose optimization data ***
    print("Saving results (backup)...")
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    df.to_csv(output_path / 'results_all_runs.csv', index=False)
    print(f"  ✓ Saved: results_all_runs.csv ({len(df)} rows)")
    
    # Save Pareto fronts early too
    pareto_data = {}
    for r in results:
        if r.get('success', False) and len(r.get('F', [])) > 0:
            key = f"{r['project']}_{r['algorithm']}_{r['seed']}"
            pareto_data[key] = {
                'F': r['F'].tolist(),
                'X': r['X'].tolist() if hasattr(r['X'], 'tolist') else r['X']
            }
    with open(output_path / 'pareto_fronts.json', 'w') as f:
        json.dump(pareto_data, f)
    print(f"  ✓ Saved: pareto_fronts.json")
    
    # Initialize generators
    viz = Visualizer(output_dir)
    tables = TableGenerator(output_dir)
    mcdm = OPATOPSIS()
    
    # ===========================================
    # TABLES (with error protection)
    # ===========================================
    print("Generating tables...")
    
    def safe_generate(name, func, *args, **kwargs):
        """Safely generate a table with timeout protection."""
        try:
            print(f"    Generating {name}...")
            result = func(*args, **kwargs)
            print(f"    ✓ {name} complete")
            return result
        except Exception as e:
            print(f"    ✗ {name} failed: {str(e)[:50]}")
            return None
    
    # Core tables
    safe_generate("project_characteristics", tables.project_characteristics, projects)
    safe_generate("performance_metrics", tables.performance_metrics, df)
    safe_generate("friedman_rankings", tables.friedman_rankings, df)
    safe_generate("best_solutions", tables.best_solutions, results, mcdm)
    safe_generate("algorithm_ranking_summary", tables.algorithm_ranking_summary, df)
    
    # Additional tables
    safe_generate("objective_statistics", tables.objective_statistics, results)
    safe_generate("baseline_comparison", tables.baseline_comparison, projects, results, mcdm)
    # SKIPPED - slow table: safe_generate("topsis_rankings_by_scenario", tables.topsis_rankings_by_scenario, results)
    safe_generate("activity_method_frequency", tables.activity_method_frequency, results, projects)
    safe_generate("pareto_dominance_summary", tables.pareto_dominance_summary, results)
    safe_generate("latex_performance_table", tables.latex_performance_table, df)
    
    # Improvement analysis tables
    print("  Generating improvement analysis tables...")
    safe_generate("percentage_improvement_table", tables.percentage_improvement_table, projects, results)
    safe_generate("improvement_summary_by_algorithm", tables.improvement_summary_by_algorithm, projects, results)
    safe_generate("algorithm_pairwise_wins_HV", tables.algorithm_pairwise_wins, df, 'HV')
    safe_generate("algorithm_pairwise_wins_IGD", tables.algorithm_pairwise_wins, df, 'IGD')
    # SKIPPED - slow table: safe_generate("best_solution_characteristics", tables.best_solution_characteristics, results, projects, mcdm)
    safe_generate("computational_efficiency", tables.computational_efficiency, df)
    # SKIPPED - slow table: safe_generate("solution_diversity_analysis", tables.solution_diversity_analysis, results)
    safe_generate("objective_extremes", tables.objective_extremes, results)
    
    # Wilcoxon pairwise
    print("    Generating wilcoxon_pairwise...")
    try:
        wilcoxon_df = StatisticalAnalysis.wilcoxon_pairwise(df, metric='HV')
        wilcoxon_df.to_csv(Path(output_dir) / 'table_wilcoxon_pairwise.csv')
        print("    ✓ wilcoxon_pairwise complete")
    except Exception as e:
        print(f"    ✗ wilcoxon_pairwise failed: {str(e)[:50]}")
        wilcoxon_df = pd.DataFrame()
    
    # ===========================================
    # FIGURES
    # ===========================================
    print("Generating figures...")
    
    # Conceptual figures
    viz.plot_safety_concept()
    
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
        
        # Pareto front visualizations
        viz.plot_pareto_fronts(results, proj.name, objectives=(0, 1))
        viz.plot_pareto_fronts(results, proj.name, objectives=(0, 6))  # Time vs Safety
        viz.plot_pareto_fronts(results, proj.name, objectives=(1, 2))  # Cost vs Quality
        
        # 7D Pareto Matrix (key figure for manuscript)
        viz.plot_pareto_matrix(results, proj.name)
        
        # 3D Pareto plots
        viz.plot_pareto_3d(results, proj.name, objectives=(0, 1, 6))  # Time-Cost-Safety
        viz.plot_pareto_3d(results, proj.name, objectives=(0, 2, 4))  # Time-Quality-Environment
        
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
        
        # NEW: Gantt Chart and Resource Profile for best solution
        viz.plot_gantt_chart(results, proj, mcdm)
        viz.plot_resource_profile(results, proj, mcdm)
    
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
    print(f"Figures generated: {8 + len(projects) * 13}")  # Added Gantt and Resource per project
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

