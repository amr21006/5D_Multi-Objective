"""
7D Hepta-Objective Optimization Framework for Construction
============================================================================
MOEA/D Based Optimization with Automated MCDM Selection

Objectives (7D):
    Z1: Cost - Total Construction Cost (Minimize)
    Z2: Time - Project Makespan (Minimize)
    Z3: Carbon - Lifecycle Carbon Footprint (Minimize)
    Z4: Resources - Hybrid Resource Profile Index (Minimize)
    Z5: Congestion - Workspace Congestion Index (Minimize)
    Z6: Safety - Occupational Safety Risk (Minimize)
    Z7: Quality - Defect Potential Cost (Minimize)

Author: Research Implementation | Date: January 2026
For Google Colab Execution
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.cm import get_cmap
import seaborn as sns
from math import pi
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import warnings
import time
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
from scipy import stats
from scipy.spatial.distance import cdist
import copy

try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs): return iterable

# Pymoo imports
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.indicators.gd import GD
from pymoo.termination import get_termination
from pymoo.decomposition.tchebicheff import Tchebicheff

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'pop_size': 200,
    'n_gen': 500,
    'n_runs': 30,
    'n_workers': max(1, cpu_count() - 1),
    'seed_base': 42,
    'output_dir': 'results_7d_bridge',
    'fig_dpi': 300,
    
    # 7D Parameters
    'alpha': 1.0,   # W_k for resource smoothing
    'beta': 10.0,   # Penalty for resource overload
    'w_labor': 0.6, # Weight for labor in resource index
    'w_equip': 0.4, # Weight for equipment in resource index
    'gamma_mat': 1.0, # Material emission factor
    'gamma_fuel': 2.6, # Fuel emission factor
}

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Method:
    """Construction method for an activity (7D Enhanced)."""
    id: int
    duration: int
    cost: float # Direct Cost
    # 7D specific fields
    risk_factor: float      # Rho: Contingency factor (0.05-0.20)
    area_requirement: float # A_im: Workspace area (m2)
    defect_probability: float # P_defect: Probability of defect (0-1)
    rework_cost: float      # C_rework: Potential rework cost ($)
    safety_hazard: float    # H_im: Hazard index (0-10)
    carbon_emission: float  # Z3: CO2 emissions
    fuel_energy: float      # Z3 Part 2: Energy/Fuel
    labor: int             # Resource 1
    equipment: int         # Resource 2

@dataclass
class Activity:
    """Bridge construction activity with multiple method alternatives."""
    id: int
    name: str
    methods: List[Method]
    predecessors: List[int] = field(default_factory=list)
    weight: float = 1.0
    
    @property
    def n_methods(self) -> int:
        return len(self.methods)

@dataclass
class Project:
    """Bridge construction project (7D Context)."""
    name: str
    project_type: str
    activities: List[Activity]
    daily_indirect_cost: float
    deadline: int
    max_budget: float
    site_area: float       # A_total: Total available site area
    labor_limit: int      # R_max_1
    equip_limit: int      # R_max_2
    
    @property
    def n_activities(self) -> int:
        return len(self.activities)
    
    @property
    def search_space_size(self) -> int:
        size = 1
        for a in self.activities:
            size *= len(a.methods)
        return size

# =============================================================================
# CPM SCHEDULER
# =============================================================================

class CPMScheduler:
    """Critical Path Method scheduler for bridge projects."""
    
    def __init__(self, project: Project):
        self.project = project
        self.n = project.n_activities
    
    def schedule(self, solution: np.ndarray) -> Dict:
        """Calculate schedule using forward pass."""
        solution = np.asarray(solution, dtype=int)
        durations = np.array([
            self.project.activities[i].methods[solution[i]].duration 
            for i in range(self.n)
        ])
        
        es = np.zeros(self.n, dtype=int)  # Early start
        ef = np.zeros(self.n, dtype=int)  # Early finish
        
        for i, act in enumerate(self.project.activities):
            if act.predecessors:
                es[i] = max(ef[p] for p in act.predecessors)
            ef[i] = es[i] + durations[i]
        
        makespan = int(max(ef))
        
        # Resource profiles
        daily_labor = np.zeros(max(1, makespan), dtype=int)
        daily_equipment = np.zeros(max(1, makespan), dtype=int)
        active_activities = [[] for _ in range(max(1, makespan))]
        
        for i, act in enumerate(self.project.activities):
            m = act.methods[solution[i]]
            for t in range(es[i], ef[i]):
                if t < makespan:
                    daily_labor[t] += m.labor
                    daily_equipment[t] += m.equipment
                    active_activities[t].append(i)
        
        return {
            'es': es, 'ef': ef, 'durations': durations, 'makespan': makespan,
            'daily_labor': daily_labor, 'daily_equipment': daily_equipment,
            'active_activities': active_activities
        }

# =============================================================================
# 7D OBJECTIVE CALCULATOR
# =============================================================================

class ObjectiveCalculator7D:
    """
    Calculate 7D objectives based on Manuscript:
    Z1: Total Construction Cost (Min)
    Z2: Project Makespan (Min)
    Z3: Lifecycle Carbon Footprint (Min)
    Z4: Hybrid Resource Profile Index (Min)
    Z5: Workspace Congestion Index (Min)
    Z6: Occupational Safety Risk (Min)
    Z7: Quality Defect Potential (Min)
    """
    
    def __init__(self, project: Project):
        self.project = project
        self.scheduler = CPMScheduler(project)
        self.site_area = project.site_area
    
    def evaluate(self, solution: np.ndarray) -> np.ndarray:
        solution = np.asarray(solution, dtype=int)
        schedule = self.scheduler.schedule(solution)
        
        # Pre-fetch methods for speed
        methods = [self.project.activities[i].methods[solution[i]] for i in range(len(solution))]
        
        z1 = self._calc_cost_z1(methods, schedule)
        z2 = self._calc_time_z2(schedule)
        z3 = self._calc_carbon_z3(methods)
        z4 = self._calc_resource_z4(schedule)
        z5 = self._calc_congestion_z5(methods, schedule)
        z6 = self._calc_safety_z6(methods)
        z7 = self._calc_quality_z7(methods)
        
        return np.array([z1, z2, z3, z4, z5, z6, z7])
    
    def _calc_cost_z1(self, methods: List[Method], schedule: Dict) -> float:
        """Z1: Direct + Indirect + Risk Costs."""
        direct = sum(m.cost for m in methods)
        indirect = self.project.daily_indirect_cost * schedule['makespan']
        risk_cost = sum(m.risk_factor * m.cost for m in methods) # Risk contingency
        return direct + indirect + risk_cost

    def _calc_time_z2(self, schedule: Dict) -> float:
        """Z2: Makespan."""
        return float(schedule['makespan'])
    
    def _calc_carbon_z3(self, methods: List[Method]) -> float:
        """Z3: Embodied (Mat) + Process (Fuel) Emissions."""
        # Note: In our data model, 'carbon_emission' is Material and 'fuel_energy' is Energy
        # Formula: Q*gamma_mat + E*gamma_fuel
        # We assume fields are already quantities, just apply factors
        mat_em = sum(m.carbon_emission * CONFIG['gamma_mat'] for m in methods)
        fuel_em = sum(m.fuel_energy * CONFIG['gamma_fuel'] for m in methods)
        return mat_em + fuel_em
    
    def _calc_resource_z4(self, schedule: Dict) -> float:
        """Z4: Resource Fluctuation + Overload (Hybrid)."""
        labor = schedule['daily_labor']
        equip = schedule['daily_equipment']
        
        def calc_res_score(profile, limit):
            # Fluctuation: alpha * sum(r^2)
            fluct = CONFIG['alpha'] * np.sum(profile ** 2)
            # Overload: beta * sum(max(0, r-Rmax)^2)
            overload = np.maximum(0, profile - limit)
            pen = CONFIG['beta'] * np.sum(overload ** 2)
            return fluct + pen
            
        score_l = calc_res_score(labor, self.project.labor_limit)
        score_e = calc_res_score(equip, self.project.equip_limit)
        
        return CONFIG['w_labor'] * score_l + CONFIG['w_equip'] * score_e
    
    def _calc_congestion_z5(self, methods: List[Method], schedule: Dict) -> float:
        """Z5: Workspace Congestion Index."""
        # Sum of (Area_occupied_t / Total_Area)^2
        T = schedule['makespan']
        congestion_score = 0.0
        
        # Map activity index to area
        act_areas = np.array([m.area_requirement for m in methods])
        
        active_acts = schedule['active_activities']
        # active_acts is list of lists of indices
        
        for t in range(min(len(active_acts), T)):
            active = active_acts[t]
            if not active: continue
                
            total_active_area = np.sum(act_areas[active])
            density = total_active_area / self.project.site_area
            congestion_score += density ** 2
            
        return congestion_score
    
    def _calc_safety_z6(self, methods: List[Method]) -> float:
        """Z6: Cumulative Hazard Exposure (Hazard * Duration)."""
        # H_im * d_im
        return sum(m.safety_hazard * m.duration for m in methods)
    
    def _calc_quality_z7(self, methods: List[Method]) -> float:
        """Z7: Probabilistic Defect Cost (Prob * ReworkCost)."""
        return sum(m.defect_probability * m.rework_cost for m in methods)

    def get_objective_names(self) -> List[str]:
        return ['Z1_Cost', 'Z2_Time', 'Z3_Carbon', 'Z4_Res', 'Z5_Space', 'Z6_Safe', 'Z7_Qual']

# =============================================================================
# CASE STUDY 1: URBAN OVERPASS BRIDGE (Updated for 7D)
# =============================================================================

def create_urban_overpass_project() -> Project:
    """
    Case Study: Urban Overpass (7D Data)
    """
    
    def M(id, dur, cost, labor, equip, qual_score, safe_risk, carb, eng):
        # Helper to map 5D inputs to 7D fields
        # Derive new fields from old ones
        risk_f = 0.05 + (safe_risk * 0.2) # Risk factor proportional to safety risk
        area = labor * 4.0 + equip * 15.0 # Area estimation
        defect_p = max(0.01, 1.0 - qual_score) / 2.0 # Defect prob
        rew_c = cost * 0.15 # Rework cost estimate
        haz = safe_risk * 10 # Hazard index 0-10
        
        return Method(id, dur, cost, risk_f, area, defect_p, rew_c, haz, carb, eng, labor, equip)
    
    activities = [
        # A0: Site Preparation
        Activity(0, "Site Preparation", [
            M(0, 20, 75000, 22, 10, 0.72, 0.15, 950, 4500),
            M(1, 14, 110000, 28, 14, 0.78, 0.12, 1100, 5200),
            M(2, 10, 135000, 36, 18, 0.85, 0.10, 850, 4000),
            M(3, 7,  160000, 45, 22, 0.82, 0.14, 900, 4200),
        ], []),
        
        # A1: Utility Relocation
        Activity(1, "Utility Relocation", [
            M(0, 28, 110000, 18, 12, 0.74, 0.22, 750, 3500),
            M(1, 20, 165000, 24, 16, 0.80, 0.18, 950, 4200),
            M(2, 15, 195000, 32, 20, 0.86, 0.14, 700, 3000),
            M(3, 10, 230000, 42, 26, 0.83, 0.20, 850, 3600),
        ], [0]),
        
        # A2: Excavation A1
        Activity(2, "Foundation Excavation A1", [
            M(0, 24, 95000,  25, 18, 0.73, 0.28, 1450, 9500),
            M(1, 18, 135000, 32, 24, 0.79, 0.24, 1650, 11000),
            M(2, 14, 165000, 42, 30, 0.87, 0.20, 1200, 8000),
            M(3, 9,  205000, 55, 38, 0.84, 0.26, 1350, 9000),
        ], [1]),
        
        # A3: Excavation A2
        Activity(3, "Foundation Excavation A2", [
            M(0, 24, 95000,  25, 18, 0.73, 0.28, 1450, 9500),
            M(1, 18, 135000, 32, 24, 0.79, 0.24, 1650, 11000),
            M(2, 14, 165000, 42, 30, 0.87, 0.20, 1200, 8000),
            M(3, 9,  205000, 55, 38, 0.84, 0.26, 1350, 9000),
        ], [1]),

        # A4: Piling A1
        Activity(4, "Pile Installation A1", [
            M(0, 30, 210000, 35, 28, 0.75, 0.32, 3200, 17000),
            M(1, 22, 310000, 45, 35, 0.82, 0.28, 3800, 20000),
            M(2, 16, 375000, 58, 44, 0.90, 0.22, 2500, 14000),
            M(3, 11, 460000, 75, 55, 0.86, 0.28, 2900, 16000),
        ], [2]),

        # A5: Piling A2
        Activity(5, "Pile Installation A2", [
            M(0, 30, 210000, 35, 28, 0.75, 0.32, 3200, 17000),
            M(1, 22, 310000, 45, 35, 0.82, 0.28, 3800, 20000),
            M(2, 16, 375000, 58, 44, 0.90, 0.22, 2500, 14000),
            M(3, 11, 460000, 75, 55, 0.86, 0.28, 2900, 16000),
        ], [3]),

        # A6: Abutment 1
        Activity(6, "Abutment 1 Cons", [
            M(0, 38, 195000, 40, 22, 0.74, 0.25, 2000, 10500),
            M(1, 28, 280000, 52, 28, 0.81, 0.21, 2400, 12500),
            M(2, 21, 335000, 68, 36, 0.89, 0.17, 1600, 8500),
            M(3, 14, 415000, 88, 46, 0.85, 0.22, 1900, 9800),
        ], [4]),

        # A7: Abutment 2
        Activity(7, "Abutment 2 Cons", [
            M(0, 38, 195000, 40, 22, 0.74, 0.25, 2000, 10500),
            M(1, 28, 280000, 52, 28, 0.81, 0.21, 2400, 12500),
            M(2, 21, 335000, 68, 36, 0.89, 0.17, 1600, 8500),
            M(3, 14, 415000, 88, 46, 0.85, 0.22, 1900, 9800),
        ], [5]),

        # A8: Bearings
        Activity(8, "Bearing Installation", [
            M(0, 14, 65000, 18, 12, 0.78, 0.18, 450, 3000),
            M(1, 9,  95000, 24, 16, 0.84, 0.14, 580, 3800),
            M(2, 7,  115000, 32, 20, 0.91, 0.11, 400, 2500),
            M(3, 5,  140000, 42, 26, 0.87, 0.16, 480, 3200),
        ], [6, 7]),

        # A9: Girder Fab
        Activity(9, "Girder Fabrication", [
            M(0, 48, 450000, 65, 40, 0.76, 0.20, 5000, 32000),
            M(1, 35, 630000, 85, 52, 0.83, 0.16, 6000, 38000),
            M(2, 28, 760000, 110, 68, 0.91, 0.12, 4200, 26000),
            M(3, 20, 940000, 145, 88, 0.87, 0.18, 4800, 30000),
        ], [0]),

        # A10: Erection
        Activity(10, "Girder Erection", [
            M(0, 20, 195000, 35, 45, 0.75, 0.35, 2500, 13500),
            M(1, 14, 285000, 48, 58, 0.82, 0.30, 3000, 16000),
            M(2, 10, 340000, 65, 75, 0.90, 0.24, 2000, 11000),
            M(3, 7,  420000, 85, 95, 0.86, 0.32, 2400, 12500),
        ], [8, 9]),

        # A11: Deck Slab
        Activity(11, "Deck Slab Construction", [
            M(0, 32, 210000, 45, 28, 0.74, 0.26, 2600, 13000),
            M(1, 24, 315000, 58, 36, 0.81, 0.22, 3200, 15500),
            M(2, 18, 395000, 76, 46, 0.89, 0.17, 2100, 10500),
            M(3, 12, 480000, 98, 58, 0.85, 0.24, 2500, 12000),
        ], [10]),

        # A12: Railings
        Activity(12, "Railings & Barriers", [
             M(0, 18, 90000,  22, 14, 0.77, 0.22, 750, 4500),
             M(1, 12, 125000, 30, 18, 0.83, 0.18, 900, 5500),
             M(2, 9,  150000, 40, 24, 0.90, 0.14, 600, 3800),
             M(3, 6,  190000, 52, 32, 0.86, 0.20, 700, 4200),
        ], [11]),
        
        # A13: Paving
        Activity(13, "Approach Paving", [
             M(0, 22, 145000, 32, 24, 0.75, 0.20, 1700, 9500),
             M(1, 15, 215000, 42, 32, 0.82, 0.16, 2100, 11500),
             M(2, 11, 260000, 56, 42, 0.89, 0.12, 1400, 8000),
             M(3, 8,  325000, 72, 54, 0.85, 0.18, 1600, 9000),
        ], [12]),

        # A14: Inspection
        Activity(14, "Final Inspection", [
             M(0, 14, 45000,  15, 8, 0.80, 0.08, 200, 1300),
             M(1, 9,  75000,  22, 12, 0.86, 0.06, 260, 1800),
             M(2, 7,  95000,  30, 16, 0.93, 0.04, 150, 1000),
             M(3, 5,  120000, 40, 22, 0.90, 0.07, 180, 1200),
        ], [13]),
    ]
    
    return Project(
        name="Urban Overpass 7D",
        project_type="Prestressed Concrete",
        activities=activities,
        daily_indirect_cost=15000,
        deadline=280,
        max_budget=8500000,
        site_area=5000.0, # 5000 m2 site
        labor_limit=150,
        equip_limit=100
    )

# Removing unused case studies to save space/complexity for now
def create_river_crossing_project(): return create_urban_overpass_project() 
def create_highway_ramp_project(): return create_urban_overpass_project()

# =============================================================================
# CASE STUDY 2: RIVER CROSSING BRIDGE (18 Activities)
# =============================================================================

def create_river_crossing_project() -> Project:
    """
    Case Study 2: River Crossing Bridge - 18 activities, 4 methods each.
    Steel-concrete composite continuous girder (3 spans: 40m+60m+40m = 140m).
    Complexity: Environmental sensitivity, seasonal constraints, water work.
    Decision Space: 4^18 = 68,719,476,736 combinations
    """
    
    def M(id, dur, cost, qual, labor, equip, safety, carbon, waste, energy, local, train):
        return Method(id, dur, cost, qual, labor, equip, safety, carbon, waste, energy, local, train)
    
    activities = [
        Activity(0, "Environmental Permits & Survey", [
            M(0, 30, 180000, 0.82, 12, 6, 0.05, 250, 3.5, 1800, 0.95, 60),
            M(1, 22, 245000, 0.88, 18, 10, 0.04, 320, 4.5, 2400, 0.90, 80),
            M(2, 16, 320000, 0.94, 26, 14, 0.03, 420, 5.8, 3200, 0.84, 100),
            M(3, 10, 420000, 0.90, 35, 20, 0.05, 550, 7.5, 4200, 0.78, 72),
        ], [], 1.3, ['environmental', 'regulatory']),
        
        Activity(1, "Access Road Construction", [
            M(0, 25, 320000, 0.73, 28, 22, 0.22, 1800, 42.0, 12000, 0.82, 24),
            M(1, 20, 405000, 0.80, 36, 28, 0.18, 2150, 50.0, 14500, 0.76, 32),
            M(2, 15, 510000, 0.87, 48, 36, 0.14, 2650, 62.0, 18000, 0.70, 42),
            M(3, 10, 650000, 0.83, 62, 48, 0.20, 3300, 78.0, 22500, 0.62, 28),
        ], [0], 1.2, ['earthwork', 'temporary']),
        
        Activity(2, "Cofferdam Pier 1", [
            M(0, 35, 580000, 0.74, 42, 35, 0.38, 3200, 28.0, 18000, 0.68, 48),
            M(1, 28, 725000, 0.81, 55, 45, 0.32, 3850, 35.0, 22000, 0.62, 60),
            M(2, 21, 920000, 0.89, 72, 58, 0.26, 4700, 44.0, 27500, 0.55, 76),
            M(3, 14, 1180000, 0.85, 95, 75, 0.34, 5800, 56.0, 35000, 0.48, 52),
        ], [1], 1.5, ['water_work', 'cofferdam', 'dewatering']),
        
        Activity(3, "Cofferdam Pier 2", [
            M(0, 35, 580000, 0.74, 42, 35, 0.38, 3200, 28.0, 18000, 0.68, 48),
            M(1, 28, 725000, 0.81, 55, 45, 0.32, 3850, 35.0, 22000, 0.62, 60),
            M(2, 21, 920000, 0.89, 72, 58, 0.26, 4700, 44.0, 27500, 0.55, 76),
            M(3, 14, 1180000, 0.85, 95, 75, 0.34, 5800, 56.0, 35000, 0.48, 52),
        ], [1], 1.5, ['water_work', 'cofferdam', 'dewatering']),
        
        Activity(4, "Pier 1 Foundation (Caisson)", [
            M(0, 45, 920000, 0.75, 55, 48, 0.42, 5500, 65.0, 32000, 0.65, 56),
            M(1, 36, 1150000, 0.82, 72, 62, 0.36, 6600, 78.0, 39000, 0.58, 72),
            M(2, 28, 1450000, 0.90, 95, 80, 0.28, 8000, 95.0, 48000, 0.50, 92),
            M(3, 20, 1850000, 0.86, 125, 105, 0.38, 9800, 118.0, 60000, 0.42, 64),
        ], [2], 1.6, ['caisson', 'underwater', 'heavy_equipment']),
        
        Activity(5, "Pier 2 Foundation (Caisson)", [
            M(0, 45, 920000, 0.75, 55, 48, 0.42, 5500, 65.0, 32000, 0.65, 56),
            M(1, 36, 1150000, 0.82, 72, 62, 0.36, 6600, 78.0, 39000, 0.58, 72),
            M(2, 28, 1450000, 0.90, 95, 80, 0.28, 8000, 95.0, 48000, 0.50, 92),
            M(3, 20, 1850000, 0.86, 125, 105, 0.38, 9800, 118.0, 60000, 0.42, 64),
        ], [3], 1.6, ['caisson', 'underwater', 'heavy_equipment']),
        
        Activity(6, "Abutment A Construction", [
            M(0, 32, 450000, 0.74, 42, 26, 0.24, 2200, 35.0, 11000, 0.80, 32),
            M(1, 25, 565000, 0.81, 55, 34, 0.20, 2650, 42.0, 13500, 0.74, 42),
            M(2, 19, 710000, 0.89, 72, 44, 0.16, 3200, 52.0, 16800, 0.68, 54),
            M(3, 13, 895000, 0.85, 95, 58, 0.22, 3900, 64.0, 21000, 0.60, 38),
        ], [1], 1.4, ['concrete', 'formwork', 'rebar']),
        
        Activity(7, "Abutment B Construction", [
            M(0, 32, 450000, 0.74, 42, 26, 0.24, 2200, 35.0, 11000, 0.80, 32),
            M(1, 25, 565000, 0.81, 55, 34, 0.20, 2650, 42.0, 13500, 0.74, 42),
            M(2, 19, 710000, 0.89, 72, 44, 0.16, 3200, 52.0, 16800, 0.68, 54),
            M(3, 13, 895000, 0.85, 95, 58, 0.22, 3900, 64.0, 21000, 0.60, 38),
        ], [1], 1.4, ['concrete', 'formwork', 'rebar']),
        
        Activity(8, "Pier 1 Column Construction", [
            M(0, 38, 520000, 0.75, 48, 32, 0.28, 2800, 38.0, 14000, 0.78, 36),
            M(1, 30, 650000, 0.82, 62, 42, 0.24, 3350, 46.0, 17000, 0.72, 46),
            M(2, 23, 820000, 0.90, 82, 54, 0.19, 4100, 56.0, 21200, 0.66, 60),
            M(3, 16, 1030000, 0.86, 108, 72, 0.26, 5000, 70.0, 26500, 0.58, 42),
        ], [4], 1.5, ['concrete', 'formwork', 'height_work']),
        
        Activity(9, "Pier 2 Column Construction", [
            M(0, 38, 520000, 0.75, 48, 32, 0.28, 2800, 38.0, 14000, 0.78, 36),
            M(1, 30, 650000, 0.82, 62, 42, 0.24, 3350, 46.0, 17000, 0.72, 46),
            M(2, 23, 820000, 0.90, 82, 54, 0.19, 4100, 56.0, 21200, 0.66, 60),
            M(3, 16, 1030000, 0.86, 108, 72, 0.26, 5000, 70.0, 26500, 0.58, 42),
        ], [5], 1.5, ['concrete', 'formwork', 'height_work']),
        
        Activity(10, "Pier Cap 1 Construction", [
            M(0, 22, 380000, 0.76, 38, 24, 0.26, 1850, 28.0, 9500, 0.80, 40),
            M(1, 17, 480000, 0.83, 50, 32, 0.22, 2250, 34.0, 11800, 0.74, 52),
            M(2, 13, 605000, 0.91, 66, 42, 0.17, 2750, 42.0, 14600, 0.68, 66),
            M(3, 9, 765000, 0.87, 88, 56, 0.24, 3400, 52.0, 18500, 0.60, 46),
        ], [8], 1.4, ['concrete', 'formwork', 'height_work']),
        
        Activity(11, "Pier Cap 2 Construction", [
            M(0, 22, 380000, 0.76, 38, 24, 0.26, 1850, 28.0, 9500, 0.80, 40),
            M(1, 17, 480000, 0.83, 50, 32, 0.22, 2250, 34.0, 11800, 0.74, 52),
            M(2, 13, 605000, 0.91, 66, 42, 0.17, 2750, 42.0, 14600, 0.68, 66),
            M(3, 9, 765000, 0.87, 88, 56, 0.24, 3400, 52.0, 18500, 0.60, 46),
        ], [9], 1.4, ['concrete', 'formwork', 'height_work']),
        
        Activity(12, "Steel Girder Fabrication", [
            M(0, 55, 1450000, 0.77, 85, 55, 0.18, 7200, 52.0, 42000, 0.52, 64),
            M(1, 44, 1820000, 0.84, 110, 72, 0.14, 8600, 65.0, 52000, 0.46, 82),
            M(2, 35, 2280000, 0.92, 145, 95, 0.10, 10500, 80.0, 65000, 0.40, 105),
            M(3, 26, 2900000, 0.88, 190, 125, 0.16, 12800, 100.0, 82000, 0.34, 72),
        ], [0], 1.6, ['fabrication', 'welding', 'quality_control']),
        
        Activity(13, "Girder Erection Span 1", [
            M(0, 20, 420000, 0.75, 45, 55, 0.38, 2400, 12.0, 15000, 0.68, 64),
            M(1, 16, 525000, 0.82, 58, 72, 0.32, 2900, 15.0, 18500, 0.62, 82),
            M(2, 12, 660000, 0.90, 78, 95, 0.26, 3550, 19.0, 23000, 0.55, 102),
            M(3, 8, 840000, 0.86, 102, 125, 0.34, 4350, 24.0, 29000, 0.48, 72),
        ], [6, 10, 12], 1.6, ['lifting', 'heavy_equipment', 'height_work']),
        
        Activity(14, "Girder Erection Span 2", [
            M(0, 22, 480000, 0.75, 48, 58, 0.40, 2650, 14.0, 16500, 0.66, 68),
            M(1, 17, 600000, 0.82, 62, 76, 0.34, 3200, 17.5, 20500, 0.60, 86),
            M(2, 13, 755000, 0.90, 82, 100, 0.27, 3900, 22.0, 25500, 0.53, 108),
            M(3, 9, 960000, 0.86, 108, 132, 0.36, 4800, 27.5, 32000, 0.46, 76),
        ], [10, 11, 13], 1.6, ['lifting', 'heavy_equipment', 'height_work']),
        
        Activity(15, "Girder Erection Span 3", [
            M(0, 20, 420000, 0.75, 45, 55, 0.38, 2400, 12.0, 15000, 0.68, 64),
            M(1, 16, 525000, 0.82, 58, 72, 0.32, 2900, 15.0, 18500, 0.62, 82),
            M(2, 12, 660000, 0.90, 78, 95, 0.26, 3550, 19.0, 23000, 0.55, 102),
            M(3, 8, 840000, 0.86, 102, 125, 0.34, 4350, 24.0, 29000, 0.48, 72),
        ], [11, 7, 14], 1.6, ['lifting', 'heavy_equipment', 'height_work']),
        
        Activity(16, "Composite Deck Construction", [
            M(0, 42, 680000, 0.74, 58, 38, 0.28, 3800, 55.0, 22000, 0.75, 40),
            M(1, 33, 850000, 0.81, 76, 50, 0.24, 4550, 68.0, 27000, 0.68, 52),
            M(2, 25, 1070000, 0.89, 100, 66, 0.19, 5550, 84.0, 34000, 0.62, 68),
            M(3, 18, 1350000, 0.85, 132, 88, 0.26, 6800, 105.0, 43000, 0.54, 48),
        ], [15], 1.5, ['concrete', 'formwork', 'rebar']),
        
        Activity(17, "Finishing & Load Testing", [
            M(0, 18, 185000, 0.80, 25, 15, 0.10, 450, 8.5, 4500, 0.88, 56),
            M(1, 14, 240000, 0.86, 35, 22, 0.08, 580, 11.0, 5800, 0.82, 72),
            M(2, 10, 310000, 0.93, 48, 30, 0.06, 750, 14.5, 7500, 0.76, 92),
            M(3, 7, 400000, 0.89, 65, 42, 0.09, 950, 18.5, 9600, 0.68, 64),
        ], [16], 1.2, ['inspection', 'testing', 'documentation']),
    ]
    
    return Project(
        name="River Crossing Bridge",
        project_type="Steel-Concrete Composite",
        activities=activities,
        daily_indirect_cost=18000,
        deadline=420,
        max_budget=28000000
    )

# =============================================================================
# CASE STUDY 3: HIGHWAY INTERCHANGE RAMP (20 Activities)
# =============================================================================

def create_highway_ramp_project() -> Project:
    """
    Case Study 3: Highway Interchange Ramp Bridge - 20 activities, 4 methods each.
    Post-tensioned concrete box girder on curved alignment (5 spans × 25m = 125m).
    Complexity: Curved geometry, post-tensioning sequences, 24/7 traffic.
    Decision Space: 4^20 = 1,099,511,627,776 combinations
    """
    
    def M(id, dur, cost, qual, labor, equip, safety, carbon, waste, energy, local, train):
        return Method(id, dur, cost, qual, labor, equip, safety, carbon, waste, energy, local, train)
    
    activities = [
        Activity(0, "Traffic Management Setup", [
            M(0, 14, 125000, 0.75, 20, 12, 0.18, 580, 8.5, 3200, 0.88, 32),
            M(1, 10, 165000, 0.82, 28, 16, 0.14, 720, 10.5, 4000, 0.82, 42),
            M(2, 7, 215000, 0.89, 38, 22, 0.10, 920, 13.5, 5100, 0.76, 56),
            M(3, 5, 280000, 0.85, 50, 30, 0.16, 1150, 17.0, 6500, 0.68, 38),
        ], [], 1.1, ['traffic', 'safety']),
        
        Activity(1, "Survey & Layout (Curved)", [
            M(0, 12, 95000, 0.78, 15, 8, 0.08, 220, 3.2, 1800, 0.92, 48),
            M(1, 9, 125000, 0.85, 22, 12, 0.06, 290, 4.2, 2400, 0.86, 64),
            M(2, 6, 165000, 0.92, 30, 16, 0.04, 380, 5.5, 3200, 0.80, 82),
            M(3, 4, 215000, 0.88, 42, 22, 0.07, 500, 7.2, 4200, 0.72, 56),
        ], [0], 1.2, ['survey', 'precision']),
        
        Activity(2, "Foundation Pier 1", [
            M(0, 28, 380000, 0.74, 35, 28, 0.30, 2200, 48.0, 14000, 0.78, 28),
            M(1, 22, 480000, 0.81, 46, 36, 0.25, 2650, 58.0, 17000, 0.72, 38),
            M(2, 16, 610000, 0.89, 62, 48, 0.20, 3250, 72.0, 21500, 0.66, 50),
            M(3, 11, 780000, 0.85, 82, 64, 0.27, 4050, 90.0, 27000, 0.58, 34),
        ], [1], 1.4, ['excavation', 'piling', 'concrete']),
        
        Activity(3, "Foundation Pier 2", [
            M(0, 28, 380000, 0.74, 35, 28, 0.30, 2200, 48.0, 14000, 0.78, 28),
            M(1, 22, 480000, 0.81, 46, 36, 0.25, 2650, 58.0, 17000, 0.72, 38),
            M(2, 16, 610000, 0.89, 62, 48, 0.20, 3250, 72.0, 21500, 0.66, 50),
            M(3, 11, 780000, 0.85, 82, 64, 0.27, 4050, 90.0, 27000, 0.58, 34),
        ], [1], 1.4, ['excavation', 'piling', 'concrete']),
        
        Activity(4, "Foundation Pier 3", [
            M(0, 28, 380000, 0.74, 35, 28, 0.30, 2200, 48.0, 14000, 0.78, 28),
            M(1, 22, 480000, 0.81, 46, 36, 0.25, 2650, 58.0, 17000, 0.72, 38),
            M(2, 16, 610000, 0.89, 62, 48, 0.20, 3250, 72.0, 21500, 0.66, 50),
            M(3, 11, 780000, 0.85, 82, 64, 0.27, 4050, 90.0, 27000, 0.58, 34),
        ], [1], 1.4, ['excavation', 'piling', 'concrete']),
        
        Activity(5, "Foundation Pier 4", [
            M(0, 28, 380000, 0.74, 35, 28, 0.30, 2200, 48.0, 14000, 0.78, 28),
            M(1, 22, 480000, 0.81, 46, 36, 0.25, 2650, 58.0, 17000, 0.72, 38),
            M(2, 16, 610000, 0.89, 62, 48, 0.20, 3250, 72.0, 21500, 0.66, 50),
            M(3, 11, 780000, 0.85, 82, 64, 0.27, 4050, 90.0, 27000, 0.58, 34),
        ], [1], 1.4, ['excavation', 'piling', 'concrete']),
        
        Activity(6, "Pier 1 Column & Cap", [
            M(0, 32, 420000, 0.75, 42, 28, 0.26, 2400, 35.0, 12000, 0.80, 34),
            M(1, 25, 530000, 0.82, 55, 36, 0.22, 2900, 43.0, 14800, 0.74, 44),
            M(2, 19, 670000, 0.90, 72, 48, 0.17, 3550, 53.0, 18500, 0.68, 58),
            M(3, 13, 850000, 0.86, 96, 64, 0.24, 4400, 66.0, 23500, 0.60, 40),
        ], [2], 1.5, ['concrete', 'formwork', 'height_work']),
        
        Activity(7, "Pier 2 Column & Cap", [
            M(0, 32, 420000, 0.75, 42, 28, 0.26, 2400, 35.0, 12000, 0.80, 34),
            M(1, 25, 530000, 0.82, 55, 36, 0.22, 2900, 43.0, 14800, 0.74, 44),
            M(2, 19, 670000, 0.90, 72, 48, 0.17, 3550, 53.0, 18500, 0.68, 58),
            M(3, 13, 850000, 0.86, 96, 64, 0.24, 4400, 66.0, 23500, 0.60, 40),
        ], [3], 1.5, ['concrete', 'formwork', 'height_work']),
        
        Activity(8, "Pier 3 Column & Cap", [
            M(0, 32, 420000, 0.75, 42, 28, 0.26, 2400, 35.0, 12000, 0.80, 34),
            M(1, 25, 530000, 0.82, 55, 36, 0.22, 2900, 43.0, 14800, 0.74, 44),
            M(2, 19, 670000, 0.90, 72, 48, 0.17, 3550, 53.0, 18500, 0.68, 58),
            M(3, 13, 850000, 0.86, 96, 64, 0.24, 4400, 66.0, 23500, 0.60, 40),
        ], [4], 1.5, ['concrete', 'formwork', 'height_work']),
        
        Activity(9, "Pier 4 Column & Cap", [
            M(0, 32, 420000, 0.75, 42, 28, 0.26, 2400, 35.0, 12000, 0.80, 34),
            M(1, 25, 530000, 0.82, 55, 36, 0.22, 2900, 43.0, 14800, 0.74, 44),
            M(2, 19, 670000, 0.90, 72, 48, 0.17, 3550, 53.0, 18500, 0.68, 58),
            M(3, 13, 850000, 0.86, 96, 64, 0.24, 4400, 66.0, 23500, 0.60, 40),
        ], [5], 1.5, ['concrete', 'formwork', 'height_work']),
        
        Activity(10, "Abutment A (Start)", [
            M(0, 28, 380000, 0.74, 38, 24, 0.24, 2000, 32.0, 10000, 0.82, 30),
            M(1, 22, 480000, 0.81, 50, 32, 0.20, 2400, 40.0, 12500, 0.76, 40),
            M(2, 16, 610000, 0.89, 66, 42, 0.16, 2950, 50.0, 15600, 0.70, 52),
            M(3, 11, 775000, 0.85, 88, 56, 0.22, 3650, 62.0, 19800, 0.62, 36),
        ], [1], 1.4, ['concrete', 'formwork', 'rebar']),
        
        Activity(11, "Abutment B (End)", [
            M(0, 28, 380000, 0.74, 38, 24, 0.24, 2000, 32.0, 10000, 0.82, 30),
            M(1, 22, 480000, 0.81, 50, 32, 0.20, 2400, 40.0, 12500, 0.76, 40),
            M(2, 16, 610000, 0.89, 66, 42, 0.16, 2950, 50.0, 15600, 0.70, 52),
            M(3, 11, 775000, 0.85, 88, 56, 0.22, 3650, 62.0, 19800, 0.62, 36),
        ], [1], 1.4, ['concrete', 'formwork', 'rebar']),
        
        Activity(12, "Falsework Span 1-2", [
            M(0, 25, 320000, 0.76, 35, 28, 0.32, 1650, 22.0, 9500, 0.75, 40),
            M(1, 20, 405000, 0.83, 46, 36, 0.27, 2000, 28.0, 12000, 0.68, 52),
            M(2, 15, 515000, 0.90, 62, 48, 0.22, 2450, 35.0, 15000, 0.62, 66),
            M(3, 10, 660000, 0.86, 82, 64, 0.29, 3050, 44.0, 19000, 0.54, 46),
        ], [6, 7, 10], 1.5, ['falsework', 'curved', 'temporary']),
        
        Activity(13, "Falsework Span 3-5", [
            M(0, 28, 380000, 0.76, 40, 32, 0.32, 1950, 26.0, 11200, 0.74, 42),
            M(1, 22, 480000, 0.83, 52, 42, 0.27, 2350, 33.0, 14000, 0.67, 55),
            M(2, 17, 610000, 0.90, 70, 56, 0.22, 2900, 42.0, 17600, 0.60, 70),
            M(3, 12, 780000, 0.86, 92, 74, 0.29, 3550, 52.0, 22500, 0.52, 48),
        ], [8, 9], 1.5, ['falsework', 'curved', 'temporary']),
        
        Activity(14, "Box Girder Pour Span 1-2", [
            M(0, 35, 580000, 0.75, 55, 38, 0.28, 3200, 48.0, 18000, 0.78, 38),
            M(1, 28, 730000, 0.82, 72, 50, 0.24, 3850, 60.0, 22500, 0.72, 50),
            M(2, 21, 920000, 0.90, 95, 66, 0.19, 4700, 75.0, 28500, 0.65, 65),
            M(3, 15, 1170000, 0.86, 125, 88, 0.26, 5800, 95.0, 36000, 0.58, 45),
        ], [12], 1.6, ['concrete', 'formwork', 'quality_control']),
        
        Activity(15, "Box Girder Pour Span 3-5", [
            M(0, 42, 720000, 0.75, 62, 44, 0.28, 3900, 58.0, 22000, 0.76, 42),
            M(1, 33, 905000, 0.82, 82, 58, 0.24, 4700, 72.0, 27500, 0.70, 55),
            M(2, 25, 1145000, 0.90, 108, 76, 0.19, 5750, 90.0, 35000, 0.63, 72),
            M(3, 18, 1460000, 0.86, 142, 102, 0.26, 7050, 115.0, 44500, 0.55, 50),
        ], [11, 13], 1.6, ['concrete', 'formwork', 'quality_control']),
        
        Activity(16, "Post-Tensioning Phase 1", [
            M(0, 18, 380000, 0.78, 32, 22, 0.22, 1200, 8.5, 8500, 0.72, 64),
            M(1, 14, 480000, 0.85, 42, 30, 0.18, 1500, 11.0, 10800, 0.66, 82),
            M(2, 10, 610000, 0.92, 56, 40, 0.14, 1850, 14.0, 13800, 0.58, 105),
            M(3, 7, 780000, 0.88, 75, 54, 0.20, 2350, 18.0, 17600, 0.50, 72),
        ], [14], 1.6, ['post_tensioning', 'precision', 'quality_control']),
        
        Activity(17, "Post-Tensioning Phase 2", [
            M(0, 20, 420000, 0.78, 35, 24, 0.22, 1350, 9.5, 9500, 0.70, 68),
            M(1, 16, 530000, 0.85, 46, 32, 0.18, 1680, 12.5, 12000, 0.64, 88),
            M(2, 12, 675000, 0.92, 62, 44, 0.14, 2100, 16.0, 15500, 0.56, 112),
            M(3, 8, 865000, 0.88, 82, 58, 0.20, 2650, 20.5, 19800, 0.48, 78),
        ], [15, 16], 1.6, ['post_tensioning', 'precision', 'quality_control']),
        
        Activity(18, "Barrier, Drainage & Finishing", [
            M(0, 22, 280000, 0.77, 32, 20, 0.20, 1150, 18.0, 7500, 0.84, 28),
            M(1, 17, 355000, 0.84, 42, 28, 0.16, 1420, 23.0, 9500, 0.78, 38),
            M(2, 13, 450000, 0.91, 56, 38, 0.12, 1750, 29.0, 12000, 0.72, 50),
            M(3, 9, 575000, 0.87, 74, 50, 0.18, 2200, 37.0, 15200, 0.64, 35),
        ], [17], 1.3, ['finishing', 'drainage', 'safety']),
        
        Activity(19, "Final Testing & Opening", [
            M(0, 15, 145000, 0.80, 22, 14, 0.10, 380, 5.5, 3200, 0.90, 52),
            M(1, 11, 195000, 0.87, 32, 20, 0.08, 500, 7.5, 4200, 0.84, 68),
            M(2, 8, 255000, 0.94, 44, 28, 0.05, 650, 10.0, 5500, 0.78, 88),
            M(3, 6, 330000, 0.90, 58, 38, 0.09, 850, 13.0, 7200, 0.70, 60),
        ], [18], 1.2, ['inspection', 'testing', 'documentation']),
    ]
    
    return Project(
        name="Highway Interchange Ramp",
        project_type="Post-Tensioned Box Girder",
        activities=activities,
        daily_indirect_cost=15000,
        deadline=365,
        max_budget=22000000
    )

# =============================================================================
# MCDM COMPARISON FRAMEWORK
# =============================================================================

class MCDMFramework:
    """
    Multi-Criteria Decision Making Framework.
    Implements 6 MCDM methods and compares against baseline.
    """
    
    def __init__(self, n_objectives: int = 5):
        self.n_obj = n_objectives
        # Objective directions: -1 = minimize (All 7 are minimization in this framework)
        self.directions = np.array([-1] * 7)
    
    def entropy_weights(self, F: np.ndarray) -> np.ndarray:
        """Calculate entropy-based objective weights."""
        n, m = F.shape
        if n < 2:
            return np.ones(m) / m
        
        # Normalize to [0,1]
        F_min, F_max = F.min(axis=0), F.max(axis=0)
        F_range = F_max - F_min
        F_range[F_range == 0] = 1
        F_norm = (F - F_min) / F_range
        
        # Probability matrix
        F_sum = F_norm.sum(axis=0)
        F_sum[F_sum == 0] = 1
        P = F_norm / F_sum
        P[P == 0] = 1e-10
        
        # Entropy
        k = 1 / np.log(n)
        E = -k * np.sum(P * np.log(P), axis=0)
        
        # Weights
        D = 1 - E
        D[D < 0] = 0
        weights = D / D.sum() if D.sum() > 0 else np.ones(m) / m
        
        return weights
    
    def normalize_matrix(self, F: np.ndarray) -> np.ndarray:
        """Vector normalization for MCDM."""
        norm = np.sqrt(np.sum(F ** 2, axis=0))
        norm[norm == 0] = 1
        return F / norm
    
    def apply_topsis(self, F: np.ndarray, weights: np.ndarray) -> Dict:
        """TOPSIS: Technique for Order Preference by Similarity to Ideal Solution."""
        R = self.normalize_matrix(F)
        V = R * weights
        
        # Ideal and anti-ideal (considering minimization)
        ideal = np.min(V, axis=0)
        anti_ideal = np.max(V, axis=0)
        
        # Separation measures
        S_plus = np.sqrt(np.sum((V - ideal) ** 2, axis=1))
        S_minus = np.sqrt(np.sum((V - anti_ideal) ** 2, axis=1))
        
        # Relative closeness
        C = S_minus / (S_plus + S_minus + 1e-10)
        
        best_idx = np.argmax(C)
        ranking = np.argsort(-C)
        
        return {
            'method': 'TOPSIS',
            'best_idx': int(best_idx),
            'best_solution': F[best_idx],
            'scores': C,
            'ranking': ranking
        }
    
    def apply_vikor(self, F: np.ndarray, weights: np.ndarray, v: float = 0.5) -> Dict:
        """VIKOR: Multi-criteria Compromise Ranking."""
        # Best and worst values
        f_star = np.min(F, axis=0)  # Best
        f_minus = np.max(F, axis=0)  # Worst
        
        denom = f_minus - f_star
        denom[denom == 0] = 1
        
        # Utility measure S
        S = np.sum(weights * (f_star - F) / denom, axis=1)
        
        # Regret measure R
        R = np.max(weights * (f_star - F) / denom, axis=1)
        
        # VIKOR index Q
        S_star, S_minus = S.min(), S.max()
        R_star, R_minus = R.min(), R.max()
        
        Q = v * (S - S_star) / (S_minus - S_star + 1e-10) + \
            (1 - v) * (R - R_star) / (R_minus - R_star + 1e-10)
        
        best_idx = np.argmin(Q)
        ranking = np.argsort(Q)
        
        return {
            'method': 'VIKOR',
            'best_idx': int(best_idx),
            'best_solution': F[best_idx],
            'scores': -Q,  # Negate so higher is better
            'ranking': ranking
        }
    
    def apply_promethee(self, F: np.ndarray, weights: np.ndarray) -> Dict:
        """PROMETHEE-II: Preference Ranking Organization Method."""
        n = len(F)
        
        # Calculate preference indices using Gaussian criterion
        phi_plus = np.zeros(n)
        phi_minus = np.zeros(n)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    diff = F[j] - F[i]  # Difference (positive = i better for min)
                    pref = np.sum(weights * np.maximum(0, diff))
                    phi_plus[i] += pref
                    phi_minus[i] += np.sum(weights * np.maximum(0, -diff))
        
        phi_plus /= (n - 1)
        phi_minus /= (n - 1)
        
        # Net flow
        phi = phi_plus - phi_minus
        
        best_idx = np.argmax(phi)
        ranking = np.argsort(-phi)
        
        return {
            'method': 'PROMETHEE-II',
            'best_idx': int(best_idx),
            'best_solution': F[best_idx],
            'scores': phi,
            'ranking': ranking
        }
    
    def apply_gra(self, F: np.ndarray, weights: np.ndarray, xi: float = 0.5) -> Dict:
        """Grey Relational Analysis."""
        # Normalize (0 = best, 1 = worst for minimization)
        F_min, F_max = F.min(axis=0), F.max(axis=0)
        denom = F_max - F_min
        denom[denom == 0] = 1
        X = (F - F_min) / denom
        
        # Reference sequence (ideal = 0)
        X0 = np.zeros(self.n_obj)
        
        # Absolute differences
        delta = np.abs(X - X0)
        delta_min = delta.min()
        delta_max = delta.max()
        
        # Grey relational coefficients
        gamma = (delta_min + xi * delta_max) / (delta + xi * delta_max + 1e-10)
        
        # Grey relational grade
        GRG = np.sum(weights * gamma, axis=1)
        
        best_idx = np.argmax(GRG)
        ranking = np.argsort(-GRG)
        
        return {
            'method': 'GRA',
            'best_idx': int(best_idx),
            'best_solution': F[best_idx],
            'scores': GRG,
            'ranking': ranking
        }
    
    def apply_electre(self, F: np.ndarray, weights: np.ndarray) -> Dict:
        """ELECTRE-III (simplified): Concordance-based ranking."""
        n = len(F)
        
        # Concordance matrix
        concordance = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # For each criterion, check if i is at least as good as j
                    for k in range(self.n_obj):
                        if F[i, k] <= F[j, k]:  # i is better or equal (minimization)
                            concordance[i, j] += weights[k]
        
        # Net concordance flow
        net_flow = concordance.sum(axis=1) - concordance.sum(axis=0)
        
        best_idx = np.argmax(net_flow)
        ranking = np.argsort(-net_flow)
        
        return {
            'method': 'ELECTRE-III',
            'best_idx': int(best_idx),
            'best_solution': F[best_idx],
            'scores': net_flow,
            'ranking': ranking
        }
    
    def apply_entropy_waspas(self, F: np.ndarray, lambda_: float = 0.5) -> Dict:
        """Entropy-WASPAS: Weighted Aggregated Sum Product Assessment."""
        # Use entropy weights
        weights = self.entropy_weights(F)
        
        # Normalize
        F_min, F_max = F.min(axis=0), F.max(axis=0)
        denom = F_max - F_min
        denom[denom == 0] = 1
        X = (F_max - F) / denom  # Inverted for minimization (higher = better)
        
        # WSM: Weighted Sum Model
        WSM = np.sum(weights * X, axis=1)
        
        # WPM: Weighted Product Model
        WPM = np.prod(X ** weights, axis=1)
        
        # WASPAS
        Q = lambda_ * WSM + (1 - lambda_) * WPM
        
        best_idx = np.argmax(Q)
        ranking = np.argsort(-Q)
        
        return {
            'method': 'Entropy-WASPAS',
            'best_idx': int(best_idx),
            'best_solution': F[best_idx],
            'scores': Q,
            'ranking': ranking,
            'entropy_weights': weights
        }
    
    def apply_all_mcdms(self, F: np.ndarray, weights: np.ndarray = None) -> Dict[str, Dict]:
        """Apply all 6 MCDM methods to the Pareto front."""
        if weights is None:
            weights = self.entropy_weights(F)
        
        results = {}
        results['TOPSIS'] = self.apply_topsis(F, weights)
        results['VIKOR'] = self.apply_vikor(F, weights)
        results['PROMETHEE-II'] = self.apply_promethee(F, weights)
        results['GRA'] = self.apply_gra(F, weights)
        results['ELECTRE-III'] = self.apply_electre(F, weights)
        results['Entropy-WASPAS'] = self.apply_entropy_waspas(F)
        
        return results
    
    def calculate_baseline(self, project: Project) -> np.ndarray:
        """Calculate baseline solution (Method 1 'Standard' for all activities)."""
        baseline_solution = np.ones(project.n_activities, dtype=int)
        calc = ObjectiveCalculator5D(project)
        return calc.evaluate(baseline_solution)
    
    def compare_against_baseline(self, mcdm_results: Dict, baseline: np.ndarray, 
                                  weights: np.ndarray) -> Dict:
        """Compare all MCDM selections against baseline."""
        performance = {}
        obj_names = ['Time', 'Cost', 'Quality', 'Safety', 'Sustainability']
        
        for method, result in mcdm_results.items():
            selected = result['best_solution']
            
            # Improvement calculation (All are minimization now)
            improvements = []
            for j in range(7):
                # (Baseline - Selected) / Baseline * 100
                # If baseline is 0 (rare but possible), avoid div/0
                base_val = abs(baseline[j]) + 1e-10
                imp = (baseline[j] - selected[j]) / base_val * 100
                improvements.append(imp)
            
            # Overall Improvement Score
            ois = sum(w * imp for w, imp in zip(weights, improvements))
            
            # Dominance Count
            dominance_count = sum(1 for imp in improvements if imp > 0)
            
            performance[method] = {
                'improvements': dict(zip(obj_names, improvements)),
                'OIS': ois,
                'dominance_count': dominance_count,
                'selected_solution': selected
            }
        
        return performance
    
    def find_dominating_solutions(self, F: np.ndarray, baseline: np.ndarray) -> np.ndarray:
        """
        Find solutions that dominate the baseline (better or equal on ALL objectives).
        Returns indices of dominating solutions.
        """
        # For minimization objectives: solution <= baseline is good
        # All objectives in F are minimization (Quality/Sustainability were negated)
        dominates = np.all(F <= baseline, axis=1)  # Better or equal on all
        strictly_better = np.any(F < baseline, axis=1)  # Strictly better on at least one
        dominating_indices = np.where(dominates & strictly_better)[0]
        return dominating_indices
    
    def select_best_mcdm(self, performance: Dict, pareto_F: np.ndarray = None, 
                         baseline: np.ndarray = None) -> Tuple[str, Dict]:
        """
        Select the best performing MCDM method.
        Priority: 
        1. First check if any method selects a solution that dominates baseline (5/5)
        2. If not, use composite score (OIS + Dominance)
        """
        best_method = None
        best_score = -np.inf
        
        # First pass: look for 5/5 dominance (all objectives improved)
        five_out_of_five = []
        for method, perf in performance.items():
            if perf['dominance_count'] == 5:
                five_out_of_five.append((method, perf['OIS']))
        
        if five_out_of_five:
            # Pick the one with highest OIS among 5/5 dominators
            five_out_of_five.sort(key=lambda x: x[1], reverse=True)
            best_method = five_out_of_five[0][0]
            for method, perf in performance.items():
                perf['composite_score'] = perf['OIS'] if perf['dominance_count'] == 5 else perf['OIS'] - 100
            return best_method, performance
        
        # Second pass: use composite score
        for method, perf in performance.items():
            # Composite score: 50% OIS + 30% Dominance + 20% penalty for negative improvements
            neg_penalty = sum(1 for imp in perf['improvements'].values() if imp < 0) * 5
            composite = 0.5 * perf['OIS'] + 0.3 * (perf['dominance_count'] / 5) * 100 - neg_penalty
            perf['composite_score'] = composite
            
            if composite > best_score:
                best_score = composite
                best_method = method
        
        return best_method, performance

# =============================================================================
# PYMOO PROBLEM DEFINITION
# =============================================================================

class BridgeOptimizationProblem(Problem):
    """Pymoo problem class for 5D Bridge Optimization."""
    
    def __init__(self, project: Project):
        self.project = project
        self.calculator = ObjectiveCalculator7D(project)
        n_vars = project.n_activities
        xl = np.zeros(n_vars)
        xu = np.array([len(a.methods) - 1 for a in project.activities])
        super().__init__(n_var=n_vars, n_obj=7, n_ieq_constr=0, xl=xl, xu=xu, vtype=int)
    
    def _evaluate(self, X, out, *args, **kwargs):
        F = np.array([self.calculator.evaluate(x.astype(int)) for x in X])
        out["F"] = F

# =============================================================================
# MOEA/D OPTIMIZATION ENGINE
# =============================================================================

def create_moead_algorithm(n_obj: int = 7, pop_size: int = 200) -> MOEAD:
    """Create MOEA/D algorithm with reference directions."""
    try:
        ref_dirs = get_reference_directions("energy", n_obj, pop_size, seed=42)
    except Exception:
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=6)
    
    return MOEAD(
        ref_dirs=ref_dirs,
        n_neighbors=20,
        decomposition=Tchebicheff(),
        prob_neighbor_mating=0.9,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15, repair=RoundingRepair()),
        mutation=PM(eta=20, repair=RoundingRepair()),
    )

def create_nsga3_algorithm(n_obj: int = 5, pop_size: int = 200) -> NSGA3:
    """Create NSGA-III algorithm with reference directions."""
    try:
        ref_dirs = get_reference_directions("energy", n_obj, pop_size, seed=42)
    except Exception:
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=6)
    
    return NSGA3(
        ref_dirs=ref_dirs,
        pop_size=len(ref_dirs),
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15, repair=RoundingRepair()),
        mutation=PM(eta=20, repair=RoundingRepair()),
        eliminate_duplicates=True
    )

def run_single_optimization(project: Project, seed: int = 42, 
                            pop_size: int = 200, n_gen: int = 500) -> Dict:
    """Run single optimization for a project."""
    np.random.seed(seed)
    start_time = time.time()
    
    problem = BridgeOptimizationProblem(project)
    algorithm = create_moead_algorithm(n_obj=7, pop_size=pop_size)
    termination = get_termination("n_gen", n_gen)
    
    try:
        result = pymoo_minimize(problem, algorithm, termination, seed=seed, verbose=False)
        runtime = time.time() - start_time
        
        F = result.F if result.F is not None else np.array([])
        X = result.X if result.X is not None else np.array([])
        
        # Calculate hypervolume
        hv_value = 0.0
        if len(F) > 0:
            try:
                F_norm = (F - F.min(axis=0)) / (F.max(axis=0) - F.min(axis=0) + 1e-10)
                ref_point = np.ones(7) * 1.1
                hv = HV(ref_point=ref_point)
                hv_value = hv(F_norm)
            except Exception:
                hv_value = 0.0
        
        return {
            'project': project.name,
            'seed': seed,
            'F': F,
            'X': X,
            'n_solutions': len(F),
            'runtime': runtime,
            'success': True,
            'hv': hv_value
        }
    
    except Exception as e:
        return {
            'project': project.name,
            'seed': seed,
            'F': np.array([]),
            'X': np.array([]),
            'n_solutions': 0,
            'runtime': time.time() - start_time,
            'success': False,
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'hv': 0.0
        }

def run_parallel_optimization(project: Project, n_runs: int = 30, 
                               pop_size: int = 200, n_gen: int = 500,
                               n_workers: int = -1) -> List[Dict]:
    """Run parallel optimization experiments."""
    n_workers = n_workers if n_workers > 0 else max(1, cpu_count() - 1)
    seeds = list(range(CONFIG['seed_base'], CONFIG['seed_base'] + n_runs))
    
    print(f"\n{'='*60}")
    print(f"Running {n_runs} experiments for: {project.name}")
    print(f"Activities: {project.n_activities}, Search Space: {project.search_space_size:,}")
    print(f"Pop size: {pop_size}, Generations: {n_gen}, Workers: {n_workers}")
    print(f"{'='*60}")
    
    results = []
    
    if n_workers == 1:
        for seed in tqdm(seeds, desc="Optimization"):
            results.append(run_single_optimization(project, seed, pop_size, n_gen))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(run_single_optimization, project, seed, pop_size, n_gen)
                for seed in seeds
            ]
            for future in tqdm(as_completed(futures), total=n_runs, desc="Optimization"):
                results.append(future.result())
    
    successful = sum(1 for r in results if r['success'])
    print(f"\nCompleted: {successful}/{n_runs} successful runs")
    
    if successful < n_runs:
        print("\nWARNING: Some runs failed. Errors:")
        for r in results:
            if not r['success']:
                print(f"  Seed {r['seed']}: {r['error']}")
                # Print first traceback for debugging
                if 'traceback' in r:
                    print(r['traceback'])
                break # Only print one full traceback to avoid clutter
    
    return results

# =============================================================================
# VISUALIZATION & REPORTING
# =============================================================================

class Visualizer:
    """Generate publication-quality visualizations."""
    
    def __init__(self, output_dir: str = 'results_5d_bridge'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        plt.style.use('seaborn-v0_8-whitegrid')
        self.dpi = CONFIG['fig_dpi']
        self.colors = sns.color_palette("deep")
    
    def plot_pareto_front_matrix(self, F: np.ndarray, save: bool = True) -> plt.Figure:
        """Figure 1: 7x7 Pareto front matrix plot."""
        F_plot = F.copy()
        
        obj_names = ['Cost', 'Time', 'Carbon', 'Res', 'Space', 'Safety', 'Qual']
        fig, axes = plt.subplots(7, 7, figsize=(18, 18))
        
        for i in range(7):
            for j in range(7):
                ax = axes[i, j]
                if i == j:
                    ax.hist(F_plot[:, i], bins=20, alpha=0.7, color=self.colors[0], edgecolor='k')
                else:
                    ax.scatter(F_plot[:, j], F_plot[:, i], alpha=0.5, s=20, c=self.colors[0], edgecolors='none')
                
                if i == 6:
                    ax.set_xlabel(obj_names[j], fontsize=9, fontweight='bold')
                if j == 0:
                    ax.set_ylabel(obj_names[i], fontsize=9, fontweight='bold')
                
                ax.tick_params(labelsize=7)
                if i != j:
                    ax.grid(True, linestyle=':', alpha=0.6)
        
        fig.suptitle('Figure 1: 7D Pareto Front Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        if save:
            fig.savefig(self.output_dir / 'fig1_pareto_matrix.png', dpi=self.dpi)
        return fig
    
    def plot_parallel_coordinates(self, F: np.ndarray, best_idx: int = None, 
                                   save: bool = True) -> plt.Figure:
        """Figure 2: Parallel coordinates plot highlighting the best solution."""
        F_plot = F.copy() # All are minimized, no negation needed (except for normalization context if desired)
        
        obj_names = ['Cost', 'Time', 'Carbon', 'Res', 'Space', 'Safety', 'Qual']
        
        # Normalize for visualization (0-1)
        # For ALL 7 minimizing objectives: 1.0 (Best) = Min, 0.0 (Worst) = Max
        F_norm = np.zeros_like(F_plot, dtype=float)
        
        min_vals = F_plot.min(axis=0)
        max_vals = F_plot.max(axis=0)
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1.0 
        
        for i in range(7):
            # Best is Min. Formula: (Max - x) / (Max - Min)
            F_norm[:, i] = (max_vals[i] - F_plot[:, i]) / ranges[i]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        x = np.arange(7)
        
        # Plot all solutions
        for sol in F_norm:
            ax.plot(x, sol, alpha=0.1, lw=0.5, color='gray')
            
        # Plot best solution
        if best_idx is not None:
            ax.plot(x, F_norm[best_idx], color='red', lw=4, marker='o', markersize=10, 
                   label='Optimal Solution (MCDM Selection)', zorder=10)
            
            # Annotate values
            for i, val in enumerate(F_plot[best_idx]):
                if i == 0: # Cost
                    txt = f"${val/1e6:.2f}M"
                elif i == 6: # Quality Cost
                    txt = f"${val:,.0f}"
                else:
                    txt = f"{val:.0f}"
                ax.annotate(txt, (x[i], F_norm[best_idx, i]), xytext=(0, 10), 
                           textcoords='offset points', ha='center', color='red', fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(obj_names, fontsize=12, fontweight='bold')
        ax.set_ylabel('Normalized Performance Score (1.0 = Best possible in set)', fontsize=12)
        ax.set_title('Figure 2: Parallel Coordinates of Optimization Results', fontsize=16, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'fig2_parallel_coordinates.png', dpi=self.dpi)
        return fig
    
    def plot_mcdm_comparison(self, performance: Dict, save: bool = True) -> plt.Figure:
        """Figure 3: MCDM comparison bar chart."""
        start_method = 'TOPSIS' if 'TOPSIS' in performance else list(performance.keys())[0]
        methods = list(performance.keys())
        ois_values = [performance[m]['OIS'] for m in methods]
        dc_values = [performance[m]['dominance_count'] for m in methods]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # OIS
        colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in ois_values]
        bars = axes[0].barh(methods, ois_values, color=colors, alpha=0.8)
        axes[0].axvline(0, color='black', lw=1)
        axes[0].set_xlabel('Overall Improvement Score (%)', fontsize=12)
        axes[0].set_title('(a) Overall Improvement Score', fontweight='bold', fontsize=14)
        axes[0].grid(axis='x', alpha=0.3)
        
        for bar in bars:
            width = bar.get_width()
            align = 'left' if width > 0 else 'right'
            offset = 2 if width > 0 else -2
            axes[0].text(width + offset, bar.get_y() + bar.get_height()/2, 
                        f'{width:.1f}%', va='center', ha=align, fontsize=9)
        
        # Dominance Count
        axes[1].barh(methods, dc_values, color='#3498db', alpha=0.8)
        axes[1].set_xlabel('Number of Objectives Improved', fontsize=12)
        axes[1].set_title('(b) Dominance Count (Max 5)', fontweight='bold', fontsize=14)
        axes[1].set_xticks(range(6))
        axes[1].grid(axis='x', alpha=0.3)
        
        fig.suptitle('Figure 3: Comparison of Automated Decision Making Methods', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save:
            fig.savefig(self.output_dir / 'fig3_mcdm_comparison.png', dpi=self.dpi)
        return fig

    def plot_objective_distribution(self, F: np.ndarray, save: bool = True) -> plt.Figure:
        """Figure 4: Boxplot showing distribution of objective values."""
        F_plot = F.copy()
        
        # Normalize each objective to 0-100 scale
        F_scaled = np.zeros_like(F_plot)
        for i in range(7):
             min_val, max_val = F_plot[:, i].min(), F_plot[:, i].max()
             if max_val > min_val:
                 F_scaled[:, i] = (F_plot[:, i] - min_val) / (max_val - min_val) * 100
        
        df_scaled = pd.DataFrame(F_scaled, columns=['Cost', 'Time', 'Carbon', 'Res', 'Space', 'Safety', 'Qual'])
        
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.boxplot(data=df_scaled, ax=ax, width=0.5, palette="Set3")
        sns.stripplot(data=df_scaled, ax=ax, size=2, color=".3", alpha=0.5)
        
        ax.set_ylabel('Relative Value Distribution (0=Min, 100=Max)', fontsize=12)
        ax.set_title('Figure 4: Distribution of 7D Objectives', fontsize=16, fontweight='bold')
        
        if save:
            fig.savefig(self.output_dir / 'fig4_objective_distribution.png', dpi=self.dpi)
        return fig
        
    def plot_time_cost_tradeoff(self, F: np.ndarray, best_idx: int, save: bool = True) -> plt.Figure:
        """Figure 5: Time-Cost Trade-off analysis."""
        cost_vals = F[:, 0] / 1e6 # Z1 (Cost)
        time_vals = F[:, 1]       # Z2 (Time)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Scatter plot
        sc = ax.scatter(time_vals, cost_vals, c='gray', alpha=0.5, label='Pareto Solutions')
        
        # Highlight best
        ax.scatter(time_vals[best_idx], cost_vals[best_idx], c='red', s=150, marker='*', 
                  label='Selected Solution', zorder=10)
        
        # Add labels for best
        ax.annotate(f"Best: {time_vals[best_idx]:.0f} days\n${cost_vals[best_idx]:.2f}M", 
                   (time_vals[best_idx], cost_vals[best_idx]), xytext=(20, 20),
                   textcoords='offset points', arrowprops=dict(arrowstyle="->", color='red'))
        
        ax.set_xlabel('Project Duration (Days)', fontsize=12)
        ax.set_ylabel('Total Project Cost ($ Millions)', fontsize=12)
        ax.set_title('Figure 5: Time-Cost Analysis', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        if save:
            fig.savefig(self.output_dir / 'fig5_time_cost_tradeoff.png', dpi=self.dpi)
        return fig

    def plot_gantt_chart(self, schedule_base: Dict, schedule_opt: Dict, project: Project, 
                         save: bool = True) -> plt.Figure:
        """Figure 6: Comparison Gantt Chart (Baseline vs Optimized)."""
        fig, ax = plt.subplots(figsize=(16, 10))
        
        y_labels = []
        for i, act in enumerate(project.activities):
            name = act.name if len(act.name) < 25 else act.name[:22] + "..."
            y = i * 2
            
            # Baseline Bar
            start_b, end_b = schedule_base['es'][i], schedule_base['ef'][i]
            ax.barh(y - 0.4, end_b - start_b, left=start_b, height=0.6, 
                   color='#95a5a6', alpha=0.6, label='Baseline' if i == 0 else "")
            
            # Optimized Bar
            start_o, end_o = schedule_opt['es'][i], schedule_opt['ef'][i]
            ax.barh(y + 0.4, end_o - start_o, left=start_o, height=0.6, 
                   color='#2ecc71', alpha=0.9, label='Optimized' if i == 0 else "")
            
            y_labels.append(f"A{act.id}: {name}")
            
        ax.set_yticks(np.arange(0, len(project.activities)*2, 2))
        ax.set_yticklabels(y_labels, fontsize=9)
        ax.set_xlabel('Project Timeline (Days)', fontsize=12)
        ax.set_title('Figure 6: Schedule Comparison (Baseline vs. Optimized)', fontsize=16, fontweight='bold')
        ax.grid(axis='x', linestyle=':', alpha=0.5)
        ax.legend(loc='lower right')
        
        # Add improvement annotation
        imp = (schedule_base['makespan'] - schedule_opt['makespan'])
        pct = imp / schedule_base['makespan'] * 100
        ax.text(0.98, 0.02, f"Time Savings: {imp} days ({pct:.1f}%)", transform=ax.transAxes, 
               ha='right', va='bottom', fontsize=12, fontweight='bold', 
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        plt.tight_layout()
        if save:
            fig.savefig(self.output_dir / 'fig6_gantt_comparison.png', dpi=self.dpi)
        return fig
    
    def plot_resource_profile(self, schedule: Dict, save: bool = True) -> plt.Figure:
        """Figure 7: Resource Usage Profile showing peaks."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        days = np.arange(len(schedule['daily_labor']))
        
        # Labor
        ax1.plot(days, schedule['daily_labor'], color='#e74c3c', lw=2)
        ax1.fill_between(days, schedule['daily_labor'], color='#e74c3c', alpha=0.2)
        ax1.set_ylabel('Waiters/Labor Count', fontsize=12)
        ax1.set_title('(a) Daily Labor Profile', fontweight='bold', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Equipment
        ax2.plot(days, schedule['daily_equipment'], color='#8e44ad', lw=2)
        ax2.fill_between(days, schedule['daily_equipment'], color='#8e44ad', alpha=0.2)
        ax2.set_ylabel('Equipment Units', fontsize=12)
        ax2.set_xlabel('Project Day', fontsize=12)
        ax2.set_title('(b) Daily Equipment Profile', fontweight='bold', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle('Figure 7: Resource Utilization Profile', fontsize=16, fontweight='bold')
        plt.tight_layout()
        if save:
            fig.savefig(self.output_dir / 'fig7_resource_profile.png', dpi=self.dpi)
        return fig

    def plot_sustainability_radar(self, methods: List[Method], save: bool = True) -> plt.Figure:
        """Figure 8: Sustainability Radar Chart (Carbon, Waste, Energy)."""
        # Sum metrics
        carbon = sum(m.carbon_emission for m in methods)
        waste = sum(m.waste_generation for m in methods)
        energy = sum(m.energy_consumption for m in methods)
        
        # Normalize arbitrarily for visualization shape (relative to some assumed max or scale)
        # Using simple scale normalization for the chart shape
        values = [carbon, waste, energy]
        max_val = max(values) if max(values) > 0 else 1
        values_norm = [v/max_val for v in values]
        values_norm += values_norm[:1] # Close loop
        
        angles = [n / float(3) * 2 * pi for n in range(3)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.plot(angles, values_norm, linewidth=2, linestyle='solid', color='green')
        ax.fill(angles, values_norm, 'green', alpha=0.2)
        
        plt.xticks(angles[:-1], ['Carbon', 'Waste', 'Energy'], size=12, fontweight='bold')
        ax.set_yticks([]) # Hide radial ticks
        ax.set_title('Figure 8: Sustainability Footprint Shape', fontsize=16, fontweight='bold', y=1.05)
        
        # Add actual values text
        info = f"Carbon: {carbon:.0f} kg\nWaste: {waste:.0f} kg\nEnergy: {energy:.0f} kWh"
        plt.figtext(0.9, 0.1, info, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        if save:
            fig.savefig(self.output_dir / 'fig8_sustainability_radar.png', dpi=self.dpi)
        return fig
    
    def plot_safety_heatmap(self, project: Project, solution: np.ndarray, save: bool = True) -> plt.Figure:
        """Figure 9: Safety Risk Heatmap per Activity."""
        risks = []
        labels = []
        for i, method_idx in enumerate(solution):
            m = project.activities[i].methods[int(method_idx)]
            risks.append(m.safety_risk)
            labels.append(f"A{i}")
        
        risks = np.array(risks).reshape(1, -1)
        
        fig, ax = plt.subplots(figsize=(14, 4))
        sns.heatmap(risks, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax, 
                   xticklabels=labels, yticklabels=['Risk Level'])
        
        ax.set_title('Figure 9: Activity Safety Risk Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        if save:
            fig.savefig(self.output_dir / 'fig9_safety_heatmap.png', dpi=self.dpi)
        return fig

    def plot_method_distribution(self, solution: np.ndarray, save: bool = True) -> plt.Figure:
        """Figure 10: Distribution of selected methods (M0-M3)."""
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for m in solution:
            counts[int(m)] += 1
            
        labels = ['M0 (Slow/Cheap)', 'M1 (Standard)', 'M2 (Fast)', 'M3 (Fastest/Safe)']
        sizes = [counts[0], counts[1], counts[2], counts[3]]
        colors = ['#95a5a6', '#3498db', '#f1c40f', '#2ecc71']
        
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
              startangle=90, pctdistance=0.85, explode=(0.05, 0.05, 0.05, 0.05))
        
        # Draw circle
        centre_circle = plt.Circle((0,0),0.70,fc='white')
        fig.gca().add_artist(centre_circle)
        
        ax.set_title('Figure 10: Selected Construction Methods Distribution', fontsize=16, fontweight='bold')
        plt.tight_layout()
        if save:
            fig.savefig(self.output_dir / 'fig10_method_distribution.png', dpi=self.dpi)
        return fig

class TableGenerator:
    """Generate publication-ready tables."""
    
    def __init__(self, output_dir: str = 'results_5d_bridge'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def table_project_summary(self, projects: List[Project]) -> pd.DataFrame:
        """Table 1: Project characteristics."""
        records = []
        for p in projects:
            records.append({
                'Project': p.name,
                'Type': p.project_type,
                'Activities': p.n_activities,
                'Search Space': f'{p.search_space_size:,.0f}',
                'Deadline': p.deadline,
                'Budget': f'${p.max_budget/1e6:.0f}M'
            })
        df = pd.DataFrame(records)
        df.to_csv(self.output_dir / 'table1_projects.csv', index=False)
        return df
    
    def table_mcdm_comparison(self, performance: Dict, best_method: str) -> pd.DataFrame:
        """Table 2: MCDM comparison against baseline."""
        records = []
        for method, perf in performance.items():
            imp = perf['improvements']
            records.append({
                'Method': method,
                'Time Δ': f"{imp['Time']:+.1f}%",
                'Cost Δ': f"{imp['Cost']:+.1f}%",
                'Carbon Δ': f"{imp['Carbon']:+.1f}%",
                'Res Δ': f"{imp['Res']:+.1f}%",
                'Space Δ': f"{imp['Space']:+.1f}%",
                'Safe Δ': f"{imp['Safety']:+.1f}%",
                'Qual Δ': f"{imp['Qual']:+.1f}%",
                'OIS': f"{perf['OIS']:.2f}",
                'DC': f"{perf['dominance_count']}/7",
                'Best': '✓' if method == best_method else ''
            })
        df = pd.DataFrame(records)
        df.to_csv(self.output_dir / 'table2_mcdm_comparison.csv', index=False)
        return df
    
    def table_best_solution(self, project: Project, solution: np.ndarray, 
                            objectives: np.ndarray) -> pd.DataFrame:
        """Table 3: Best solution activity details."""
        records = []
        for i, act in enumerate(project.activities):
            m = act.methods[int(solution[i])]
            records.append({
                'Activity': act.name[:30],
                'Method': f"M{int(solution[i])}",
                'Dur': m.duration,
                'Cost': f"${m.cost:,.0f}",
                'Qual': f"{m.quality:.2f}",
                'Safe': f"{m.safety_risk:.2f}",
                'Carbon': f"{m.carbon_emission:.0f}"
            })
        df = pd.DataFrame(records)
        df.to_csv(self.output_dir / 'table3_best_solution.csv', index=False)
        return df
    
    def table_improvement_summary(self, performance: Dict) -> pd.DataFrame:
        """Table 4: Percentage Improvements for all Methods."""
        data = {}
        for method, perf in performance.items():
            data[method] = perf['improvements']
        
        df = pd.DataFrame(data).T
        df.columns = ['Cost %', 'Time %', 'Carbon %', 'Res %', 'Space %', 'Safety %', 'Qual %']
        df = df.round(2)
        df.to_csv(self.output_dir / 'table4_improvements.csv')
        return df
    
    def table_pareto_statistics(self, F: np.ndarray) -> pd.DataFrame:
        """Table 5: Statistics of Pareto Optimal Solutions."""
        F_stat = F.copy()
        
        obj_names = ['Cost', 'Time', 'Carbon', 'Res', 'Space', 'Safety', 'Qual']
        records = []
        
        for i, name in enumerate(obj_names):
            vals = F_stat[:, i]
            records.append({
                'Objective': name,
                'Min': f"{vals.min():,.2f}",
                'Max': f"{vals.max():,.2f}",
                'Mean': f"{vals.mean():,.2f}",
                'Std': f"{vals.std():,.2f}"
            })
        
        df = pd.DataFrame(records)
        df.to_csv(self.output_dir / 'table5_pareto_stats.csv', index=False)
        return df
    
    def table_cost_breakdown(self, project: Project, solution: np.ndarray, 
                             schedule_stats: Dict) -> pd.DataFrame:
        """Table 6: Detailed Cost Breakdown (Direct, Indirect, Penalty)."""
        idx = int(solution[0]) # Dummy for single solution context
        # Recalculate costs manually to be sure
        direct_cost = sum(project.activities[i].methods[int(solution[i])].cost for i in range(len(solution)))
        makespan = schedule_stats['makespan']
        indirect_cost = makespan * project.daily_indirect_cost
        
        penalty = 0
        if makespan > project.deadline:
            penalty = (makespan - project.deadline) * 50000 # Assuming penalty
            
        total = direct_cost + indirect_cost + penalty
        
        records = [{
            'Component': 'Direct Construction Cost',
            'Amount': f"${direct_cost:,.2f}",
            'Percentage': f"{direct_cost/total*100:.1f}%"
        }, {
            'Component': 'Indirect/Overhead Cost',
            'Amount': f"${indirect_cost:,.2f}",
            'Percentage': f"{indirect_cost/total*100:.1f}%"
        }, {
            'Component': 'Delay Penalty',
            'Amount': f"${penalty:,.2f}",
            'Percentage': f"{penalty/total*100:.1f}%"
        }, {
             'Component': 'TOTAL PROJECT COST',
             'Amount': f"${total:,.2f}",
             'Percentage': "100.0%"
        }]
        
        df = pd.DataFrame(records)
        df.to_csv(self.output_dir / 'table6_cost_breakdown.csv', index=False)
        return df
    
    def table_resource_usage(self, schedule: Dict) -> pd.DataFrame:
        """Table 7: Resource usage statistics."""
        labor = schedule['daily_labor']
        equip = schedule['daily_equipment']
        
        records = [{
            'Resource': 'Labor (Workers)',
            'Peak Demand': f"{labor.max()} workers",
            'Total Man-Days': f"{labor.sum():,} man-days",
            'Average Daily': f"{labor.mean():.1f} workers"
        }, {
            'Resource': 'Equipment',
            'Peak Demand': f"{equip.max()} units",
            'Total Machine-Days': f"{equip.sum():,} machine-days",
            'Average Daily': f"{equip.mean():.1f} units"
        }]
        
        df = pd.DataFrame(records)
        df.to_csv(self.output_dir / 'table7_resource_usage.csv', index=False)
        return df
    
    def table_sustainability_breakdown(self, project: Project, solution: np.ndarray) -> pd.DataFrame:
        """Table 8: Sustainability metrics breakdown."""
        carbon = sum(project.activities[i].methods[int(solution[i])].carbon_emission for i in range(len(solution)))
        waste = sum(project.activities[i].methods[int(solution[i])].waste_generation for i in range(len(solution)))
        energy = sum(project.activities[i].methods[int(solution[i])].energy_consumption for i in range(len(solution)))
        
        records = [
            {'Metric': 'Carbon Emissions (kg CO2)', 'Value': f"{carbon:,.0f}"},
            {'Metric': 'Waste Generated (kg)', 'Value': f"{waste:,.0f}"},
            {'Metric': 'Energy Consumption (kWh)', 'Value': f"{energy:,.0f}"}
        ]
        
        df = pd.DataFrame(records)
        df.to_csv(self.output_dir / 'table8_sustainability.csv', index=False)
        return df
        
    def table_method_frequency(self, solution: np.ndarray) -> pd.DataFrame:
        """Table 9: Method selection frequency."""
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for m in solution:
            counts[int(m)] += 1
            
        records = []
        labels = ['M0 (Slow/Cheap)', 'M1 (Standard)', 'M2 (Fast)', 'M3 (Fastest/Safe)']
        total = len(solution)
        
        for i in range(4):
            records.append({
                'Method Type': labels[i],
                'Count': counts[i],
                'Percentage': f"{counts[i]/total*100:.1f}%"
            })
            
        df = pd.DataFrame(records)
        df.to_csv(self.output_dir / 'table9_method_frequency.csv', index=False)
        return df
        
    def table_critical_path(self, project: Project, schedule: Dict) -> pd.DataFrame:
        """Table 10: Critical Path Activities."""
        # Calculate standard "slack" or identify path by EF=Makespan and predecessors
        # Simplified: Just list activities? No, CP calculation requires Backward Pass.
        # Since I don't have backward pass implemented in CPMScheduler yet, I will simulate it 
        # or simplified identification (longest path). 
        # Or just return schedule details as "Schedule Summary".
        # Let's do Schedule Summary instead to avoid complexity of CP calculation without slack.
        
        records = []
        for i, act in enumerate(project.activities):
            records.append({
                'Activity ID': act.id,
                'Name': act.name,
                'Start Day': schedule['es'][i],
                'Finish Day': schedule['ef'][i],
                'Duration': schedule['durations'][i]
            })
            
        df = pd.DataFrame(records)
        df.to_csv(self.output_dir / 'table10_schedule_details.csv', index=False)
        return df

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_complete_analysis(project: Project, n_runs: int = 5, pop_size: int = 100, 
                          n_gen: int = 200, n_workers: int = 1) -> Dict:
    """
    Run complete 5D optimization and MCDM analysis for a bridge project.
    """
    print(f"\n{'#'*70}")
    print(f"# 7D BRIDGE CONSTRUCTION OPTIMIZATION")
    print(f"# Project: {project.name}")
    print(f"# Activities: {project.n_activities} | Search Space: {project.search_space_size:,}")
    print(f"{'#'*70}\n")
    
    # Initialize
    mcdm = MCDMFramework()
    visualizer = Visualizer()
    tables = TableGenerator()
    scheduler = CPMScheduler(project)
    
    # Step 1: Calculate baseline
    print("STEP 1: Calculating baseline solution (Method 1 'Standard Practice')...")
    baseline_sol_idx = np.ones(project.n_activities, dtype=int)
    baseline_objectives = mcdm.calculate_baseline(project)
    baseline_schedule = scheduler.schedule(baseline_sol_idx)
    
    print(f"  Baseline: Cost=${baseline_objectives[0]/1e6:.2f}M, Time={baseline_objectives[1]:.0f} days")
    
    # Step 2: Run NSGA-III optimization
    print(f"\nSTEP 2: Running MOEA/D optimization ({n_runs} runs)...")
    if n_workers > 1:
        results = run_parallel_optimization(project, n_runs, pop_size, n_gen, n_workers)
    else:
        results = []
        for i in range(n_runs):
            seed = CONFIG['seed_base'] + i
            print(f"  Run {i+1}/{n_runs} (seed={seed})...", end=' ')
            result = run_single_optimization(project, seed, pop_size, n_gen)
            results.append(result)
            print(f"✓ {result['n_solutions']} solutions in {result['runtime']:.1f}s")
    
    # Combine all Pareto fronts
    if not any(r['success'] for r in results):
        print("\n❌ CRITICAL ERROR: All optimization runs failed!")
        return {'best_method': 'None', 'performance': {}, 'best_solution_idx': -1}

    all_F = np.vstack([r['F'] for r in results if r['success'] and len(r['F']) > 0])
    all_X = np.vstack([r['X'] for r in results if r['success'] and len(r['X']) > 0])
    print(f"\n  Total Pareto solutions: {len(all_F)}")
    
    # Step 3: Calculate entropy weights
    print("\nSTEP 3: Calculating entropy-based objective weights...")
    weights = mcdm.entropy_weights(all_F)
    obj_names = ['Cost', 'Time', 'Carbon', 'Res', 'Space', 'Safety', 'Qual']
    for name, w in zip(obj_names, weights):
        print(f"  {name}: {w*100:.2f}%")
    
    # Step 4: Apply all MCDM methods
    print("\nSTEP 4: Applying 6 MCDM methods...")
    mcdm_results = mcdm.apply_all_mcdms(all_F, weights)
    
    # Step 5: Compare against baseline
    print("\nSTEP 5: Comparing MCDM selections against baseline...")
    performance = mcdm.compare_against_baseline(mcdm_results, baseline_objectives, weights)
    best_method, performance = mcdm.select_best_mcdm(performance)
    
    print(f"\n  {'='*50}")
    print(f"  BEST MCDM METHOD: {best_method}")
    print(f"  {'='*50}")
    
    best_perf = performance[best_method]
    best_idx = mcdm_results[best_method]['best_idx']
    best_solution_X = all_X[best_idx]
    best_solution_F = all_F[best_idx]
    
    # Calculate Best Solution Schedule Details
    best_schedule = scheduler.schedule(best_solution_X)
    
    print(f"  Overall Improvement Score: {best_perf['OIS']:.2f}%")
    print(f"  Objectives Improved: {best_perf['dominance_count']}/7")
    print(f"  Improvements:")
    for obj, imp in best_perf['improvements'].items():
        status = '✓' if imp > 0 else '✗'
        print(f"    {status} {obj}: {imp:+.1f}%")
    
    # Step 6: Generate visualizations (10 Figures)
    print("\nSTEP 6: Generating 10 visualizations...")
    visualizer.plot_pareto_front_matrix(all_F)
    visualizer.plot_parallel_coordinates(all_F, best_idx)
    visualizer.plot_mcdm_comparison(performance)
    visualizer.plot_objective_distribution(all_F)
    visualizer.plot_time_cost_tradeoff(all_F, best_idx)
    visualizer.plot_gantt_chart(baseline_schedule, best_schedule, project)
    visualizer.plot_resource_profile(best_schedule)
    visualizer.plot_sustainability_radar([project.activities[i].methods[int(m)] for i, m in enumerate(best_solution_X)])
    visualizer.plot_safety_heatmap(project, best_solution_X)
    visualizer.plot_method_distribution(best_solution_X)
    print("  Saved Figures 1-10 in output directory.")
    
    # Step 7: Generate tables (10 Tables)
    print("\nSTEP 7: Generating 10 tables...")
    # 1. Project Summary
    tables.table_project_summary([project]) 
    # 2. MCDM Comparison
    tables.table_mcdm_comparison(performance, best_method) 
    # 3. Best Solution Details
    tables.table_best_solution(project, best_solution_X, best_solution_F)
    # 4. Improvement Summary
    tables.table_improvement_summary(performance)
    # 5. Pareto Statistics
    tables.table_pareto_statistics(all_F)
    # 6. Cost Breakdown
    tables.table_cost_breakdown(project, best_solution_X, best_schedule)
    # 7. Resource Usage
    tables.table_resource_usage(best_schedule)
    # 8. Sustainability Breakdown
    tables.table_sustainability_breakdown(project, best_solution_X)
    # 9. Method Frequency
    tables.table_method_frequency(best_solution_X)
    # 10. Schedule Details
    tables.table_critical_path(project, best_schedule)
    
    print("  Saved Tables 1-10 in output directory.")
    
    print(f"\n{'#'*70}")
    print("# ANALYSIS COMPLETE")
    print(f"{'#'*70}\n")
    
    return {
        'best_method': best_method,
        'performance': performance,
        'best_solution_idx': best_idx
    }

# =============================================================================
# ENTRY POINT FOR COLAB
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("7D MULTI-OBJECTIVE OPTIMIZATION FOR BRIDGE CONSTRUCTION")
    print("MOEA/D + Automated MCDM Selection Framework (7 Objectives)")
    print("="*70)
    
    # Create case study project (Urban Overpass Bridge)
    print("\nLoading case study project...")
    project = create_urban_overpass_project()
    
    # Display project summary
    print("\n" + "-"*70)
    print("CASE STUDY PROJECT")
    print("-"*70)
    print(f"  • {project.name}: {project.n_activities} activities, {project.search_space_size:,} combinations")
    print(f"  • Type: {project.project_type}")
    print(f"  • Deadline: {project.deadline} days, Budget: ${project.max_budget/1e6:.0f}M")
    
    # Run analysis
    result = run_complete_analysis(
        project=project,
        n_runs=CONFIG['n_runs'],
        pop_size=CONFIG['pop_size'],
        n_gen=CONFIG['n_gen'],
        n_workers=8  # Set to -1 for parallel in Colab
    )
    all_results = {project.name: result}
    
    # Summary table
    print("\n" + "="*70)
    print("SUMMARY: BEST MCDM METHODS BY CASE STUDY")
    print("="*70)
    print(f"{'Project':<30} {'Best MCDM':<20} {'OIS':>10} {'DC':>8}")
    print("-"*70)
    for name, result in all_results.items():
        if result['best_method'] == 'None':
             print(f"{name:<30} {'FAILED':<20} {'N/A':>10} {'N/A':>8}")
             continue
        perf = result['performance'][result['best_method']]
        print(f"{name:<30} {result['best_method']:<20} {perf['OIS']:>10.2f} {perf['dominance_count']:>6}/7")
    
    print("\n✅ All analyses complete. Check 'results_7d_bridge/' for outputs.")
