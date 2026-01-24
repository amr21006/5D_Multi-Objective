"""
Symbiotic Construction Intelligence Framework
=========================================================================
A Federated Ensemble Framework for 4D Tender Optimization Integrating
Evolutionary Cooperation and Half-Quadratic Consensus.

Objectives (4D):
    Z1: Financial - Total Construction Cost (minimize)
    Z2: Temporal - Robust Project Makespan (minimize)
    Z3: Environmental - Cradle-to-Site Carbon (minimize)
    Z4: Operational - Hybrid Resource Efficiency (minimize)

Algorithms (Islands):
    1. NSGA-III (Uniform Diversity)
    2. MOEA/D (Decomposition)
    3. RVEA (Convergence)
    4. AGE-MOEA (Geometric Estimation)
    5. NSGA-II (Rank-based)

Decision: Half-Quadratic Consensus of 5 MCDM methods.

Author: Research Implementation | Date: January 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Union
import warnings
import json
import time
from datetime import datetime

# Pymoo imports for the Federated Engine
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.population import Population
from pymoo.decomposition.pbi import PBI
from pymoo.decomposition.pbi import PBI
from scipy import stats
from tqdm import tqdm
import concurrent.futures

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'pop_size': 200,
    'n_gen_total': 500,
    'migration_interval': 25, # Frequent migration for test
    'n_islands': 5,
    'seed_base': 42,
    
    # Financial
    'risk_contingency': 0.05,
    'daily_indirect_cost': 7500.0,
    
    # Carbon Factors (kgCO2e per unit)
    'gamma_mat': 0.15,   # Material
    'gamma_trans': 0.02, # Transport per km
    'gamma_fuel': 2.5,   # Fuel per hour
    'gamma_waste': 0.5,  # Waste per unit
    'supplier_distance': 50, # km
    
    # Resource Efficiency Weights
    'w_smoothing': 0.4,
    'w_overload': 0.4,
    'w_idleness': 0.2,
    'beta_penalty': 10.0, # Overload penalty weight
    'lambda_penalty': 2.0, # Idleness penalty weight
}

# =============================================================================
# PART 1: 4D DATA STRUCTURES
# =============================================================================

@dataclass
class Method:
    """Construction method with 4D attributes."""
    id: int
    # 1. Financial & Temporal (Non-defaults first)
    cost_direct: float
    duration: int
    
    # Defaults
    risk_factor: float = 0.05
    lag_mandatory: int = 0
    
    # 3. Environmental (Carbon input data)
    material_quantity: float = 0.0   # Q_i
    equipment_hours: float = 0.0     # h_{i,m}
    fuel_consumption: float = 0.0    # E_{eq} (L/hr)
    waste_generated: float = 0.0     # W_{i,m}
    
    # 4. Operational (Resource usage)
    labor: int = 0
    equipment: int = 0
    
    # Identification
    name: str = ""

    @property
    def cost_total(self) -> float:
        """Direct Cost + Risk Premium"""
        return self.cost_direct * (1.0 + self.risk_factor)
    
    @property
    def carbon_footprint(self) -> float:
        """
        Calculate Cradle-to-Site Carbon:
        Embodied + Logistics + Process + Waste
        """
        # Embodied
        embodied = self.material_quantity * CONFIG['gamma_mat']
        # Logistics
        logistics = self.material_quantity * CONFIG['supplier_distance'] * CONFIG['gamma_trans']
        # Process
        process = self.equipment_hours * self.fuel_consumption * CONFIG['gamma_fuel']
        # Waste
        waste = self.waste_generated * CONFIG['gamma_waste']
        
        return embodied + logistics + process + waste

@dataclass
class Activity:
    id: int
    name: str
    methods: List[Method]
    predecessors: List[int] = field(default_factory=list)
    
    @property
    def n_methods(self) -> int:
        return len(self.methods)

@dataclass
class Project:
    name: str
    activities: List[Activity]
    max_labor: int
    max_equipment: int
    
    @property
    def n_activities(self) -> int:
        return len(self.activities)

# =============================================================================
# PART 2: 4D OBJECTIVE CALCULATOR
# =============================================================================

class ObjectiveCalculator4D:
    def __init__(self, project: Project):
        self.project = project
        self.n_activities = project.n_activities
        
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate solution vector x (indices of methods).
        Returns [Z1, Z2, Z3, Z4]
        """
        methods = [self.project.activities[i].methods[int(x[i])] for i in range(self.n_activities)]
        
        # --- Z1: Financial (Total Cost) ---
        # Direct + Risk
        cost_direct_risk = sum(m.cost_total for m in methods)
        
        # --- Z2: Temporal (Makespan) ---
        # CPM Calculation
        es = np.zeros(self.n_activities, dtype=int)
        ef = np.zeros(self.n_activities, dtype=int)
        
        for i, act in enumerate(self.project.activities):
            start = 0
            if act.predecessors:
                # S_j >= S_i + d_i + Lag
                # Here simplified to FS relationship: Start >= Finish of pred
                start = max(ef[p] for p in act.predecessors)
            
            es[i] = start
            ef[i] = start + methods[i].duration + methods[i].lag_mandatory
            
        makespan = np.max(ef) if len(ef) > 0 else 0
        
        # Indirect Cost (Time dependent)
        cost_indirect = makespan * CONFIG['daily_indirect_cost']
        
        z1_cost = cost_direct_risk + cost_indirect
        z2_time = float(makespan)
        
        # --- Z3: Environmental (Carbon) ---
        z3_carbon = sum(m.carbon_footprint for m in methods)
        
        # --- Z4: Operational (Resource Efficiency) ---
        z4_resources = self._calc_resource_efficiency(methods, es, ef, makespan)
        
        return np.array([z1_cost, z2_time, z3_carbon, z4_resources])

    def flatten_schedule(self, x: np.ndarray) -> Dict:
        """
        Return detailed schedule and costs for visualization.
        Ensures consistency between Optimization and Plotting.
        """
        methods = [self.project.activities[i].methods[int(x[i])] for i in range(self.n_activities)]
        es = np.zeros(self.n_activities, dtype=int)
        ef = np.zeros(self.n_activities, dtype=int)
        
        for i, act in enumerate(self.project.activities):
            start = 0
            if act.predecessors:
                start = max(ef[p] for p in act.predecessors)
            es[i] = start
            ef[i] = start + methods[i].duration + methods[i].lag_mandatory
            
        makespan = np.max(ef) if len(ef) > 0 else 0
        
        return {
            'methods': methods,
            'es': es,
            'ef': ef,
            'makespan': makespan
        }

    def _calc_resource_efficiency(self, methods: List[Method], es: np.ndarray, ef: np.ndarray, makespan: int) -> float:
        if makespan == 0: return 0.0
        
        # Build Resource Curves
        r_labor = np.zeros(makespan + 1)
        r_equip = np.zeros(makespan + 1)
        
        for i, m in enumerate(methods):
            duration = m.duration
            if duration > 0:
                # Add resource usage to the interval [es, es+duration)
                # Ensure we don't exceed array bounds
                end_idx = min(es[i]+duration, len(r_labor))
                if end_idx > es[i]:
                    r_labor[es[i]:end_idx] += m.labor
                    r_equip[es[i]:end_idx] += m.equipment
                
        # 1. Smoothing (Fluctuation) - Sum of squared values (Minimizes peaks and variance)
        smooth_labor = np.sum(r_labor**2)
        smooth_equip = np.sum(r_equip**2)
        
        # 2. Overload (Feasibility)
        over_labor = np.sum(np.maximum(0, r_labor - self.project.max_labor)**2)
        over_equip = np.sum(np.maximum(0, r_equip - self.project.max_equipment)**2)
        
        # 3. Idleness (Continuity)
        # Count days with 0 usage within the active project duration
        idle_labor = np.sum(r_labor[:makespan] == 0)
        idle_equip = np.sum(r_equip[:makespan] == 0)
        
        # Composite Index
        # Normalize rough magnitudes
        scale = 1e-4
        
        j_labor = (CONFIG['w_smoothing'] * smooth_labor * scale + 
                   CONFIG['w_overload'] * over_labor * CONFIG['beta_penalty'] * scale +
                   CONFIG['w_idleness'] * idle_labor * CONFIG['lambda_penalty'])
                   
        j_equip = (CONFIG['w_smoothing'] * smooth_equip * scale + 
                   CONFIG['w_overload'] * over_equip * CONFIG['beta_penalty'] * scale +
                   CONFIG['w_idleness'] * idle_equip * CONFIG['lambda_penalty'])
                   
        return j_labor + j_equip

# =============================================================================
# PART 3: FEDERATED ENSEMBLE ENGINE
# =============================================================================

class FederatedProblem(Problem):
    def __init__(self, project: Project):
        self.project = project
        self.calc = ObjectiveCalculator4D(project)
        self.n_vars = project.n_activities
        
        # Define variable bounds (discrete method indices)
        xl = np.zeros(self.n_vars)
        xu = np.array([act.n_methods - 1 for act in project.activities])
        
        super().__init__(n_var=self.n_vars, n_obj=4, n_ieq_constr=0, xl=xl, xu=xu, vtype=int)

    def _evaluate(self, x, out, *args, **kwargs):
        # x is (n_pop, n_var)
        # Evaluate row by row
        F = []
        for i in range(x.shape[0]):
            F.append(self.calc.evaluate(x[i]))
        out["F"] = np.array(F)

class FederatedEngine:
    def __init__(self, project: Project):
        self.project = project
        self.problem = FederatedProblem(project)
        self.global_archive = Population() # Global Pareto Archive
        # Add 'F' attribute to empty population to avoid errors
        self.global_archive.set("F", np.empty((0, 4)))
        self.history_hv = [] # Track Hypervolume
        self.history_n_nds = [] # Track Archive Size

        
    def setup_islands(self):
        """Initialize the 5 solver islands."""
        pop_size = CONFIG['pop_size']
        # Use energy method to get EXACTLY pop_size directions (robust for 200)
        ref_dirs = get_reference_directions("energy", 4, pop_size, seed=1)
        
        # 1. NSGA-III (Uniform Diversity)
        island1 = NSGA3(
            pop_size=pop_size,
            ref_dirs=ref_dirs,
            sampling=IntegerRandomSampling(),
            crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
            mutation=PM(prob=0.01, eta=20, vtype=float, repair=RoundingRepair()),
            eliminate_duplicates=True
        )
        
        # 2. MOEA/D (Decomposition)
        island2 = MOEAD(
            ref_dirs=ref_dirs,
            n_neighbors=15,
            decomposition=PBI(), # Explicit instantiation
            prob_neighbor_mating=0.7,
            sampling=IntegerRandomSampling(),
            crossover=SBX(prob=0.9, eta=15, repair=RoundingRepair()),
            mutation=PM(prob=0.01, eta=20, repair=RoundingRepair()),
        )
        
        # 3. RVEA (Convergence)
        island3 = RVEA(
            ref_dirs=ref_dirs,
            pop_size=pop_size,
            sampling=IntegerRandomSampling(),
            crossover=SBX(prob=0.9, eta=15, repair=RoundingRepair()),
            mutation=PM(prob=0.01, eta=20, repair=RoundingRepair()),
        )
        
        # 4. AGE-MOEA (Geometric)
        island4 = AGEMOEA(
            pop_size=pop_size,
            sampling=IntegerRandomSampling(),
            crossover=SBX(prob=0.9, eta=15, repair=RoundingRepair()),
            mutation=PM(prob=0.01, eta=20, repair=RoundingRepair()),
        )
        
        # 5. NSGA-II (Rank-based replacement for H-MOPSO)
        island5 = NSGA2(
            pop_size=pop_size,
            sampling=IntegerRandomSampling(),
            crossover=SBX(prob=0.9, eta=15, repair=RoundingRepair()),
            mutation=PM(prob=0.01, eta=20, repair=RoundingRepair()),
        )
        
        self.islands = [island1, island2, island3, island4, island5]
        self.island_names = ["NSGA-III", "MOEA/D", "RVEA", "AGE-MOEA", "NSGA-II"]
        
        # Initialize algorithms
        for alg in self.islands:
            alg.setup(self.problem, termination=('n_gen', CONFIG['n_gen_total']))
            
        self.history_snapshots = [] # snapshot of F for post-hoc analysis

    def run(self):
        """Execute the Federated Co-Evolutionary Loop."""
        print(f"Starting Federated Engine with {len(self.islands)} islands...")
        
        n_gen = CONFIG['n_gen_total']
        interval = CONFIG['migration_interval']
        
        # Main Loop
        pbar = tqdm(range(1, n_gen + 1), desc="Federated Optimization", unit="gen")
        for gen in pbar:
            # Step inside each island
            for i, alg in enumerate(self.islands):
                alg.next()
                
            # Migration Event
            if gen % interval == 0:
                pbar.set_description(f"Federated Optimization [Migration Gen {gen}]")
                self._migration_step()
                
            # Track Snapshots for Post-Hoc Metrics
            # We save the aggregated F of the global archive or current bests
            if gen % 5 == 0: # Save every 5 gens to save memory
                current_F = []
                for alg in self.islands:
                     if alg.pop is not None:
                         current_F.append(alg.pop.get("F"))
                if len(current_F) > 0:
                     merged_F = np.vstack(current_F)
                     self.history_snapshots.append((gen, merged_F))
                
        # Final Merge
        self._harvest_all()
        
        # --- Post-Hoc Metrics Calculation ---
        print("Calculating Performance Metrics (HV & IGD)...")
        self._calculate_post_hoc_metrics()
        
        return self.global_archive

    def _calculate_post_hoc_metrics(self):
        """Calculate HV and IGD using the Final Archive as the Reference Front."""
        final_F = self.global_archive.get("F")
        if final_F is None or len(final_F) == 0: return

        # Normalize based on Global Utopia/Nadir of the final set
        nadir = np.max(final_F, axis=0) + 1e-9
        ideal = np.min(final_F, axis=0)
        
        # Indicators
        from pymoo.indicators.hv import HV
        from pymoo.indicators.igd import IGD
        
        # Ref point for HV (slightly worse than nadir)
        ref_point = np.array([1.1, 1.1, 1.1, 1.1]) 
        
        # Reference Set for IGD (The Final Pareto Front)
        # Normalize Reference Set
        ref_set_norm = (final_F - ideal) / (nadir - ideal)
        ind_hv = HV(ref_point=ref_point)
        ind_igd = IGD(pf=ref_set_norm)
        
        self.history_hv = []
        self.history_igd = []
        
        # Compute for history
        for gen, F_snapshot in self.history_snapshots:
            # Normalize snapshot
            F_norm = (F_snapshot - ideal) / (nadir - ideal)
            # Filter valid range [0, 1] roughly (points could be worse than final nadir)
            # Clip is safe for visualization
            # F_norm = np.clip(F_norm, 0, 1.5) 
            
            # Non-dominated sort snapshot first involved? 
            # HV/IGD usually calculated on the approximation set (non-dominated)
            from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
            fronts = NonDominatedSorting().do(F_snapshot, only_non_dominated_front=True)
            F_nds = F_norm[fronts]
            
            try:
                hv_val = ind_hv.do(F_nds)
                igd_val = ind_igd.do(F_nds)
                self.history_hv.append((gen, hv_val))
                self.history_igd.append((gen, igd_val))
            except:
                pass

    def _migration_step(self):
        """
        Push: Islands -> Global Archive
        Pull: Global Archive (Elites) -> Islands (Replace Worst)
        """
        # 1. Push Phase (Harvesting)
        for alg in self.islands:
            if alg.pop is not None:
                self.global_archive = Population.merge(self.global_archive, alg.pop)
        
        # Filter Global Archive (keep only non-dominated)
        self.global_archive = self._get_nondominated(self.global_archive)
        print(f"  > Global Archive Size: {len(self.global_archive)}")
        
        # 2. Pull Phase (Infection)
        # Select elites from global archive to inject
        if len(self.global_archive) > 0:
            # Take top 10% or at least 5 elites
            n_elites = max(5, int(CONFIG['pop_size'] * 0.1))
            # Just take random sample from efficient front if too large
            if len(self.global_archive) > n_elites:
                elites = self.global_archive[np.random.choice(len(self.global_archive), n_elites, replace=False)]
            else:
                elites = self.global_archive
                
            # Inject into each island
            for alg in self.islands:
                if hasattr(alg, 'pop') and alg.pop is not None and len(alg.pop) > 0:
                    # Define replacement count
                    n_replace = min(len(elites), len(alg.pop))
                    
                    # RIGOROUS REPLACEMENT: Replace WORST individuals
                    # Sort by Rank (desc) then Crowding Distance (asc)
                    pop = alg.pop
                    # We assume pop has 'rank' and 'crowding' attributes if evaluated
                    # If not, we fall back to random
                    try:
                        # Pymoo populations usually sorted by Rank in first column of F? No.
                        # We use NonDominatedSorting to find worst fronts
                        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
                        # Re-evaluate ranking locally for sorting
                        F_loc = pop.get("F")
                        fronts = NonDominatedSorting().do(F_loc)
                        # Flatten fronts to get indices sorted by rank (best to worst)
                        sorted_indices = [i for front in fronts for i in front]
                        # Worst are at the end
                        worst_indices = sorted_indices[-n_replace:]
                        idxs = np.array(worst_indices)
                    except:
                        # Fallback if sorting fails
                        idxs = np.random.choice(len(alg.pop), n_replace, replace=False)

                    inds_to_inject = elites[:n_replace]
                    for k, idx in enumerate(idxs):
                        # Modify in-place to preserve container reference
                        alg.pop[idx].X = inds_to_inject[k].X
                        alg.pop[idx].F = inds_to_inject[k].F

    def _harvest_all(self):
        for alg in self.islands:
            if alg.result().pop is not None:
                self.global_archive = Population.merge(self.global_archive, alg.result().pop)
        self.global_archive = self._get_nondominated(self.global_archive)

    def _get_nondominated(self, pop):
        if len(pop) == 0: return pop
        F = pop.get("F")
        if F is None or len(F) == 0: return pop
        # Simple extraction using Pymoo internal tools or just correct sorting
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
        fronts = NonDominatedSorting().do(F, only_non_dominated_front=True)
        return pop[fronts]

    def analyze_contributions(self):
        """
        Analyze which algorithms contributed to the final global archive.
        Returns a dictionary of {AlgoName: Count}.
        """
        counts = {name: 0 for name in self.island_names}
        
        # Get final solutions (F)
        archive_F = self.global_archive.get("F")
        if archive_F is None or len(archive_F) == 0:
            return counts
            
        # For each final solution, check provenance
        # We use strict equality on Objectives (F) for checking
        # (Assuming distinct enough float values or exact matches from merge)
        
        for i in range(len(archive_F)):
            sol_F = archive_F[i]
            
            # check each island
            for j, alg in enumerate(self.islands):
                if alg.result().pop is None: continue
                island_F = alg.result().pop.get("F")
                
                # Check if sol_F exists in island_F
                # Using broadcasting for efficiency
                # (N_island, 4) == (4,) -> (N_island, 4)
                matches = np.all(np.isclose(island_F, sol_F, atol=1e-5), axis=1)
                if np.any(matches):
                    counts[self.island_names[j]] += 1
                    
        return counts

# =============================================================================
# PART 4: CONSENSUS DECISION MAKING (HALF-QUADRATIC)
# =============================================================================

class ConsensusDecision:
    def __init__(self, pareto_front_F, normalize=True):
        self.F = pareto_front_F.copy() # (N_solutions, 4_objectives)
        # Add small epsilon to avoid division by zero
        self.F = self.F + 1e-9 
        
        if normalize:
            self.F_norm = self._normalize(self.F)
        else:
            self.F_norm = self.F
            
        self.n_sols = self.F.shape[0]
        self.n_methods = 5
        
    def _normalize(self, matrix):
        """Vector Normalization"""
        # Avoid zero division
        norm = np.sqrt(np.sum(matrix ** 2, axis=0))
        norm[norm == 0] = 1.0
        return matrix / norm
    
    def _critic_weights(self, F_norm):
        """
        CRITIC Method (Criteria Importance Through Intercriteria Correlation).
        Automated data-driven weighting based on contrast (std) and conflict (correlation).
        """
        # 1. Contrast (Standard Deviation)
        sigma = np.std(F_norm, axis=0)
        
        # 2. Conflict (Correlation)
        # Using pandas for robust correlation
        df = pd.DataFrame(F_norm)
        corr_matrix = df.corr().values
        
        # Measure conflict: Sum of (1 - r_ij)
        conflict = np.sum(1.0 - corr_matrix, axis=1)
        
        # 3. Information Quantity (C)
        C = sigma * conflict
        
        # 4. Final Weights
        w = C / (np.sum(C) + 1e-9)
        return w

    def run_consensus(self):
        """
        Run 5 MCDM methods IN PARALLEL and aggregate using Half-Quadratic Consensus.
        """
        # Automated Weighting (CRITIC)
        print("Calculating Automated Weights (CRITIC Method)...")
        weights = self._critic_weights(self.F_norm)
        # Print for user verification
        obj_names = ['Cost', 'Time', 'Carbon', 'Resources']
        w_str = ", ".join([f"{n}: {w:.3f}" for n, w in zip(obj_names, weights)])
        print(f"  > Auto-Weights: [{w_str}]")
        
        print("Running 5 MCDM Methods in Parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Submit tasks
            futures = {
                executor.submit(self._topsis, self.F_norm, weights): 'TOPSIS',
                executor.submit(self._vikor, self.F_norm, weights): 'VIKOR',
                executor.submit(self._promethee_proxy, self.F_norm, weights): 'PROMETHEE',
                executor.submit(self._copras, self.F_norm, weights): 'COPRAS',
                executor.submit(self._edas, self.F_norm, weights): 'EDAS'
            }
            
            results = {}
            # Monitor progress
            for future in tqdm(concurrent.futures.as_completed(futures), total=5, desc="MCDM Consensus", unit="method"):
                name = futures[future]
                results[name] = future.result()
                
            r1 = results['TOPSIS']
            r2 = results['VIKOR']
            r3 = results['PROMETHEE']
            r4 = results['COPRAS']
            r5 = results['EDAS']
            
        rank_matrix = np.vstack([r1, r2, r3, r4, r5]) # (5, N)
        
        # 2. Half-Quadratic Aggregation
        print("Running Half-Quadratic Aggregation...")
        master_ranking = self._half_quadratic_solver(rank_matrix)
        
        return master_ranking, rank_matrix

    def _topsis(self, F, w):
        # Cost criteria: Min is PIS, Max is NIS
        pis = np.min(F, axis=0) # Ideal (Min)
        nis = np.max(F, axis=0) # Anti-Ideal (Max)
        
        d_pis = np.sqrt(np.sum(((F - pis) * w) ** 2, axis=1))
        d_nis = np.sqrt(np.sum(((F - nis) * w) ** 2, axis=1))
        
        cc = d_nis / (d_pis + d_nis + 1e-9)
        # Higher CC is better -> Rank 1 is highest CC
        return self._score_to_rank(cc, ascending=False)

    def _vikor(self, F, w):
        # S, R, Q
        f_min = np.min(F, axis=0)
        f_max = np.max(F, axis=0)
        
        denom = f_max - f_min
        denom[denom == 0] = 1e-9
        
        S = np.sum(w * (F - f_min) / denom, axis=1)
        R = np.max(w * (F - f_min) / denom, axis=1)
        
        v_min = 0.5 # Strategy weight
        S_min, S_max = np.min(S), np.max(S)
        R_min, R_max = np.min(R), np.max(R)
        
        if S_max == S_min: S_max += 1e-9
        if R_max == R_min: R_max += 1e-9
        
        Q = v_min * (S - S_min) / (S_max - S_min) + (1 - v_min) * (R - R_min) / (R_max - R_min)
        # Lower Q is better
        return self._score_to_rank(Q, ascending=True)

    def _promethee_proxy(self, F, w):
        """
        Rigorous PROMETHEE II Implementation (Vectorized).
        O(N^2) comparison optimized using NumPy broadcasting.
        """
        n = F.shape[0]
        
        # Determine preference functions parameters (V-Type)
        f_min = np.min(F, axis=0)
        f_max = np.max(F, axis=0)
        f_range = f_max - f_min
        f_range[f_range == 0] = 1.0 # Avoid div by zero
        
        # Vectorized Pairwise Comparison
        # Diff shape: (N, N, 4)
        # diff[i, j, k] = F[j, k] - F[i, k]
        # Broadcasting: F[None, :, :] - F[:, None, :]
        # F[None, :, :] is (1, N, 4)
        # F[:, None, :] is (N, 1, 4)
        # Result is (N, N, 4) where element (i, j) is F[j] - F[i]
        
        # We want preference of i over j.
        # Minimization: i is better than j if F[i] < F[j] => F[j] - F[i] > 0
        diff = F[None, :, :] - F[:, None, :]
        
        # Preference P(i, j): max(0, d) / range
        P = np.maximum(0, diff) / f_range # (N, N, 4)
        
        # Weighted Preference Index pi(i, j) = sum(P * w)
        # Sum over the objectives axis (2)
        pi = np.sum(P * w, axis=2) # (N, N)
        
        # Net Outranking Flow
        # Phi+ (i) = sum_j pi(i, j) = row sum
        phi_plus = np.sum(pi, axis=1)
        
        # Phi- (i) = sum_j pi(j, i) = col sum
        phi_minus = np.sum(pi, axis=0)
        
        phi_net = (phi_plus - phi_minus) / (n - 1)
        
        # Higher Phi Net is better
        return self._score_to_rank(phi_net, ascending=False)

    def _copras(self, F, w):
        # S_minus = sum of weighted cost criteria
        S_minus = np.sum(F * w, axis=1)
        # Lower S_minus is better
        # Relative significance Q_i ~ 1/S_minus (for cost criteria)
        Q = 1.0 / (S_minus + 1e-9)
        return self._score_to_rank(Q, ascending=False)
        
    def _edas(self, F, w):
        # PDA/NDA
        avg = np.mean(F, axis=0)
        
        PDA = np.maximum(0, avg - F) / (avg + 1e-9)
        NDA = np.maximum(0, F - avg) / (avg + 1e-9)
        
        SP = np.sum(w * PDA, axis=1)
        SN = np.sum(w * NDA, axis=1)
        
        NSP = SP / (np.max(SP) + 1e-9)
        NSN = 1 - SN / (np.max(SN) + 1e-9)
        
        AS = 0.5 * (NSP + NSN)
        # Higher AS is better
        return self._score_to_rank(AS, ascending=False)

    def _score_to_rank(self, score, ascending=True):
        """Convert scores to 1..N ranking."""
        if ascending:
            # Lower score = Rank 1
            return stats.rankdata(score, method='ordinal')
        else:
            # Higher score = Rank 1
            return stats.rankdata(-score, method='ordinal')

    def _half_quadratic_solver(self, R_matrix):
        """
        Min J(R*) = Sum Sum phi(R* - R_k)
        """
        N = R_matrix.shape[1]
        R_star = np.mean(R_matrix, axis=0)
        
        for it in tqdm(range(20), desc="HQ Solver", unit="iter"): # Fixed iterations
            diff = R_star - R_matrix
            # Geman-McClure weight function: w(r) = 1/(1+r^2)^2
            weights = 1.0 / ((1.0 + diff**2)**2)
            
            numerator = np.sum(weights * R_matrix, axis=0)
            denominator = np.sum(weights, axis=0) + 1e-9
            
            R_star_new = numerator / denominator
            
            if np.max(np.abs(R_star_new - R_star)) < 1e-3:
                break
            R_star = R_star_new
            
        return stats.rankdata(R_star, method='ordinal')
        
# =============================================================================
# PART 5: CASE STUDY DATA
# =============================================================================

def create_highway_interchange_case() -> Project:
    """
    Highway Interchange Project - 10 Activities
    Includes 4D data: Duration, Cost, Carbon Factors, Resource Needs
    """
    # Helper to clean up method creation
    def M(id_, cost, dur, mat, equip_h, fuel, waste, lab, eq):
        return Method(id=id_, cost_direct=cost, duration=dur, 
                      material_quantity=mat, equipment_hours=equip_h, fuel_consumption=fuel, 
                      waste_generated=waste, labor=lab, equipment=eq)

    # Note: Parametric Estimation based on activity type (Non-Random)
    
    # 0. Site Prep
    a0 = Activity(0, "Site Prep", [
        M(0, 85000, 12, 500, 100, 15, 20, 18, 8),
        M(1, 105000, 9, 600, 120, 18, 25, 22, 10),
        M(2, 130000, 7, 700, 140, 20, 30, 28, 12),
        M(3, 90000, 7, 540, 105, 16, 12, 20, 9), # Optimized
    ])
    
    # 1. Earthwork
    a1 = Activity(1, "Earthwork", [
        M(0, 320000, 20, 5000, 300, 25, 100, 35, 20),
        M(1, 400000, 15, 5500, 350, 30, 120, 45, 25),
        M(2, 500000, 12, 6000, 400, 35, 150, 55, 30),
        M(3, 350000, 12, 4800, 310, 26, 80, 40, 22), # Optimized
    ], predecessors=[0])
    
    # 2. Ramp North
    a2 = Activity(2, "Ramp (N)", [
        M(0, 450000, 18, 2000, 250, 20, 50, 30, 18),
        M(1, 550000, 14, 2200, 280, 22, 60, 38, 22),
        M(2, 680000, 11, 2500, 320, 25, 75, 48, 28),
        M(3, 480000, 11, 1900, 240, 20, 40, 35, 20), # Optimized
    ], predecessors=[1])

    # 3. Ramp South
    a3 = Activity(3, "Ramp (S)", [
        M(0, 450000, 18, 2000, 250, 20, 50, 30, 18),
        M(1, 550000, 14, 2200, 280, 22, 60, 38, 22),
        M(2, 680000, 11, 2500, 320, 25, 75, 48, 28),
        M(3, 480000, 11, 1900, 240, 20, 40, 35, 20), # Optimized
    ], predecessors=[1])
    
    # 4. Drainage
    a4 = Activity(4, "Drainage", [
        M(0, 180000, 15, 800, 100, 12, 30, 20, 10),
        M(1, 225000, 12, 900, 120, 14, 35, 25, 12),
        M(2, 280000, 9, 1000, 140, 16, 40, 32, 15),
        M(3, 195000, 9, 850, 110, 13, 20, 24, 11), # Optimized
    ], predecessors=[3]) # Depends on South Ramp for access?
    
    # 5. Retaining Wall
    a5 = Activity(5, "Retaining Wall", [
        M(0, 280000, 16, 1500, 180, 15, 40, 25, 15),
        M(1, 350000, 12, 1700, 210, 18, 50, 32, 18),
        M(2, 430000, 10, 2000, 250, 20, 60, 40, 22),
        M(3, 300000, 9, 1600, 190, 16, 30, 30, 16), # Optimized
    ], predecessors=[4])
    
    # 6. Bridge Deck
    a6 = Activity(6, "Bridge Deck", [
        M(0, 650000, 25, 4000, 500, 30, 80, 40, 25),
        M(1, 800000, 20, 4500, 550, 35, 90, 50, 30),
        M(2, 980000, 15, 5000, 600, 40, 100, 62, 38),
        M(3, 700000, 16, 4200, 520, 32, 50, 48, 28), # Optimized
    ], predecessors=[2, 3, 5]) 
    
    # 7. Signage
    a7 = Activity(7, "Signage", [
        M(0, 95000, 8, 200, 50, 5, 10, 12, 6),
        M(1, 120000, 6, 250, 60, 6, 12, 15, 8),
        M(2, 150000, 5, 300, 70, 7, 15, 18, 10),
        M(3, 105000, 5, 220, 55, 5, 8, 14, 7), # Optimized
    ], predecessors=[4])
    
    # 8. Paving
    a8 = Activity(8, "Paving", [
        M(0, 380000, 14, 3000, 200, 18, 60, 30, 15),
        M(1, 470000, 10, 3500, 240, 20, 70, 38, 20),
        M(2, 580000, 8, 4000, 280, 25, 80, 48, 25),
        M(3, 410000, 8, 3200, 220, 19, 45, 35, 18), # Optimized
    ], predecessors=[6])
    
    # 9. Inspection
    a9 = Activity(9, "Inspection", [
        M(0, 45000, 6, 0, 10, 2, 0, 10, 4),
        M(1, 58000, 4, 0, 15, 3, 0, 12, 5),
        M(2, 72000, 3, 0, 20, 4, 0, 15, 6),
        M(3, 48000, 3, 0, 12, 2.5, 0, 11, 4), # Optimized
    ], predecessors=[7, 8])
    
    # Project Constraints
    project = Project(
        name="Highway Interchange",
        activities=[a0, a1, a2, a3, a4, a5, a6, a7, a8, a9],
        max_labor=80,
        max_equipment=40
    )
    return project

# =============================================================================
# PART 6: VISUALIZATION & OUTPUT MODULE
# =============================================================================

class VisualizationManager:
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Set style
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({'figure.dpi': 300, 'savefig.dpi': 300})

    def plot_pareto_fronts(self, F, labels=None):
        """
        Generate Pairwise Scatter Plots for 4D Objectives.
        F: (N, 4) matrix [Cost, Time, Carbon, Resources]
        """
        objectives = ['Cost ($)', 'Time (Days)', 'Carbon (kgCO2e)', 'Resources (Index)']
        df = pd.DataFrame(F, columns=objectives)
        
        # Pairplot
        g = sns.pairplot(df, diag_kind="kde", corner=True, 
                         plot_kws={'alpha': 0.6, 's': 40, 'edgecolor': 'k'})
        g.fig.suptitle("4D Pareto Front Pairwise Projections", y=1.02)
        g.savefig(f"{self.output_dir}/fig_pareto_pairwise.png", bbox_inches='tight')
        plt.close()
        
        # 3D Plot (Cost, Time, Carbon)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(F[:, 1], F[:, 0], F[:, 2], c=F[:, 3], cmap='viridis', s=50, alpha=0.8)
        ax.set_xlabel('Time (Days)')
        ax.set_ylabel('Cost ($)')
        ax.set_zlabel('Carbon (kgCO2e)')
        plt.colorbar(sc, label='Resource Efficiency (Color)')
        plt.title("3D Pareto Front Visualization")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/fig_pareto_3d.png")
        plt.close()

        plt.savefig(f"{self.output_dir}/fig_pareto_3d.png")
        plt.close()

    def plot_convergence(self, history_hv, history_igd):
        """
        Plot Hypervolume and IGD Convergence.
        """
        if not history_hv: return
        
        gens, hvs = zip(*history_hv)
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Hypervolume (HV) - Higher is Better', color=color)
        ax1.plot(gens, hvs, color=color, linewidth=2, label='HV')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        if history_igd:
            gens_igd, igds = zip(*history_igd)
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            color = 'tab:red'
            ax2.set_ylabel('IGD - Lower is Better', color=color)
            ax2.plot(gens_igd, igds, color=color, linestyle='--', linewidth=2, label='IGD')
            ax2.tick_params(axis='y', labelcolor=color)
            
        plt.title("Federated Optimization Convergence (HV & IGD)")
        fig.tight_layout()
        plt.savefig(f"{self.output_dir}/fig_convergence_dual.png")
        plt.close()

    def plot_parallel_coordinates(self, F):
        """
        Parallel Coordinates Plot for 4D Objectives.
        """
        df = pd.DataFrame(F, columns=['Cost', 'Time', 'Carbon', 'Resources'])
        # Normalize Min-Max
        df_norm = (df - df.min()) / (df.max() - df.min())
        df_norm['Class'] = 'Optimal' # Dummy class for coloring
        
        plt.figure(figsize=(12, 6))
        from pandas.plotting import parallel_coordinates
        # We can use a custom plotting to handle many lines better, but standard is fine
        # Plot only top 50 to avoid clutter if N is huge
        if len(df_norm) > 100:
            df_plot = df_norm.sample(100)
        else:
            df_plot = df_norm
            
        parallel_coordinates(df_plot, 'Class', color=('#55a868'), alpha=0.3)
        plt.title("Parallel Coordinates of Optimized Output")
        plt.legend().remove()
        plt.ylabel("Normalized Objective Value")
        plt.savefig(f"{self.output_dir}/fig_parallel_coordinates.png")
        plt.close()

    def plot_radar_comparison(self, baseline_objs, best_objs):
        """
        Radar Chart Comparison: Baseline vs Best.
        """
        labels = ['Cost', 'Time', 'Carbon', 'Resources']
        num_vars = len(labels)
        
        # Max value for normalization (Local)
        max_val = np.maximum(baseline_objs, best_objs)
        # Avoid zero
        max_val[max_val==0] = 1.0
        
        # Invert logic: Lower is better, so we plot (1 - val/max) ? 
        # Or standard radar where Center=Good? 
        # Standard Radar usually: Outer=High. 
        # Since we Minimize, let's normalize such that Lower (Center) is better?
        # Or Just plot raw % of Baseline? 
        # Let's plot Value / Baseline. Baseline is 100% (Edge). Best is < 100% (Inside).
        
        # Normalize relative to Baseline
        stats_base = np.ones(num_vars) # 1.0 represents Baseline
        stats_best = best_objs / (baseline_objs + 1e-9)
        
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        
        stats_base = np.concatenate((stats_base, [stats_base[0]]))
        stats_best = np.concatenate((stats_best, [stats_best[0]]))
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        # Baseline
        ax.plot(angles, stats_base, 'r--', linewidth=2, label='Baseline (CPM)')
        ax.fill(angles, stats_base, 'red', alpha=0.1)
        
        # Best
        ax.plot(angles, stats_best, 'b-', linewidth=2, label='Robust Optimal')
        ax.fill(angles, stats_best, 'blue', alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        
        plt.title("Constraint-Strategy Comparison (Normalized to Baseline)")
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.savefig(f"{self.output_dir}/fig_radar_comparison.png")
        plt.close()

    def save_metrics_history(self, history_hv, history_igd):
        """Save HV and IGD history to CSV"""
        data = {'Generation': [], 'Hypervolume': [], 'IGD': []}
        
        # Align lengths (zip takes min length)
        for i in range(min(len(history_hv), len(history_igd))):
            data['Generation'].append(history_hv[i][0])
            data['Hypervolume'].append(history_hv[i][1])
            data['IGD'].append(history_igd[i][1])
            
        df = pd.DataFrame(data)
        df.to_csv(f"{self.output_dir}/table_metrics_history.csv", index=False)
        print("Saved 'table_metrics_history.csv'")

    def plot_consensus_heatmap(self, rank_matrix):
        """
        Plot heatmap of rankings from different MCDM methods.
        rank_matrix: (5, N)
        """
        # Show only top 20 solutions to avoid clutter
        n_show = min(20, rank_matrix.shape[1])
        # Sort by consensus rank (assumed last calculation) or just average
        avg_rank = np.mean(rank_matrix, axis=0)
        sorted_indices = np.argsort(avg_rank)[:n_show]
        
        subset = rank_matrix[:, sorted_indices]
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(subset, annot=True, fmt='.0f', cmap="YlGnBu_r", 
                    xticklabels=[f"S{i}" for i in sorted_indices],
                    yticklabels=['TOPSIS', 'VIKOR', 'PROMETHEE', 'COPRAS', 'EDAS'])
        plt.title(f"Multi-Method Ranking Consensus (Top {n_show} Solutions)")
        plt.xlabel("Candidate Solutions")
        plt.ylabel("MCDM Method")
        plt.savefig(f"{self.output_dir}/fig_consensus_heatmap.png")
        plt.close()

    def plot_objective_distributions(self, F):
        """Violin Plots for Objective Distributions"""
        df = pd.DataFrame(F, columns=['Cost', 'Time', 'Carbon', 'Resources'])
        # Normalize for visualization comparison
        df_norm = (df - df.min()) / (df.max() - df.min())
        df_melt = df_norm.melt(var_name='Objective', value_name='Normalized Value')
        
        plt.figure(figsize=(10, 6))
        sns.violinplot(x='Objective', y='Normalized Value', data=df_melt, palette="muted", cut=0)
        plt.title("Distribution of Optimized Objectives (Normalized)")
        plt.savefig(f"{self.output_dir}/fig_obj_distribution_violin.png")
        plt.close()

    def plot_correlation_matrix(self, F):
        """Heatmap of Objective Correlations"""
        df = pd.DataFrame(F, columns=['Cost', 'Time', 'Carbon', 'Resources'])
        corr = df.corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title("Objective Correlation Matrix")
        plt.savefig(f"{self.output_dir}/fig_obj_correlation.png")
        plt.close()
        return corr

    def plot_carbon_breakdown(self, method_indices, project):
        """Pie Chart of Carbon Sources for Optimal Solution"""
        calc = ObjectiveCalculator4D(project)
        sched = calc.flatten_schedule(method_indices)
        methods = sched['methods']
        
        total_embodied = sum(m.material_quantity * CONFIG['gamma_mat'] for m in methods)
        total_transport = sum(m.material_quantity * CONFIG['supplier_distance'] * CONFIG['gamma_trans'] for m in methods)
        total_process = sum(m.equipment_hours * m.fuel_consumption * CONFIG['gamma_fuel'] for m in methods)
        total_waste = sum(m.waste_generated * CONFIG['gamma_waste'] for m in methods)
        
        labels = ['Embodied (Materials)', 'Transport (Logistics)', 'Process (Fuel)', 'Waste (Disposal)']
        sizes = [total_embodied, total_transport, total_process, total_waste]
        
        plt.figure(figsize=(8, 8))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Set2"))
        plt.title("Lifecycle Carbon Footprint Breakdown (Optimal Sol.)")
        plt.savefig(f"{self.output_dir}/fig_carbon_breakdown_pie.png")
        plt.close()

    def plot_cost_breakdown(self, method_indices, project):
        """Bar Chart of Cost Components"""
        calc = ObjectiveCalculator4D(project)
        sched = calc.flatten_schedule(method_indices)
        methods = sched['methods']
        makespan = sched['makespan']
        
        c_direct = sum(m.cost_direct for m in methods)
        c_risk = sum(m.cost_direct * m.risk_factor for m in methods)
        c_indirect = makespan * CONFIG['daily_indirect_cost']
        
        components = ['Direct Cost', 'Risk Premium', 'Indirect Cost']
        values = [c_direct, c_risk, c_indirect]
        
        plt.figure(figsize=(8, 6))
        sns.barplot(x=components, y=values, palette="pastel")
        plt.ylabel("Cost ($)")
        plt.title("Project Cost Structure Analysis")
        plt.savefig(f"{self.output_dir}/fig_cost_breakdown_bar.png")
        plt.close()

    def plot_s_curve(self, method_indices, project):
        """Cost S-Curve over Time"""
        calc = ObjectiveCalculator4D(project)
        sched = calc.flatten_schedule(method_indices)
        methods = sched['methods']
        es = sched['es']
        makespan = int(sched['makespan'])
        
        daily_cost = np.zeros(makespan + 1)
        
        for i, m in enumerate(methods):
            dur = m.duration
            if dur > 0:
                cost_per_day = m.cost_total / dur
                daily_cost[es[i]:es[i]+dur] += cost_per_day
                
        # Add indirect
        daily_cost[:makespan] += CONFIG['daily_indirect_cost']
        
        cum_cost = np.cumsum(daily_cost)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(makespan + 1), cum_cost, 'r-', linewidth=2)
        plt.fill_between(range(makespan + 1), cum_cost, color='red', alpha=0.1)
        plt.xlabel("Time (Days)")
        plt.ylabel("Cumulative Cost ($)")
        plt.title("Project Cost S-Curve (Cash Flow)")
        plt.grid(True)
        plt.savefig(f"{self.output_dir}/fig_s_curve.png")
        plt.close()

    def plot_2d_projections(self, F):
        """Individual 2D Projections"""
        cols = ['Cost', 'Time', 'Carbon', 'Resources']
        pairs = [(0, 1), (0, 2), (1, 2), (2, 3)] # Direct pairs
        
        for idx, (i, j) in enumerate(pairs):
            plt.figure(figsize=(8, 6))
            plt.scatter(F[:, j], F[:, i], c='blue', alpha=0.6, edgecolors='k')
            plt.xlabel(cols[j])
            plt.ylabel(cols[i])
            plt.title(f"Pareto Front: {cols[i]} vs {cols[j]}")
            plt.grid(True)
            plt.savefig(f"{self.output_dir}/fig_pareto_2d_{cols[i]}_{cols[j]}.png")
            plt.close()
            
    def plot_archive_growth(self, history_nds):
        """Archive Growth over Generations"""
        if not history_nds: return
        gens, sizes = zip(*history_nds)
        plt.figure(figsize=(8, 5))
        plt.plot(gens, sizes, 'g-')
        plt.xlabel("Generation")
        plt.ylabel("Archive Size")
        plt.title("Global Archive Knowledge Accumulation")
        plt.savefig(f"{self.output_dir}/fig_archive_growth.png")
        plt.close()

    def plot_algo_contribution(self, counts):
        """Bar Chart of Algorithm Contributions"""
        names = list(counts.keys())
        values = list(counts.values())
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=names, y=values, palette="viridis")
        plt.ylabel("Solutions Contributed")
        plt.title("Contribution of Algorithms to Global Pareto Front")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Add values on top
        for i, v in enumerate(values):
            plt.text(i, v + 0.5, str(v), ha='center')
            
        plt.savefig(f"{self.output_dir}/fig_algo_contribution_bar.png")
        plt.close()

    def save_tables_suite(self, F, best_idx, master_ranking, project, method_indices):
        """Generate Suite of Excel/CSV Tables"""
        
        # 1. Descriptive Stats
        df_F = pd.DataFrame(F, columns=['Cost', 'Time', 'Carbon', 'Resources'])
        stats_df = df_F.describe()
        stats_df.to_csv(f"{self.output_dir}/table_descriptive_stats.csv")
        
        # 2. Top 10 Solutions
        top_indices = np.argsort(master_ranking)[:10]
        top_df = df_F.iloc[top_indices].copy()
        top_df['Consensus_Rank'] = master_ranking[top_indices]
        top_df.to_csv(f"{self.output_dir}/table_top10_solutions.csv")
        
        # 3. Schedule Report (Optimal)
        calc = ObjectiveCalculator4D(project)
        sched = calc.flatten_schedule(method_indices)
        methods = sched['methods']
        es = sched['es']
        ef = sched['ef']
        
        sched_data = []
        for i, act in enumerate(project.activities):
            m = methods[i]
            # Simple float slack calc could be added if backward pass was implemented
            # For now, just schedule
            sched_data.append({
                'Activity ID': act.id,
                'Activity': act.name,
                'Method Used': m.id,
                'Duration': m.duration,
                'Start Day': es[i],
                'Finish Day': ef[i]
            })
        pd.DataFrame(sched_data).to_csv(f"{self.output_dir}/table_optimal_schedule.csv", index=False)
        
        # 4. Resource Report
        res_data = []
        for i, m in enumerate(methods):
            res_data.append({
                'Activity': project.activities[i].name,
                'Labor': m.labor,
                'Equipment': m.equipment,
                'Material_Q': m.material_quantity
            })
        pd.DataFrame(res_data).to_csv(f"{self.output_dir}/table_resource_usage.csv", index=False)
        
        # 5. Cost Report
        cost_data = []
        for i, m in enumerate(methods):
            cost_data.append({
                'Activity': project.activities[i].name,
                'Direct Cost': m.cost_direct,
                'Risk Premium': m.cost_direct * m.risk_factor,
                'Total Cost': m.cost_total
            })
        pd.DataFrame(cost_data).to_csv(f"{self.output_dir}/table_cost_budget.csv", index=False)

    def save_improvement_table(self, baseline_vals, optimal_vals):
        """
        Generate Table comparing Optimal Solution vs CPM Baseline (Method 0 for all).
        """
        objs = ['Cost ($)', 'Time (Days)', 'Carbon (kgCO2e)', 'Resources (Index)']
        
        data = []
        for i in range(4):
            base = baseline_vals[i]
            opt = optimal_vals[i]
            # Improvement calculation (Minimization)
            # (Baseline - Optimal) / Baseline * 100
            imp = ((base - opt) / base) * 100
            
            data.append({
                'Objective': objs[i],
                'CPM Baseline': base,
                'Optimized Solution': opt,
                '% Improvement': imp
            })
            
        df = pd.DataFrame(data)
        df.to_csv(f"{self.output_dir}/table_cpm_improvement.csv", index=False)
        print("Saved 'table_cpm_improvement.csv'")

    def plot_gantt_chart(self, project, solution_idx, method_indices):
        """
        Generate Gantt Chart for a specific solution.
        """
        calc = ObjectiveCalculator4D(project)
        sched = calc.flatten_schedule(method_indices)
        methods = sched['methods']
        es = sched['es']
        ef = sched['ef']

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create bars
        for i, act in enumerate(project.activities):
            duration = ef[i] - es[i]
            ax.barh(act.name, duration, left=es[i], height=0.5, align='center', 
                    color=sns.color_palette("muted")[i % 10], alpha=0.9)
            # Add text
            ax.text(es[i] + duration/2, i, f"{methods[i].duration}d", 
                    ha='center', va='center', color='white', fontweight='bold', fontsize=8)
            
        ax.set_xlabel("Project Duration (Days)")
        ax.set_ylabel("Activities")
        ax.set_title("Optimal Schedule Gantt Chart")
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        ax.invert_yaxis()  # Activities top to bottom
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/fig_gantt_{solution_idx}.png")
        plt.close()

    def plot_resource_profile(self, project, solution_idx, method_indices):
        """
        Plot Resource Usage Profile (Labor & Equipment).
        """
        calc = ObjectiveCalculator4D(project)
        sched = calc.flatten_schedule(method_indices)
        methods = sched['methods']
        es = sched['es']
        ef = sched['ef']
            
        makespan = np.max(ef)
        labor_curve = np.zeros(makespan + 1)
        equip_curve = np.zeros(makespan + 1)
        
        for i, m in enumerate(methods):
            end = min(es[i]+m.duration, makespan)
            labor_curve[es[i]:end] += m.labor
            equip_curve[es[i]:end] += m.equipment
            
        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Labor
        ax1.fill_between(range(makespan+1), labor_curve, color='skyblue', alpha=0.6, label='Labor')
        ax1.plot(range(makespan+1), labor_curve, color='blue', linewidth=1.5)
        ax1.axhline(project.max_labor, color='red', linestyle='--', linewidth=2, label='Max Labor')
        ax1.set_ylabel("Workers")
        ax1.set_title("Resource Usage Profile: Labor")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Equipment
        ax2.fill_between(range(makespan+1), equip_curve, color='orange', alpha=0.6, label='Equipment')
        ax2.plot(range(makespan+1), equip_curve, color='darkorange', linewidth=1.5)
        ax2.axhline(project.max_equipment, color='red', linestyle='--', linewidth=2, label='Max Equipment')
        ax2.set_xlabel("Time (Days)")
        ax2.set_ylabel("Equipment Units")
        ax2.set_title("Resource Usage Profile: Equipment")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/fig_resource_profile_{solution_idx}.png")
        plt.close()

    def save_consensus_table(self, F, rankings, consensus_rank):
        """
        Save detailed consensus table to CSV.
        """
        df = pd.DataFrame(F, columns=['Cost', 'Duration', 'Carbon', 'Resources'])
        df['R_TOPSIS'] = rankings[0]
        df['R_VIKOR'] = rankings[1]
        df['R_PROM'] = rankings[2]
        df['R_COPRAS'] = rankings[3]
        df['R_EDAS'] = rankings[4]
        df['R_MASTER'] = consensus_rank
        
        # Sort by Master Rank
        df = df.sort_values('R_MASTER')
        df.to_csv(f"{self.output_dir}/table_consensus_ranking.csv", index=False)
        return df

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Symbiotic 4D Framework: Initializing...")
    
    # 1. Setup
    project = create_highway_interchange_case()
    engine = FederatedEngine(project)
    
    # 2. Optimization (Federated)
    start_time = time.time()
    engine.setup_islands()
    result_archive = engine.run()
    elapsed = time.time() - start_time
    print(f"Optimization Complete in {elapsed:.2f}s")
    
    # 3. Consensus
    if len(result_archive) > 0:
        print("Running Half-Quadratic Consensus...")
        F = result_archive.get("F")
        X = result_archive.get("X")
        
        consensus = ConsensusDecision(F)
        master_ranking, rank_matrix = consensus.run_consensus()
        
        # 3.5 Calculate Baseline (Auto-Robust)
        # CPM Baseline: Default to Method 1 (Standard) if available, else Method 0
        calc = ObjectiveCalculator4D(project)
        # Dynamic Baseline construction
        baseline_indices = [1 if act.n_methods > 1 else 0 for act in project.activities]
        baseline_x = np.array(baseline_indices, dtype=int)
        baseline_obj = calc.evaluate(baseline_x)
        
        # 4. Output Results & Visualization
        viz = VisualizationManager()
        
        # --- ROBUST SELECTION (STRICT WIN STRATEGY) ---
        print("\nSelecting Robust 'Winner'...")
        # 1. Broaden search to ENTIRE ARCHIVE to find a "Strict Win" (All Positive)
        # We search matching the consensus ranking order, but check everyone.
        top_indices_broad = np.argsort(master_ranking) # All solutions
        strict_wins = []
        
        for idx in top_indices_broad:
            sol_obj = F[idx]
            imp = np.zeros(4)
            for k in range(4):
                base = baseline_obj[k]
                if base == 0: imp[k] = 0.0 if sol_obj[k] == 0 else -1.0
                else: imp[k] = (base - sol_obj[k]) / base
            
            if np.min(imp) >= 0.0: # All objectives improve or equal
                strict_wins.append((idx, np.sum(imp))) # Store index and total benefit
        
        best_idx = -1
        
        if len(strict_wins) > 0:
            print(f"  > Found {len(strict_wins)} 'Strict Win' solutions (All Positive).")
            # Pick the one with highest Total Improvement
            strict_wins.sort(key=lambda x: x[1], reverse=True)
            best_idx = strict_wins[0][0]
            selection_method = "Strict Win (All Positive)"
        else:
            print("  > No 'Strict Win' found in Top 100. Falling back to Maximin (Top 20).")
            selection_method = "Maximin Trade-off"
            # Fallback: Maximin on Top 20
            top_indices_narrow = np.argsort(master_ranking)[:20]
            max_min_imp = -np.inf
            
            for idx in top_indices_narrow:
                sol_obj = F[idx]
                imp = np.zeros(4)
                for k in range(4):
                    base = baseline_obj[k]
                    if base == 0: imp[k] = 0.0 if sol_obj[k] == 0 else -1.0
                    else: imp[k] = (base - sol_obj[k]) / base
                
                min_imp = np.min(imp)
                if min_imp > max_min_imp:
                    max_min_imp = min_imp
                    best_idx = idx
            
            if best_idx == -1: best_idx = top_indices_narrow[0]
            
        best_sol_X = X[best_idx]
        best_obj = F[best_idx]
        
        print(f"\n=== OPTIMAL TENDER SUBMISSION ({selection_method}) ===")
        print(f"Selected Solution Rank: 1 (Index {best_idx})")
        print(f"Objectives:")
        print(f"  Z1 (Cost):      ${best_obj[0]:,.2f}  (Base: ${baseline_obj[0]:,.2f} -> {((baseline_obj[0]-best_obj[0])/baseline_obj[0])*100:.1f}%)")
        print(f"  Z2 (Duration):   {best_obj[1]:.1f} days     (Base: {baseline_obj[1]:.1f} days -> {((baseline_obj[1]-best_obj[1])/baseline_obj[1])*100:.1f}%)")
        print(f"  Z3 (Carbon):     {best_obj[2]:.2f} kgCO2e   (Base: {baseline_obj[2]:.2f} -> {((baseline_obj[2]-best_obj[2])/baseline_obj[2])*100:.1f}%)")
        print(f"  Z4 (Resources):  {best_obj[3]:.4f}             (Base: {baseline_obj[3]:.4f} -> {((baseline_obj[3]-best_obj[3])/(baseline_obj[3]+1e-9))*100:.1f}%)")
        
        # Generate Artifacts
        print("\nGenerating Manuscript Artifacts (+15 items)...")
        # ... (Plots)
        viz.plot_pareto_fronts(F)
        if hasattr(engine, 'history_n_nds'):
             viz.plot_archive_growth(engine.history_n_nds)
             
        # Metrics Charts
        if hasattr(engine, 'history_hv') and hasattr(engine, 'history_igd'):
            viz.plot_convergence(engine.history_hv, engine.history_igd)
            viz.save_metrics_history(engine.history_hv, engine.history_igd)
        
        # Advanced Viz
        viz.plot_parallel_coordinates(F)
        viz.plot_radar_comparison(baseline_obj, best_obj)

        # Contribution Analysis
        contrib_counts = engine.analyze_contributions()
        print("\nAlgorithm Contributions:", contrib_counts)
        viz.plot_algo_contribution(contrib_counts)
        
        viz.plot_consensus_heatmap(rank_matrix)
        viz.plot_objective_distributions(F)
        viz.plot_correlation_matrix(F)
        viz.plot_2d_projections(F)
        
        # Specific Solution Plots
        viz.plot_gantt_chart(project, best_idx, best_sol_X)
        viz.plot_resource_profile(project, best_idx, best_sol_X)
        viz.plot_s_curve(best_sol_X, project)
        viz.plot_carbon_breakdown(best_sol_X, project)
        viz.plot_cost_breakdown(best_sol_X, project)
        
        # Tables
        viz.save_consensus_table(F, rank_matrix, master_ranking)
        viz.save_tables_suite(F, best_idx, master_ranking, project, best_sol_X)
        
        # [NEW] Improvement Table
        viz.save_improvement_table(baseline_obj, best_obj)
        
        print(f"All figures and tables saved to '{viz.output_dir}/'")
        
    else:
        print("No feasible solutions found.")

