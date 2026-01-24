# 5D Multi-Objective Optimization Framework for Bridge Construction Planning with Automated Decision-Making

## NSGA-III Based Optimization with Adaptive MCDM Auto-Selection

---

# 1. Introduction & Research Gap

## 1.1 Background

Bridge construction projects involve complex interdependencies among time, cost, quality, safety, and sustainability objectives. Traditional optimization approaches often consider only 2-3 objectives (time-cost-quality trade-offs), neglecting critical sustainability and safety dimensions that are increasingly mandated by regulations and stakeholder expectations.

## 1.2 Research Gap

Despite advances in many-objective optimization algorithms, current literature lacks:

1. **Integrated 5D framework** specifically designed for bridge construction considering Time-Cost-Quality-Safety-Sustainability simultaneously
2. **Automated MCDM selection** that adapts decision-making method based on Pareto front characteristics
3. **Uncertainty quantification** for bridge-specific construction methods and environmental conditions
4. **Validation framework** combining numerical experiments with real-world case studies

## 1.3 Research Objectives

1. Develop a 5D many-objective optimization model for bridge construction planning
2. Apply NSGA-III algorithm for generating diverse Pareto-optimal solutions
3. Design an automated MCDM selection mechanism that chooses the optimal decision method based on solution space topology
4. Validate the framework using three complex bridge construction case studies

---

# 2. 5D Objective Framework

## 2.1 Decision Variables

For a bridge construction project with $n$ activities, each activity $i$ has $m_i$ execution methods. The decision variable $x_{ij} \in \{0, 1\}$ indicates whether method $j$ is selected for activity $i$:

$$\sum_{j=1}^{m_i} x_{ij} = 1 \quad \forall i \in \{1, 2, ..., n\}$$

## 2.2 Objective Functions

### Z₁: Project Duration (Minimize)

$$Z_1 = \max_{p \in P} \left( \sum_{i \in p} \sum_{j=1}^{m_i} d_{ij} \cdot x_{ij} \right)$$

Where:
- $P$ = set of all paths through the project network
- $d_{ij}$ = duration of activity $i$ using method $j$

### Z₂: Total Project Cost (Minimize)

$$Z_2 = \sum_{i=1}^{n} \sum_{j=1}^{m_i} \left( DC_{ij} + IC_{ij} \right) \cdot x_{ij} + C_{indirect} \cdot Z_1$$

Where:
- $DC_{ij}$ = direct cost (labor + material + equipment)
- $IC_{ij}$ = indirect cost allocation
- $C_{indirect}$ = daily indirect cost rate

**Direct Cost Breakdown:**

$$DC_{ij} = \sum_{r=1}^{R} \left( q_{ijr} \cdot c_r \right) + \sum_{e=1}^{E} \left( u_{ije} \cdot c_e \cdot d_{ij} \right) + M_{ij}$$

### Z₃: Quality Index (Maximize)

$$Z_3 = \sum_{i=1}^{n} w_i^Q \cdot \sum_{j=1}^{m_i} Q_{ij} \cdot x_{ij}$$

**Quality Score Components:**

$$Q_{ij} = \alpha_1 \cdot Q_{ij}^{precision} + \alpha_2 \cdot Q_{ij}^{durability} + \alpha_3 \cdot Q_{ij}^{compliance}$$

With $\alpha_1 + \alpha_2 + \alpha_3 = 1$

### Z₄: Safety Risk Index (Minimize)

$$Z_4 = \sum_{i=1}^{n} \sum_{j=1}^{m_i} \left( \sum_{h=1}^{H} P_{ijh} \cdot S_{ijh} \right) \cdot x_{ij}$$

**Risk Probability Model:**

$$P_{ijh} = P_{base,h} \cdot \prod_{k=1}^{K} RF_{ijk}$$

Where $RF_{ijk}$ = risk factor multipliers (weather, height, equipment type, worker experience)

### Z₅: Sustainability Index (Maximize)

$$Z_5 = \beta_1 \cdot S_{carbon} + \beta_2 \cdot S_{waste} + \beta_3 \cdot S_{energy} + \beta_4 \cdot S_{social}$$

**Carbon Footprint Component:**

$$S_{carbon} = 1 - \frac{\sum_{i=1}^{n} \sum_{j=1}^{m_i} CO2_{ij} \cdot x_{ij}}{CO2_{max}}$$

$$CO2_{ij} = \sum_{m=1}^{M} \left( q_{ijm} \cdot EF_m \right) + \sum_{e=1}^{E} \left( u_{ije} \cdot d_{ij} \cdot EF_e \right)$$

---

# 3. Constraints

## 3.1 Precedence Constraints

$$ES_k \geq EF_i \quad \forall (i,k) \in E$$

## 3.2 Resource Constraints

$$\sum_{i \in A_t} \sum_{j=1}^{m_i} r_{ijl} \cdot x_{ij} \leq R_l^{max} \quad \forall l \in L, \forall t$$

## 3.3 Budget Constraint

$$Z_2 \leq B_{max}$$

## 3.4 Deadline Constraint

$$Z_1 \leq T_{deadline}$$

## 3.5 Minimum Quality Constraint

$$Z_3 \geq Q_{min}$$

## 3.6 Maximum Safety Risk Constraint

$$Z_4 \leq Risk_{max}$$

---

# 4. NSGA-III Optimization

## 4.1 Algorithm Selection Justification

NSGA-III is selected for this framework because:
- **Reference point-based selection** maintains well-distributed solutions across 5 objectives
- **Proven performance** on many-objective problems (M ≥ 4)
- **Available implementations** in pymoo and other libraries (no algorithm development required)

## 4.2 NSGA-III Configuration

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Population Size | 200 | Sufficient diversity for 5 objectives |
| Generations | 500 | Convergence based on preliminary tests |
| Crossover | SBX (η=15) | Standard for real-coded problems |
| Mutation | Polynomial (η=20) | Maintains boundary constraints |
| Reference Points | Das-Dennis (p=6) | H = 462 points for 5 objectives |

## 4.3 Reference Point Generation

For 5 objectives with $p$ divisions:

$$H = \binom{M + p - 1}{p} = \binom{5 + 6 - 1}{6} = 462$$

---

# 5. Automated Optimal MCDM Selection Framework

## 5.1 Framework Overview

The framework automatically selects the **optimal MCDM method** from a pool of 6 methods based on:
1. Pareto front topology characteristics
2. Decision context (weight availability, uncertainty level)
3. Stakeholder preference structure

## 5.2 Pareto Front Characterization Metrics

**Spread Indicator (Δ):**

$$\Delta = \frac{\sum_{i=1}^{|A|-1} |d_i - \bar{d}|}{(|A|-1) \cdot \bar{d}}$$

**Knee Point Ratio (KR):**

$$K_i = \frac{\sum_{m=1}^{M} \left( f_m^{max} - f_m(x_i) \right)}{\sum_{m=1}^{M} \left( f_m(x_i) - f_m^{min} \right)}$$

$$KR = \frac{|\{x_i : K_i > K_{threshold}\}|}{|A|}$$

**Solution Diversity Index (SDI):**

$$SDI = \frac{1}{|A|} \sum_{i=1}^{|A|} \min_{j \neq i} d_{ij}$$

**Objective Correlation Matrix (OCM):**

$$\rho_{jk} = \frac{\sum_{i=1}^{|A|} (f_j(x_i) - \bar{f_j})(f_k(x_i) - \bar{f_k})}{\sqrt{\sum_{i=1}^{|A|} (f_j(x_i) - \bar{f_j})^2 \cdot \sum_{i=1}^{|A|} (f_k(x_i) - \bar{f_k})^2}}$$

## 5.3 MCDM Method Pool

| Method | Abbreviation | Strengths |
|--------|--------------|-----------|
| TOPSIS | Technique for Order Preference | Balanced trade-offs, geometric interpretation |
| VIKOR | Multi-criteria Compromise | Compromise solutions, regret minimization |
| PROMETHEE-II | Preference Ranking | Outranking with preference functions |
| Grey Relational Analysis | GRA | Handles uncertainty and incomplete data |
| ELECTRE-III | Elimination and Choice | Threshold-based concordance/discordance |
| Entropy-WASPAS | Hybrid | Objective weight determination + WASPAS ranking |

## 5.4 Comprehensive MCDM Comparison Framework

### 5.4.1 Baseline Solution Definition

The **Baseline Solution** represents the conventional practice (all activities using Method 0 - standard/normal practice):

$$x_{baseline} = \{x_{i0} = 1, x_{ij} = 0 \; \forall j \neq 0\}$$

**Baseline Objective Values:**
$$F_{baseline} = [Z_1^{base}, Z_2^{base}, Z_3^{base}, Z_4^{base}, Z_5^{base}]$$

### 5.4.2 Apply All MCDM Methods

For the Pareto front $A$ from NSGA-III, apply all 6 MCDM methods:

```python
def apply_all_mcdms(pareto_front, weights):
    """
    Apply all 6 MCDM methods to the Pareto front
    Returns: Selected solution and ranking from each method
    """
    results = {}
    
    # 1. TOPSIS
    results['TOPSIS'] = apply_topsis(pareto_front, weights)
    
    # 2. VIKOR
    results['VIKOR'] = apply_vikor(pareto_front, weights)
    
    # 3. PROMETHEE-II
    results['PROMETHEE-II'] = apply_promethee(pareto_front, weights)
    
    # 4. Grey Relational Analysis
    results['GRA'] = apply_gra(pareto_front, weights)
    
    # 5. ELECTRE-III
    results['ELECTRE-III'] = apply_electre(pareto_front, weights)
    
    # 6. Entropy-WASPAS
    results['Entropy-WASPAS'] = apply_entropy_waspas(pareto_front)
    
    return results
```

### 5.4.3 Performance Comparison Against Baseline

For each MCDM method $m$, calculate improvement over baseline:

**Improvement Index (II) for each objective:**

$$II_{mj} = \frac{Z_j^{base} - Z_j^{MCDM_m}}{Z_j^{base}} \times 100\%$$

(For maximization objectives, reverse the formula)

**Overall Improvement Score (OIS):**

$$OIS_m = \sum_{j=1}^{5} w_j \cdot II_{mj}$$

**Dominance Count (DC):**

$$DC_m = |\{j : II_{mj} > 0\}|$$

### 5.4.4 Automatic Best MCDM Selection

```python
def select_best_mcdm(mcdm_results, baseline, weights):
    """
    Compare all MCDM selections against baseline and 
    automatically select the best performing method
    """
    performance = {}
    
    for method, result in mcdm_results.items():
        selected_solution = result['best_solution']
        
        # Calculate improvement for each objective
        improvements = []
        for j in range(5):
            if j in [0, 1, 3]:  # Minimize: Time, Cost, Safety
                imp = (baseline[j] - selected_solution[j]) / baseline[j] * 100
            else:  # Maximize: Quality, Sustainability
                imp = (selected_solution[j] - baseline[j]) / baseline[j] * 100
            improvements.append(imp)
        
        # Calculate Overall Improvement Score
        ois = sum(w * imp for w, imp in zip(weights, improvements))
        
        # Count objectives improved
        dominance_count = sum(1 for imp in improvements if imp > 0)
        
        # Stability score (average rank correlation with other methods)
        stability = calculate_rank_stability(method, mcdm_results)
        
        performance[method] = {
            'improvements': improvements,
            'OIS': ois,
            'dominance_count': dominance_count,
            'stability': stability,
            'composite_score': 0.5 * ois + 0.3 * dominance_count * 20 + 0.2 * stability * 100
        }
    
    # Select method with highest composite score
    best_method = max(performance, key=lambda m: performance[m]['composite_score'])
    
    return best_method, performance
```

### 5.4.5 MCDM Selection Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Overall Improvement Score (OIS) | 50% | Weighted sum of improvements across 5 objectives |
| Dominance Count (DC) | 30% | Number of objectives improved over baseline |
| Stability Score (SS) | 20% | Consistency of ranking across sensitivity analysis |

**Composite Score:**

$$CS_m = 0.5 \cdot OIS_m + 0.3 \cdot \frac{DC_m}{5} \times 100 + 0.2 \cdot SS_m$$

### 5.4.6 Example Comparison Table

| MCDM Method | Time Δ | Cost Δ | Quality Δ | Safety Δ | Sustain Δ | OIS | DC | Best? |
|-------------|--------|--------|-----------|----------|-----------|-----|----|----|
| TOPSIS | -12.5% | +3.2% | +8.1% | -18.4% | +5.6% | 6.42 | 4/5 | |
| VIKOR | -8.3% | -2.1% | +6.5% | -15.2% | +4.8% | 5.18 | 5/5 | ✓ |
| PROMETHEE-II | -15.1% | +5.8% | +9.2% | -20.1% | +3.2% | 5.84 | 4/5 | |
| GRA | -10.2% | +1.5% | +7.3% | -16.8% | +6.1% | 6.10 | 4/5 | |
| ELECTRE-III | -9.8% | +2.3% | +5.9% | -14.5% | +4.2% | 4.62 | 4/5 | |
| Entropy-WASPAS | -11.4% | -0.5% | +7.8% | -17.9% | +5.3% | 6.38 | 5/5 | |

*Note: Negative values = improvement for minimization objectives (Time, Cost, Safety)*

## 5.5 Entropy-Based Objective Weight Determination

When stakeholder weights are unavailable, entropy method determines objective importance:

**Step 1: Normalize objectives**

$$p_{ij} = \frac{f_j(x_i)}{\sum_{i=1}^{|A|} f_j(x_i)}$$

**Step 2: Calculate entropy**

$$E_j = -\frac{1}{\ln(|A|)} \sum_{i=1}^{|A|} p_{ij} \cdot \ln(p_{ij})$$

**Step 3: Derive weights**

$$w_j = \frac{1 - E_j}{\sum_{k=1}^{M} (1 - E_k)}$$

## 5.6 Validation of Auto-Selected MCDM

**Rank Correlation (Spearman's ρ):**

$$\rho_s = 1 - \frac{6 \sum_{i=1}^{n} d_i^2}{n(n^2 - 1)}$$

**Expert Agreement Rate:**

$$EAR = \frac{|\text{Auto-selected matches Expert choice}|}{|\text{Total case studies}|}$$

**Cross-Validation:**
- Apply to 30 independent optimization runs
- Calculate consistency of best MCDM selection
- Statistical significance testing (Friedman test)

---

# 6. Validation & Verification Framework

## 6.1 Performance Metrics

**Convergence:**
- Generational Distance (GD)
- Inverted Generational Distance (IGD)

**Diversity:**
- Spacing (SP)
- Hypervolume (HV)

**MCDM Selection Accuracy:**
- Expert agreement rate
- Rank correlation with expert rankings
- Decision consistency across runs

## 6.2 Statistical Testing

- **30 independent runs** per case study
- **Wilcoxon Rank-Sum Test** for pairwise comparison
- **Friedman Test** for MCDM method comparison
- **Significance level** α = 0.05

---

# 7. Case Studies

## 7.1 Case Study 1: Urban Overpass Bridge (15 Activities)

**Project Description:**
- **Type:** Prestressed concrete box girder overpass
- **Span:** 45m single span
- **Location:** Urban intersection with heavy traffic constraints
- **Complexity:** Night work restrictions, utility relocations, traffic management

**Activity Network (15 Activities, 4 Methods Each):**

| ID | Activity | Predecessors | Methods |
|----|----------|--------------|---------|
| A1 | Site Preparation & Traffic Control | - | 4 |
| A2 | Utility Relocation | A1 | 4 |
| A3 | Foundation Excavation (Abutment 1) | A2 | 4 |
| A4 | Foundation Excavation (Abutment 2) | A2 | 4 |
| A5 | Pile Installation (Abutment 1) | A3 | 4 |
| A6 | Pile Installation (Abutment 2) | A4 | 4 |
| A7 | Abutment 1 Construction | A5 | 4 |
| A8 | Abutment 2 Construction | A6 | 4 |
| A9 | Bearing Installation | A7, A8 | 4 |
| A10 | Girder Fabrication (Off-site) | A1 | 4 |
| A11 | Girder Transportation & Erection | A9, A10 | 4 |
| A12 | Deck Slab Construction | A11 | 4 |
| A13 | Barrier & Railing Installation | A12 | 4 |
| A14 | Approach Slab & Paving | A13 | 4 |
| A15 | Final Inspection & Opening | A14 | 4 |

**Decision Space:** $4^{15} = 1,073,741,824$ combinations

**Complexity Factors:**
- Parallel abutment construction paths
- Off-site fabrication synchronization
- Night work options with cost/quality trade-offs
- Environmental restrictions on pile driving

---

## 7.2 Case Study 2: River Crossing Bridge (18 Activities)

**Project Description:**
- **Type:** Steel-concrete composite continuous girder
- **Span:** 3 spans (40m + 60m + 40m = 140m total)
- **Location:** River crossing with environmental sensitivity
- **Complexity:** Water work permits, seasonal flow variations, fish migration windows

**Activity Network (18 Activities, 4 Methods Each):**

| ID | Activity | Predecessors | Methods |
|----|----------|--------------|---------|
| B1 | Environmental Permits & Baseline Survey | - | 4 |
| B2 | Access Road Construction | B1 | 4 |
| B3 | Temporary Cofferdam (Pier 1) | B2 | 4 |
| B4 | Temporary Cofferdam (Pier 2) | B2 | 4 |
| B5 | Pier 1 Foundation (Caisson) | B3 | 4 |
| B6 | Pier 2 Foundation (Caisson) | B4 | 4 |
| B7 | Abutment A Construction | B2 | 4 |
| B8 | Abutment B Construction | B2 | 4 |
| B9 | Pier 1 Column Construction | B5 | 4 |
| B10 | Pier 2 Column Construction | B6 | 4 |
| B11 | Pier Cap 1 Construction | B9 | 4 |
| B12 | Pier Cap 2 Construction | B10 | 4 |
| B13 | Steel Girder Fabrication | B1 | 4 |
| B14 | Girder Erection Span 1 | B7, B11, B13 | 4 |
| B15 | Girder Erection Span 2 | B11, B12, B14 | 4 |
| B16 | Girder Erection Span 3 | B12, B8, B15 | 4 |
| B17 | Composite Deck Construction | B16 | 4 |
| B18 | Finishing & Load Testing | B17 | 4 |

**Decision Space:** $4^{18} = 68,719,476,736$ combinations

**Complexity Factors:**
- Parallel pier construction with water work constraints
- Seasonal construction windows (fish migration)
- Sequential span erection with intermediate dependencies
- Cofferdam removal scheduling affects environmental compliance

---

## 7.3 Case Study 3: Highway Interchange Ramp Bridge (20 Activities)

**Project Description:**
- **Type:** Post-tensioned concrete box girder on curved alignment
- **Span:** 5 spans curved ramp (25m each = 125m total)
- **Location:** Highway interchange with 24/7 traffic maintenance
- **Complexity:** Curved geometry, post-tensioning sequences, staged construction

**Activity Network (20 Activities, 4 Methods Each):**

| ID | Activity | Predecessors | Methods |
|----|----------|--------------|---------|
| C1 | Traffic Management Setup | - | 4 |
| C2 | Survey & Layout (Curved Alignment) | C1 | 4 |
| C3 | Foundation Pier 1 | C2 | 4 |
| C4 | Foundation Pier 2 | C2 | 4 |
| C5 | Foundation Pier 3 | C2 | 4 |
| C6 | Foundation Pier 4 | C2 | 4 |
| C7 | Pier 1 Column & Cap | C3 | 4 |
| C8 | Pier 2 Column & Cap | C4 | 4 |
| C9 | Pier 3 Column & Cap | C5 | 4 |
| C10 | Pier 4 Column & Cap | C6 | 4 |
| C11 | Abutment A (Start) | C2 | 4 |
| C12 | Abutment B (End) | C2 | 4 |
| C13 | Falsework Span 1-2 | C7, C8, C11 | 4 |
| C14 | Falsework Span 3-4 | C9, C10 | 4 |
| C15 | Box Girder Pour Span 1-2 | C13 | 4 |
| C16 | Box Girder Pour Span 3-4-5 | C14, C12 | 4 |
| C17 | Post-Tensioning Phase 1 | C15 | 4 |
| C18 | Post-Tensioning Phase 2 | C16, C17 | 4 |
| C19 | Barrier, Drainage & Finishing | C18 | 4 |
| C20 | Final Testing & Opening | C19 | 4 |

**Decision Space:** $4^{20} = 1,099,511,627,776$ combinations

**Complexity Factors:**
- Multiple parallel foundation activities (4 piers + 2 abutments)
- Staged falsework with load transfer requirements
- Post-tensioning sequence constraints (Phase 1 before Phase 2)
- Curved geometry affecting equipment selection and productivity
- 24/7 traffic maintenance impacts safety and cost

---

## 7.4 Case Study Comparison Summary

| Aspect | Case 1: Urban Overpass | Case 2: River Crossing | Case 3: Highway Ramp |
|--------|------------------------|------------------------|----------------------|
| Activities | 15 | 18 | 20 |
| Methods/Activity | 4 | 4 | 4 |
| Decision Space | 1.07 × 10⁹ | 6.87 × 10¹⁰ | 1.10 × 10¹² |
| Network Complexity | Medium | High | Very High |
| Parallel Paths | 2 | 4 | 6 |
| Critical Constraints | Traffic, Night Work | Environmental, Seasonal | Geometry, Staging |
| Duration Range | 8-14 months | 18-30 months | 12-20 months |
| Budget Range | $2-4M | $8-15M | $5-10M |

---

# 8. Expected Results & Analysis

## 8.1 Optimization Results Analysis

For each case study:
1. **Pareto front visualization** (5D parallel coordinates, 2D projections)
2. **Trade-off analysis** between competing objectives
3. **Solution diversity assessment** across the 5 objectives
4. **Constraint satisfaction verification**

## 8.2 MCDM Selection Analysis

| Case Study | Expected Optimal MCDM | Justification |
|------------|----------------------|---------------|
| Urban Overpass | TOPSIS or VIKOR | Balanced trade-offs, clear preferences |
| River Crossing | GRA or Entropy-WASPAS | Environmental uncertainty, complex constraints |
| Highway Ramp | PROMETHEE-II | Multiple stakeholders, staging dependencies |

## 8.3 Sensitivity Analysis

- **Objective weight sensitivity:** Vary weights ±20% and analyze ranking stability
- **Method comparison:** Apply all 6 MCDM methods and compare rankings
- **Expert validation:** Compare auto-selected MCDM results with expert choices

---

# 9. Expected Contributions

## 9.1 Theoretical Contributions

1. **5D Objective Formulation** - Comprehensive mathematical model for Time-Cost-Quality-Safety-Sustainability
2. **Automated MCDM Selection** - Novel framework for context-aware optimal method selection
3. **Bridge-Specific Objective Functions** - Tailored quality, safety, and sustainability metrics

## 9.2 Practical Contributions

1. **Decision Support Framework** - Ready-to-use tool for bridge construction planners
2. **Trade-off Visualization** - 5D analysis tools for stakeholder communication
3. **Case Study Benchmarks** - Three complex bridge projects for future research

---

# 10. Implementation

## 10.1 Software Stack

| Component | Tool/Library |
|-----------|--------------|
| Optimization | pymoo (NSGA-III) |
| MCDM Methods | scikit-criteria, pyDecision |
| Visualization | matplotlib, plotly |
| Statistical Tests | scipy.stats |

## 10.2 Reproducibility

- All case study data provided in supplementary materials
- Python code available on GitHub
- Random seeds documented for all experiments

---

# 11. Publication Plan

## 11.1 Target Journal

**Automation in Construction** (IF: 10.3)

## 11.2 Manuscript Structure

| Section | Words |
|---------|-------|
| Abstract | 250 |
| Introduction | 1,500 |
| Literature Review | 2,000 |
| Methodology (5D Model + MCDM) | 3,500 |
| Case Studies | 3,000 |
| Results & Discussion | 3,500 |
| Conclusions | 800 |
| **Total** | **~14,500** |

---

*Document Version: 2.0*
*Updated: January 20, 2026*
*Changes: Replaced FA-MaOEA with NSGA-III, added automated optimal MCDM selection, updated case studies (10-20 activities)*
