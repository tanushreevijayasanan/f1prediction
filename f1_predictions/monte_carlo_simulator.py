import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger("monte_carlo")
@dataclass
class DriverState:
    """Current race state for one driver."""
    code: str
    position: int
    avg_speed: float
    tire_compound: str
    tire_age: int
    pit_history: List[int] = field(default_factory=list)  # lap numbers where pit occurred
    laps_in_current_stint: int = 0
    dnf: bool = False
    
    def copy(self):
        return DriverState(
            code=self.code,
            position=self.position,
            avg_speed=self.avg_speed,
            tire_compound=self.tire_compound,
            tire_age=self.tire_age,
            pit_history=self.pit_history.copy(),
            laps_in_current_stint=self.laps_in_current_stint,
            dnf=self.dnf,
        )


@dataclass
class TireDegradationModel:
    """
    Per-stint tire degradation curve.
    pace_drop_per_lap = base_rate * (lap_number_in_stint ^ exp)
    """
    # Compound -> (base_rate, exponent)
    degradation: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "SOFT": (0.0045, 1.2),    # Soft tires degrade fastest (non-linear)
        "MEDIUM": (0.0025, 1.1),  # Medium more stable
        "HARD": (0.0012, 1.0),    # Hard barely degrades
        "INTER": (0.0035, 1.15),  # Intermediates worse than medium
        "WET": (0.0040, 1.2),     # Wets similar to soft on wet track
    })
    
    def get_pace_drop(self, compound: str, lap_in_stint: int) -> float:
        """
        Return pace drop (as fraction of base speed) for this lap.
        Example: 0.02 = 2% slower than stint start.
        """
        if compound not in self.degradation:
            compound = "MEDIUM"
        
        base_rate, exp = self.degradation[compound]
        return base_rate * (lap_in_stint ** exp)


class MonteCarloRaceSimulator:
    """
    Monte Carlo simulator for F1 races.
    
    Runs multiple race simulations forward to completion, tracking:
    - Lap times (with degradation)
    - Pit stops (using trained pit model)
    - Position changes
    - DNF risk
    
    Outputs probability distributions over finishing positions.
    """
    
    def __init__(
        self,
        laptime_model,
        tire_model,
        pit_model,
        current_state: Dict[str, Any],
        n_simulations: int = 50000,
    ):
        """
        Initialize simulator.
        
        Args:
            laptime_model: Trained XGBoost lap-time predictor
            tire_model: Tire degradation model (TireDegradationModel)
            pit_model: Trained pit probability model
            current_state: Dict with current race state from inference_engine
                - "n_drivers": int
                - "current_lap": int
                - "total_laps": int
                - "pre_race": {driver_code -> {"grid_position": int, ...}}
                - "speed_rank": {driver_code -> float}
                - "current_compound": {driver_code -> str}
                - "tire_age": {driver_code -> int}
                - etc.
            n_simulations: Number of Monte Carlo samples
        """
        self.laptime_model = laptime_model
        self.tire_model = tire_model
        self.pit_model = pit_model
        self.current_state = current_state
        self.n_simulations = n_simulations
        
        # Extract initial conditions
        self.current_lap = current_state.get("current_lap", 0)
        self.total_laps = current_state.get("total_laps", 70)
        self.remaining_laps = max(0, self.total_laps - self.current_lap)
        
        # Late-race probability calibration
        self.race_completion_pct = self.current_lap / self.total_laps
        self.is_late_race = self.race_completion_pct > 0.7
        
        # DRS train parameters
        self.drs_effectiveness = 0.3  # Speed boost in DRS zones
        self.dirty_air_penalty = 0.15  # Speed loss when following closely
        
        # Safety car risk (increases in late race)
        self.sc_base_risk = 0.002  # 0.2% per lap base risk
        self.sc_late_race_multiplier = 2.0 if self.is_late_race else 1.0
    
    def _initialize_drivers(self) -> Dict[str, DriverState]:
        """Create initial DriverState for each driver."""
        drivers = {}
        
        for code in self.current_state.get("pre_race", {}).keys():
            drivers[code] = DriverState(
                code=code,
                position=self.current_state.get("speed_rank", {}).get(code, 10),
                avg_speed=100.0,  # Normalized
                tire_compound=self.current_state.get("current_compound", {}).get(code, "SOFT"),
                tire_age=self.current_state.get("tire_age", {}).get(code, 0),
                laps_in_current_stint=self.current_state.get("current_stint", {}).get(code, 0),
            )
        
        return drivers
    
    def _simulate_lap(
        self,
        drivers: Dict[str, DriverState],
        lap_no: int,
    ) -> Dict[str, DriverState]:
        """
        Simulate one lap forward with DRS trains, dirty air, and safety car risk.
        """
        drivers = {k: v.copy() for k, v in drivers.items()}
        
        # Safety car check
        sc_deployed = self._check_safety_car(lap_no)
        
        for code, driver in drivers.items():
            if driver.dnf:
                continue
            
            # Simulate tire degradation for this lap
            pace_drop = self.tire_model.get_pace_drop(
                driver.tire_compound,
                driver.laps_in_current_stint + 1
            )
            
            # Apply dirty air penalty for midfield drivers (positions 6-15)
            if 6 <= driver.position <= 15:
                dirty_air_effect = self.dirty_air_penalty * (1.0 - driver.position / 15.0)
                pace_drop += dirty_air_effect
            
            # Apply DRS boost for drivers within 1 second of car ahead
            if self._has_drs_opportunity(driver, drivers):
                pace_drop -= self.drs_effectiveness * 0.5  # DRS can offset some degradation
            
            driver.avg_speed *= (1.0 - pace_drop)
            driver.tire_age += 1
            driver.laps_in_current_stint += 1
            
            # Safety car affects pit strategy
            if sc_deployed:
                pit_prob = self._predict_pit_probability_under_sc(driver, lap_no)
            else:
                pit_prob = self._predict_pit_probability(driver, lap_no)
                
            if np.random.random() < pit_prob:
                driver = self._execute_pit_stop(driver, lap_no)
            
            # DNF risk (slightly higher in late race)
            dnf_risk = 0.001 * (1.5 if self.is_late_race else 1.0)
            if np.random.random() < dnf_risk:
                driver.dnf = True
        
        # Reorder drivers by position based on speed
        speeds = [(k, v.avg_speed) for k, v in drivers.items() if not v.dnf]
        speeds.sort(key=lambda x: -x[1])  # Descending speed
        
        for position, (code, _) in enumerate(speeds, start=1):
            drivers[code].position = position
        
        return drivers
    
    def _predict_pit_probability(self, driver: DriverState, lap_no: int) -> float:
        """
        Improved pit probability with race phase consideration.
        """
        # Base probability on tire wear and stint length
        base_prob = 0.05
        
        # Tire compound-specific wear rates
        wear_rates = {"SOFT": 0.8, "MEDIUM": 0.5, "HARD": 0.3}
        wear_mult = wear_rates.get(driver.tire_compound, 0.5)
        
        # Stint length pressure
        if driver.laps_in_current_stint > 25:
            stint_pressure = 0.7
        elif driver.laps_in_current_stint > 20:
            stint_pressure = 0.4
        elif driver.laps_in_current_stint > 15:
            stint_pressure = 0.2
        else:
            stint_pressure = 0.0
        
        # Position-based strategy (midfield more likely to pit for undercut)
        if 6 <= driver.position <= 12:
            position_mult = 1.3
        else:
            position_mult = 1.0
        
        # Late race aggressive strategy
        if self.is_late_race and self.remaining_laps <= 10:
            late_race_mult = 1.5
        else:
            late_race_mult = 1.0
        
        total_prob = base_prob + (stint_pressure * wear_mult * position_mult * late_race_mult)
        return min(0.9, total_prob)  # Cap at 90%
    
    def _execute_pit_stop(self, driver: DriverState, lap_no: int) -> DriverState:
        """
        Execute a pit stop: change tires, reset age.
        Pit stop takes ~20-23s (roughly 1 lap equivalent time loss).
        """
        driver.pit_history.append(lap_no)
        driver.tire_age = 0
        driver.laps_in_current_stint = 0
        
        # Strategy: choose compound based on remaining laps and pace
        remaining = self.total_laps - lap_no
        if remaining > 30:
            driver.tire_compound = "MEDIUM"
        elif remaining > 15:
            driver.tire_compound = "HARD"
        else:
            driver.tire_compound = "SOFT"  # Aggressive finish
        
        # Speed penalty for pit stop (temporary)
        driver.avg_speed *= 0.95
        
        return driver
    
    def _has_drs_opportunity(self, driver: DriverState, drivers: Dict[str, DriverState]) -> bool:
        """Check if driver has DRS opportunity (within 1 second of car ahead)."""
        if driver.position <= 1:  # Leader never has DRS
            return False
        
        # Find car ahead
        ahead_driver = None
        for code, other in drivers.items():
            if other.position == driver.position - 1 and not other.dnf:
                ahead_driver = other
                break
        
        if ahead_driver:
            # Simplified DRS check: 30% chance if within reasonable speed range
            speed_gap = abs(driver.avg_speed - ahead_driver.avg_speed) / ahead_driver.avg_speed
            return speed_gap < 0.05 and np.random.random() < 0.3
        
        return False
    
    def _check_safety_car(self, lap_no: int) -> bool:
        """Simulate safety car deployment probability."""
        # Higher risk in late race and for specific laps
        sc_risk = self.sc_base_risk * self.sc_late_race_multiplier
        
        # Certain laps have higher SC probability (based on historical data)
        high_risk_laps = [1, 15, 30, 45]  # Start, early stops, mid-race, late stops
        if lap_no in high_risk_laps:
            sc_risk *= 3.0
        
        return np.random.random() < sc_risk
    
    def _predict_pit_probability_under_sc(self, driver: DriverState, lap_no: int) -> float:
        """Higher pit probability under safety car (free pit stop)."""
        base_prob = self._predict_pit_probability(driver, lap_no)
        sc_multiplier = 2.5  # Much more likely to pit under SC
        
        # But only if tires are worn
        if driver.laps_in_current_stint < 10:
            sc_multiplier = 1.2
        
        return min(0.95, base_prob * sc_multiplier)
    
    def _calibrate_late_race_probabilities(self, result: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Apply realistic late-race probability calibration."""
        if not self.is_late_race:
            return result
        
        # Find current leader
        leader = min(result.keys(), key=lambda k: self.current_state.get("speed_rank", {}).get(k, 999))
        
        for code, probs in result.items():
            # Exponential boost for leader based on race completion
            if code == leader:
                # Leader advantage grows exponentially as race progresses
                completion_bonus = self.race_completion_pct ** 2  # Squared for exponential growth
                leader_boost = 0.5 + (completion_bonus * 0.4)  # 50% to 90% boost
                
                # Apply to win probability
                old_win_prob = probs["win_probability"]
                new_win_prob = min(0.95, old_win_prob + leader_boost)
                
                # Redistribute probability to other drivers
                excess = new_win_prob - old_win_prob
                probs["win_probability"] = new_win_prob
                
                # Reduce other drivers' win probabilities proportionally
                for other_code in result:
                    if other_code != code and other_code != leader:
                        result[other_code]["win_probability"] *= (1.0 - excess / 0.8)
            
            # Reduce midfield volatility in very late race
            elif 5 <= self.current_state.get("speed_rank", {}).get(code, 999) <= 15:
                # Stabilize midfield positions in final 10%
                if self.race_completion_pct > 0.9:
                    volatility_reduction = 0.3
                    for pos in range(1, 21):
                        if f"p{pos}" in probs:
                            # Concentrate probability around current position
                            current_pos = self.current_state.get("speed_rank", {}).get(code, 10)
                            if abs(pos - current_pos) <= 2:
                                probs[f"p{pos}"] *= (1.0 + volatility_reduction)
                            else:
                                probs[f"p{pos}"] *= (1.0 - volatility_reduction * 0.5)
        
        return result
    
    def simulate(self) -> Dict[str, Dict[str, float]]:
        """
        Run N simulations forward to race end.
        
        Returns:
            Dict[driver_code -> {
                "p1": float,          # Prob of finishing P1
                "p2": float,          # Prob of finishing P2
                ...
                "p20": float,         # Prob of finishing P20
                "podium": float,      # Prob Top 3
                "top5": float,        # Prob Top 5
                "top10": float,       # Prob Top 10
                "points": float,      # Prob of points finish (Top 10)
                "expected_position": float,  # E[finishing position]
                "win_probability": float,    # Same as "p1"
            }]
        """
        
        all_finishes = []
        
        for sim_idx in range(self.n_simulations):
            # Initialize drivers for this simulation
            drivers = self._initialize_drivers()
            
            # Simulate remaining laps
            for lap_no in range(self.current_lap + 1, self.total_laps + 1):
                drivers = self._simulate_lap(drivers, lap_no)
            
            # Record final positions
            finish_order = {}
            dnf_drivers = []
            
            for code, driver in drivers.items():
                if driver.dnf:
                    dnf_drivers.append(code)
                else:
                    finish_order[code] = driver.position
            
            # DNF drivers finish last (classified but no points)
            for position, code in enumerate(dnf_drivers, start=len(finish_order) + 1):
                finish_order[code] = position
            
            all_finishes.append(finish_order)
        
        # Aggregate simulation results into probability distributions
        result = {}
        
        for code in self.current_state.get("pre_race", {}).keys():
            positions = [f.get(code, 20) for f in all_finishes]
            
            # Position probabilities
            pos_probs = {}
            for pos in range(1, 21):
                pos_probs[f"p{pos}"] = np.mean([p == pos for p in positions])
            
            # Aggregate probabilities
            podium = np.mean([p <= 3 for p in positions])
            top5 = np.mean([p <= 5 for p in positions])
            top10 = np.mean([p <= 10 for p in positions])
            points = top10  # F1 points awarded to Top 10
            expected_pos = np.mean(positions)
            
            result[code] = {
                **pos_probs,
                "podium": podium,
                "top5": top5,
                "top10": top10,
                "points": points,
                "expected_position": expected_pos,
                "win_probability": pos_probs["p1"],
            }
        
        # Apply late-race probability calibration
        calibrated_result = self._calibrate_late_race_probabilities(result)
        
        return calibrated_result


# Standalone utility: formatting probability distributions
def format_prob_distribution(
    driver_code: str,
    probabilities: Dict[str, float],
    include_all_positions: bool = False,
) -> str:
    """
    Pretty-print probability distribution for one driver.
    
    Example output:
        VER: Win 38% | Podium 72% | Top 5 88% | Points 96% | Expected: P5
    """
    win_p = probabilities["p1"]
    podium_p = probabilities["podium"]
    top5_p = probabilities["top5"]
    points_p = probabilities["points"]
    expected = probabilities["expected_position"]
    
    s = f"{driver_code}: "
    s += f"Win {win_p:.0%} | Podium {podium_p:.0%} | Top5 {top5_p:.0%} | Points {points_p:.0%} | "
    s += f"Expected: P{expected:.0f}"
    
    return s

