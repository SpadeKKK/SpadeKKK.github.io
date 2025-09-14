#!/usr/bin/env python3
"""
Statistical Engine for A/B Test Designer
Advanced statistical analysis, hypothesis testing, and experimental design
"""

import numpy as np
from scipy import stats
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum
import logging
import warnings

# Suppress scipy warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Configure logging
logger = logging.getLogger(__name__)

class TestType(Enum):
    """Enumeration of supported statistical test types"""
    TWO_SAMPLE_TTEST = "two_sample_ttest"
    PROPORTION_ZTEST = "proportion_ztest"
    CHI_SQUARE = "chi_square"
    MANN_WHITNEY = "mann_whitney"
    PAIRED_TTEST = "paired_ttest"
    FISHER_EXACT = "fisher_exact"

class EffectSizeType(Enum):
    """Enumeration of effect size measures"""
    COHENS_D = "cohens_d"
    GLASS_DELTA = "glass_delta"
    HEDGES_G = "hedges_g"
    COHENS_H = "cohens_h"
    CRAMER_V = "cramer_v"

@dataclass
class TestResult:
    """Data class to store comprehensive test results"""
    test_statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    effect_size: float
    effect_size_type: str
    power: float
    sample_size_control: int
    sample_size_treatment: int
    test_type: str
    alpha: float
    recommendation: str
    interpretation: str

class StatisticalEngine:
    """
    Advanced statistical engine for A/B testing and experimental design
    """
    
    def __init__(self, alpha: float = 0.05, beta: float = 0.2):
        """
        Initialize the statistical engine
        
        Args:
            alpha: Type I error rate (significance level)
            beta: Type II error rate (1 - power)
        """
        self.alpha = alpha
        self.beta = beta
        self.power = 1 - beta
        
        # Validate parameters
        if not (0 < alpha < 1):
            raise ValueError("Alpha must be between 0 and 1")
        if not (0 < beta < 1):
            raise ValueError("Beta must be between 0 and 1")
    
    def detect_test_type(self, data: Dict) -> TestType:
        """
        Automatically detect the appropriate statistical test based on data characteristics
        
        Args:
            data: Dictionary containing data characteristics
            
        Returns:
            Appropriate TestType enum value
        """
        try:
            metric_type = data.get('metric_type', 'proportion')
            sample_size = data.get('sample_size', 100)
            is_paired = data.get('is_paired', False)
            
            if metric_type == 'proportion':
                if sample_size < 30:
                    return TestType.FISHER_EXACT
                else:
                    return TestType.PROPORTION_ZTEST
            elif metric_type == 'continuous':
                if is_paired:
                    return TestType.PAIRED_TTEST
                else:
                    return TestType.TWO_SAMPLE_TTEST
            elif metric_type == 'categorical':
                return TestType.CHI_SQUARE
            elif metric_type == 'ordinal':
                return TestType.MANN_WHITNEY
            else:
                return TestType.PROPORTION_ZTEST  # Default
                
        except Exception as e:
            logger.warning(f"Error detecting test type: {e}. Using default.")
            return TestType.PROPORTION_ZTEST
    
    def calculate_effect_size(self, data: Dict, test_type: TestType) -> Tuple[float, str]:
        """
        Calculate appropriate effect size based on test type
        
        Args:
            data: Test data
            test_type: Type of statistical test
            
        Returns:
            Tuple of (effect_size, effect_size_type)
        """
        try:
            if test_type == TestType.PROPORTION_ZTEST:
                p1 = data.get('control_rate', 0)
                p2 = data.get('treatment_rate', 0)
                return self._cohens_h(p1, p2), EffectSizeType.COHENS_H.value
                
            elif test_type == TestType.TWO_SAMPLE_TTEST:
                mean1 = data.get('control_mean', 0)
                mean2 = data.get('treatment_mean', 0)
                std1 = data.get('control_std', 1)
                std2 = data.get('treatment_std', 1)
                n1 = data.get('control_n', 100)
                n2 = data.get('treatment_n', 100)
                return self._cohens_d(mean1, mean2, std1, std2, n1, n2), EffectSizeType.COHENS_D.value
                
            elif test_type == TestType.CHI_SQUARE:
                # Placeholder for Cramer's V calculation
                return 0.1, EffectSizeType.CRAMER_V.value
                
            else:
                # Default to Cohen's d
                return 0.2, EffectSizeType.COHENS_D.value
                
        except Exception as e:
            logger.error(f"Error calculating effect size: {e}")
            return 0.0, "unknown"
    
    def _cohens_d(self, mean1: float, mean2: float, std1: float, std2: float, 
                  n1: int, n2: int) -> float:
        """Calculate Cohen's d effect size"""
        try:
            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
            if pooled_std == 0:
                return 0.0
            return (mean2 - mean1) / pooled_std
        except Exception:
            return 0.0
    
    def _cohens_h(self, p1: float, p2: float) -> float:
        """Calculate Cohen's h effect size for proportions"""
        try:
            if p1 <= 0 or p1 >= 1 or p2 <= 0 or p2 >= 1:
                # Use relative difference for edge cases
                return (p2 - p1) / p1 if p1 > 0 else 0
            return 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))
        except Exception:
            return 0.0
    
    def power_analysis(self, effect_size: float, sample_size: int, 
                      test_type: TestType, alpha: float = None) -> float:
        """
        Calculate statistical power for given parameters
        
        Args:
            effect_size: Expected effect size
            sample_size: Sample size per group
            test_type: Type of statistical test
            alpha: Significance level (uses instance alpha if None)
            
        Returns:
            Statistical power (0-1)
        """
        try:
            if alpha is None:
                alpha = self.alpha
                
            if test_type == TestType.PROPORTION_ZTEST:
                return self._power_proportion_test(effect_size, sample_size, alpha)
            elif test_type == TestType.TWO_SAMPLE_TTEST:
                return self._power_ttest(effect_size, sample_size, alpha)
            else:
                # Approximate power calculation
                z_alpha = stats.norm.ppf(1 - alpha/2)
                z_beta = effect_size * np.sqrt(sample_size/2) - z_alpha
                return max(0, min(1, stats.norm.cdf(z_beta)))
                
        except Exception as e:
            logger.error(f"Error calculating power: {e}")
            return 0.8  # Conservative default
    
    def _power_proportion_test(self, effect_size: float, sample_size: int, 
                              alpha: float) -> float:
        """Calculate power for proportion test"""
        try:
            z_alpha = stats.norm.ppf(1 - alpha/2)
            # Approximate power calculation for proportion tests
            z_beta = abs(effect_size) * np.sqrt(sample_size/2) - z_alpha
            power = stats.norm.cdf(z_beta)
            return max(0, min(1, power))
        except Exception:
            return 0.8
    
    def _power_ttest(self, effect_size: float, sample_size: int, alpha: float) -> float:
        """Calculate power for t-test"""
        try:
            df = 2 * sample_size - 2
            t_alpha = stats.t.ppf(1 - alpha/2, df)
            ncp = abs(effect_size) * np.sqrt(sample_size/2)  # Non-centrality parameter
            
            # Approximate power using t-distribution
            power = 1 - stats.t.cdf(t_alpha, df, ncp) + stats.t.cdf(-t_alpha, df, ncp)
            return max(0, min(1, power))
        except Exception:
            return 0.8
    
    def sample_size_calculator(self, effect_size: float, power: float = None, 
                              test_type: TestType = TestType.PROPORTION_ZTEST,
                              alpha: float = None) -> int:
        """
        Calculate required sample size for given parameters
        
        Args:
            effect_size: Expected effect size
            power: Desired statistical power
            test_type: Type of statistical test
            alpha: Significance level
            
        Returns:
            Required sample size per group
        """
        try:
            if power is None:
                power = self.power
            if alpha is None:
                alpha = self.alpha
                
            z_alpha = stats.norm.ppf(1 - alpha/2)
            z_beta = stats.norm.ppf(power)
            
            if test_type == TestType.PROPORTION_ZTEST:
                # Assume baseline rate of 10% if not provided
                p1 = 0.1
                p2 = p1 * (1 + effect_size / 100)
                p_pooled = (p1 + p2) / 2
                
                if p2 == p1:  # Avoid division by zero
                    return 1000
                    
                n = (2 * p_pooled * (1 - p_pooled) * (z_alpha + z_beta)**2) / (p2 - p1)**2
                return max(30, int(np.ceil(n)))
                
            elif test_type == TestType.TWO_SAMPLE_TTEST:
                # For continuous variables
                if effect_size == 0:
                    return 1000
                n = (2 * (z_alpha + z_beta)**2) / (effect_size/100)**2
                return max(30, int(np.ceil(n)))
                
            else:
                # Default calculation
                n = (2 * (z_alpha + z_beta)**2) / (effect_size/100)**2
                return max(50, int(np.ceil(n)))
                
        except Exception as e:
            logger.error(f"Error calculating sample size: {e}")
            return 1000  # Safe default
    
    def run_proportion_test(self, control_conversions: int, control_visitors: int,
                           treatment_conversions: int, treatment_visitors: int) -> TestResult:
        """
        Run two-proportion z-test
        
        Args:
            control_conversions: Number of conversions in control group
            control_visitors: Number of visitors in control group
            treatment_conversions: Number of conversions in treatment group
            treatment_visitors: Number of visitors in treatment group
            
        Returns:
            TestResult object with comprehensive results
        """
        try:
            # Input validation
            if any(x < 0 for x in [control_conversions, control_visitors, 
                                  treatment_conversions, treatment_visitors]):
                raise ValueError("All inputs must be non-negative")
                
            if control_visitors == 0 or treatment_visitors == 0:
                raise ValueError("Visitor counts must be greater than zero")
                
            if control_conversions > control_visitors or treatment_conversions > treatment_visitors:
                raise ValueError("Conversions cannot exceed visitors")
            
            # Calculate proportions
            p1 = control_conversions / control_visitors
            p2 = treatment_conversions / treatment_visitors
            
            # Pooled proportion for test statistic
            p_pooled = (control_conversions + treatment_conversions) / (control_visitors + treatment_visitors)
            
            # Standard error
            se = np.sqrt(p_pooled * (1 - p_pooled) * (1/control_visitors + 1/treatment_visitors))
            
            # Test statistic and p-value
            if se == 0:
                z_score = 0
                p_value = 1.0
            else:
                z_score = (p2 - p1) / se
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            
            # Confidence interval for difference
            se_diff = np.sqrt((p1 * (1 - p1) / control_visitors) + (p2 * (1 - p2) / treatment_visitors))
            z_critical = stats.norm.ppf(1 - self.alpha/2)
            diff = p2 - p1
            ci_lower = diff - z_critical * se_diff
            ci_upper = diff + z_critical * se_diff
            
            # Effect size (Cohen's h)
            effect_size = self._cohens_h(p1, p2)
            
            # Power calculation
            power = self.power_analysis(abs(effect_size), min(control_visitors, treatment_visitors), 
                                      TestType.PROPORTION_ZTEST)
            
            # Generate recommendation and interpretation
            recommendation = self._generate_recommendation(p_value, effect_size, power)
            interpretation = self._interpret_results(p_value, effect_size, power, TestType.PROPORTION_ZTEST)
            
            return TestResult(
                test_statistic=z_score,
                p_value=p_value,
                confidence_interval=(ci_lower, ci_upper),
                is_significant=p_value < self.alpha,
                effect_size=effect_size,
                effect_size_type=EffectSizeType.COHENS_H.value,
                power=power,
                sample_size_control=control_visitors,
                sample_size_treatment=treatment_visitors,
                test_type=TestType.PROPORTION_ZTEST.value,
                alpha=self.alpha,
                recommendation=recommendation,
                interpretation=interpretation
            )
            
        except Exception as e:
            logger.error(f"Error in proportion test: {e}")
            # Return error result
            return TestResult(
                test_statistic=0.0,
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                is_significant=False,
                effect_size=0.0,
                effect_size_type="error",
                power=0.0,
                sample_size_control=control_visitors if 'control_visitors' in locals() else 0,
                sample_size_treatment=treatment_visitors if 'treatment_visitors' in locals() else 0,
                test_type="error",
                alpha=self.alpha,
                recommendation=f"Error in analysis: {str(e)}",
                interpretation="Unable to perform analysis due to error"
            )
    
    def run_ttest(self, control_data: List[float], treatment_data: List[float]) -> TestResult:
        """
        Run independent samples t-test
        
        Args:
            control_data: List of values from control group
            treatment_data: List of values from treatment group
            
        Returns:
            TestResult object
        """
        try:
            if len(control_data) == 0 or len(treatment_data) == 0:
                raise ValueError("Both groups must have data")
            
            # Calculate basic statistics
            control_mean = np.mean(control_data)
            treatment_mean = np.mean(treatment_data)
            control_std = np.std(control_data, ddof=1)
            treatment_std = np.std(treatment_data, ddof=1)
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(treatment_data, control_data, equal_var=False)
            
            # Effect size (Cohen's d)
            effect_size = self._cohens_d(control_mean, treatment_mean, control_std, treatment_std,
                                       len(control_data), len(treatment_data))
            
            # Confidence interval for difference
            diff = treatment_mean - control_mean
            pooled_se = np.sqrt(control_std**2/len(control_data) + treatment_std**2/len(treatment_data))
            df = len(control_data) + len(treatment_data) - 2
            t_critical = stats.t.ppf(1 - self.alpha/2, df)
            ci_lower = diff - t_critical * pooled_se
            ci_upper = diff + t_critical * pooled_se
            
            # Power
            power = self._power_ttest(abs(effect_size), min(len(control_data), len(treatment_data)), self.alpha)
            
            # Recommendation and interpretation
            recommendation = self._generate_recommendation(p_value, effect_size, power)
            interpretation = self._interpret_results(p_value, effect_size, power, TestType.TWO_SAMPLE_TTEST)
            
            return TestResult(
                test_statistic=t_stat,
                p_value=p_value,
                confidence_interval=(ci_lower, ci_upper),
                is_significant=p_value < self.alpha,
                effect_size=effect_size,
                effect_size_type=EffectSizeType.COHENS_D.value,
                power=power,
                sample_size_control=len(control_data),
                sample_size_treatment=len(treatment_data),
                test_type=TestType.TWO_SAMPLE_TTEST.value,
                alpha=self.alpha,
                recommendation=recommendation,
                interpretation=interpretation
            )
            
        except Exception as e:
            logger.error(f"Error in t-test: {e}")
            return TestResult(
                test_statistic=0.0, p_value=1.0, confidence_interval=(0.0, 0.0),
                is_significant=False, effect_size=0.0, effect_size_type="error",
                power=0.0, sample_size_control=0, sample_size_treatment=0,
                test_type="error", alpha=self.alpha,
                recommendation=f"Error: {str(e)}", interpretation="Analysis failed"
            )
    
    def bayesian_probability(self, control_conversions: int, control_visitors: int,
                           treatment_conversions: int, treatment_visitors: int,
                           n_samples: int = 100000) -> Dict:
        """
        Calculate Bayesian probability that treatment beats control
        
        Args:
            control_conversions: Control group conversions
            control_visitors: Control group visitors
            treatment_conversions: Treatment group conversions
            treatment_visitors: Treatment group visitors
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Dictionary with Bayesian analysis results
        """
        try:
            # Beta-binomial model with uniform priors Beta(1,1)
            alpha_control = 1 + control_conversions
            beta_control = 1 + control_visitors - control_conversions
            alpha_treatment = 1 + treatment_conversions
            beta_treatment = 1 + treatment_visitors - treatment_conversions
            
            # Monte Carlo simulation
            control_samples = np.random.beta(alpha_control, beta_control, n_samples)
            treatment_samples = np.random.beta(alpha_treatment, beta_treatment, n_samples)
            
            # Calculate probabilities
            prob_treatment_wins = np.mean(treatment_samples > control_samples)
            prob_control_wins = 1 - prob_treatment_wins
            
            # Expected lift
            lift_samples = (treatment_samples - control_samples) / control_samples
            expected_lift = np.mean(lift_samples) * 100
            lift_credible_interval = np.percentile(lift_samples * 100, [2.5, 97.5])
            
            # Risk of implementing treatment
            risk_of_loss = np.mean(treatment_samples < control_samples)
            expected_loss = np.mean(np.maximum(0, control_samples - treatment_samples)) * 100
            
            return {
                'probability_treatment_wins': round(prob_treatment_wins, 4),
                'probability_control_wins': round(prob_control_wins, 4),
                'expected_lift_percent': round(expected_lift, 2),
                'lift_credible_interval': [round(x, 2) for x in lift_credible_interval],
                'risk_of_loss': round(risk_of_loss, 4),
                'expected_loss_percent': round(expected_loss, 4),
                'certainty_level': 'High' if prob_treatment_wins > 0.95 or prob_treatment_wins < 0.05 else
                                 'Medium' if prob_treatment_wins > 0.8 or prob_treatment_wins < 0.2 else 'Low'
            }
            
        except Exception as e:
            logger.error(f"Error in Bayesian analysis: {e}")
            return {
                'error': str(e),
                'probability_treatment_wins': 0.5,
                'probability_control_wins': 0.5,
                'expected_lift_percent': 0.0
            }
    
    def sequential_testing_boundary(self, n_looks: int, alpha: float = None, 
                                  method: str = "obrien_fleming") -> List[float]:
        """
        Calculate boundaries for sequential testing
        
        Args:
            n_looks: Number of interim analyses
            alpha: Overall Type I error rate
            method: Method for boundary calculation ('obrien_fleming' or 'pocock')
            
        Returns:
            List of critical values for each look
        """
        try:
            if alpha is None:
                alpha = self.alpha
                
            boundaries = []
            
            if method == "obrien_fleming":
                # O'Brien-Fleming boundary
                for i in range(1, n_looks + 1):
                    boundary = stats.norm.ppf(1 - alpha/(2 * n_looks)) * np.sqrt(n_looks / i)
                    boundaries.append(boundary)
                    
            elif method == "pocock":
                # Pocock boundary (constant critical value)
                c = self._solve_pocock_constant(n_looks, alpha)
                boundaries = [c] * n_looks
                
            else:
                raise ValueError(f"Unknown method: {method}")
                
            return boundaries
            
        except Exception as e:
            logger.error(f"Error calculating sequential boundaries: {e}")
            return [1.96] * n_looks  # Fallback to standard critical values
    
    def _solve_pocock_constant(self, n_looks: int, alpha: float) -> float:
        """Solve for Pocock constant (simplified approximation)"""
        # Simplified approximation for Pocock boundary
        return stats.norm.ppf(1 - alpha/(2 * np.sqrt(n_looks)))
    
    def minimum_detectable_effect(self, sample_size: int, power: float = None, 
                                 alpha: float = None, baseline_rate: float = 0.1) -> float:
        """
        Calculate minimum detectable effect size
        
        Args:
            sample_size: Sample size per group
            power: Desired power
            alpha: Significance level  
            baseline_rate: Baseline conversion rate
            
        Returns:
            Minimum detectable effect size (as percentage)
        """
        try:
            if power is None:
                power = self.power
            if alpha is None:
                alpha = self.alpha
                
            z_alpha = stats.norm.ppf(1 - alpha/2)
            z_beta = stats.norm.ppf(power)
            
            # For proportion test
            p = baseline_rate
            mde = (z_alpha + z_beta) * np.sqrt(2 * p * (1 - p) / sample_size)
            
            return (mde / p) * 100  # Return as percentage
            
        except Exception as e:
            logger.error(f"Error calculating MDE: {e}")
            return 5.0  # Default 5% MDE
    
    def _generate_recommendation(self, p_value: float, effect_size: float, power: float) -> str:
        """Generate actionable recommendation based on results"""
        try:
            if p_value < self.alpha:
                if abs(effect_size) > 0.8:  # Large effect
                    return "🎉 Strong Evidence: Implement treatment immediately. Large, statistically significant effect detected."
                elif abs(effect_size) > 0.5:  # Medium effect
                    return "✅ Moderate Evidence: Implement treatment. Statistically significant effect with practical importance."
                elif abs(effect_size) > 0.2:  # Small effect
                    return "⚠️ Weak Evidence: Consider implementation. Statistically significant but small practical effect."
                else:
                    return "📊 Minimal Effect: Statistically significant but very small practical effect. Consider business context."
            else:
                if power < 0.8:
                    return "📈 Insufficient Power: Increase sample size or extend test duration for conclusive results."
                elif p_value < 0.1:
                    return "🤔 Marginally Significant: Consider extending test or investigating further before deciding."
                else:
                    return "➡️ No Clear Winner: No statistically significant difference detected. Consider practical significance."
        except Exception:
            return "❓ Unable to generate recommendation due to analysis error."
    
    def _interpret_results(self, p_value: float, effect_size: float, power: float, 
                          test_type: TestType) -> str:
        """Provide detailed interpretation of results"""
        try:
            interpretation = []
            
            # Statistical significance
            if p_value < self.alpha:
                interpretation.append(f"Result is statistically significant (p={p_value:.4f} < α={self.alpha})")
            else:
                interpretation.append(f"Result is not statistically significant (p={p_value:.4f} ≥ α={self.alpha})")
            
            # Effect size interpretation
            if test_type == TestType.PROPORTION_ZTEST:
                if abs(effect_size) < 0.2:
                    interpretation.append("Small effect size")
                elif abs(effect_size) < 0.5:
                    interpretation.append("Medium effect size")
                else:
                    interpretation.append("Large effect size")
            else:  # Cohen's d
                if abs(effect_size) < 0.2:
                    interpretation.append("Small effect size (Cohen's d < 0.2)")
                elif abs(effect_size) < 0.8:
                    interpretation.append("Medium effect size (Cohen's d = 0.2-0.8)")
                else:
                    interpretation.append("Large effect size (Cohen's d > 0.8)")
            
            # Power interpretation
            if power < 0.8:
                interpretation.append(f"Statistical power is low ({power:.2f} < 0.8) - consider increasing sample size")
            else:
                interpretation.append(f"Statistical power is adequate ({power:.2f} ≥ 0.8)")
            
            return ". ".join(interpretation) + "."
            
        except Exception:
            return "Unable to provide detailed interpretation."
    
    def generate_comprehensive_report(self, result: TestResult, test_name: str = "A/B Test") -> Dict:
        """
        Generate comprehensive test report
        
        Args:
            result: TestResult object
            test_name: Name of the test
            
        Returns:
            Comprehensive report dictionary
        """
        try:
            report = {
                "test_name": test_name,
                "test_type": result.test_type,
                "summary": {
                    "is_significant": result.is_significant,
                    "p_value": round(result.p_value, 4),
                    "effect_size": round(result.effect_size, 4),
                    "effect_size_type": result.effect_size_type,
                    "statistical_power": round(result.power, 3)
                },
                "statistical_details": {
                    "test_statistic": round(result.test_statistic, 4),
                    "confidence_interval": {
                        "lower": round(result.confidence_interval[0], 4),
                        "upper": round(result.confidence_interval[1], 4),
                        "level": f"{(1-result.alpha)*100:.0f}%"
                    },
                    "alpha": result.alpha,
                    "sample_sizes": {
                        "control": result.sample_size_control,
                        "treatment": result.sample_size_treatment,
                        "total": result.sample_size_control + result.sample_size_treatment
                    }
                },
                "interpretation": {
                    "recommendation": result.recommendation,
                    "detailed_interpretation": result.interpretation,
                    "practical_significance": self._assess_practical_significance(result.effect_size, result.effect_size_type)
                },
                "metadata": {
                    "generated_at": pd.Timestamp.now().isoformat(),
                    "engine_version": "1.0.0",
                    "significance_threshold": result.alpha,
                    "power_threshold": 0.8
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {"error": f"Failed to generate report: {str(e)}"}
    
    def _assess_practical_significance(self, effect_size: float, effect_size_type: str) -> str:
        """Assess practical significance of the effect"""
        try:
            if effect_size_type == EffectSizeType.COHENS_D.value:
                if abs(effect_size) < 0.2:
                    return "Negligible practical effect"
                elif abs(effect_size) < 0.5:
                    return "Small practical effect"
                elif abs(effect_size) < 0.8:
                    return "Medium practical effect"
                else:
                    return "Large practical effect"
            elif effect_size_type == EffectSizeType.COHENS_H.value:
                if abs(effect_size) < 0.2:
                    return "Negligible practical difference"
                elif abs(effect_size) < 0.5:
                    return "Small practical difference"
                else:
                    return "Large practical difference"
            else:
                return "Practical significance assessment not available"
        except Exception:
            return "Unable to assess practical significance"