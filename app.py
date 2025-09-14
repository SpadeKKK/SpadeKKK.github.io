#!/usr/bin/env python3
"""
A/B Test Designer Robot - Backend API
Intelligent statistical analysis & experiment design platform
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from flask import Flask, request, jsonify, render_template_string, send_from_directory
from flask_cors import CORS
import numpy as np
from scipy import stats
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production'),
    DEBUG=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true',
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max file upload
)

class ABTestDesigner:
    """Main class for A/B test design and analysis"""
    
    def __init__(self, alpha: float = 0.05, beta: float = 0.2):
        self.alpha = alpha  # Significance level
        self.beta = beta    # Type II error rate
        self.power = 1 - beta  # Statistical power
        
    def calculate_sample_size(self, effect_size: float, baseline_rate: float = 0.1, 
                            test_type: str = "proportion") -> int:
        """Calculate required sample size for A/B test"""
        try:
            z_alpha = stats.norm.ppf(1 - self.alpha/2)  # 1.96 for 95% confidence
            z_beta = stats.norm.ppf(1 - self.beta)      # 0.84 for 80% power
            
            if test_type == "proportion":
                p1 = baseline_rate
                p2 = p1 * (1 + effect_size / 100)
                p_pooled = (p1 + p2) / 2
                
                n = (2 * p_pooled * (1 - p_pooled) * (z_alpha + z_beta)**2) / (p2 - p1)**2
                return max(100, int(np.ceil(n)))  # Minimum 100 users per group
            
            elif test_type == "continuous":
                # For continuous variables (t-test)
                n = (2 * (z_alpha + z_beta)**2) / (effect_size/100)**2
                return max(50, int(np.ceil(n)))
            
            else:
                return 1000  # Default fallback
                
        except Exception as e:
            logger.error(f"Error calculating sample size: {e}")
            return 1000
    
    def calculate_test_duration(self, sample_size: int, daily_traffic: int = 500) -> int:
        """Calculate test duration based on sample size and daily traffic"""
        try:
            total_users_needed = sample_size * 2  # Control + Treatment
            days = max(7, np.ceil(total_users_needed / daily_traffic))  # Minimum 1 week
            return int(days)
        except Exception as e:
            logger.error(f"Error calculating test duration: {e}")
            return 14
    
    def perform_statistical_test(self, control_conversions: int, control_visitors: int, 
                                treatment_conversions: int, treatment_visitors: int) -> Dict:
        """Perform statistical significance test"""
        try:
            if any(x <= 0 for x in [control_visitors, treatment_visitors]):
                raise ValueError("Visitor counts must be positive")
            
            if control_conversions > control_visitors or treatment_conversions > treatment_visitors:
                raise ValueError("Conversions cannot exceed visitors")
            
            # Two-proportion z-test
            p1 = control_conversions / control_visitors
            p2 = treatment_conversions / treatment_visitors
            
            p_pooled = (control_conversions + treatment_conversions) / (control_visitors + treatment_visitors)
            se = np.sqrt(p_pooled * (1 - p_pooled) * (1/control_visitors + 1/treatment_visitors))
            
            z_score = (p2 - p1) / se if se > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score))) if se > 0 else 1.0
            
            # Effect size and confidence interval
            effect_size = ((p2 - p1) / p1 * 100) if p1 > 0 else 0
            confidence_interval = self.calculate_confidence_interval(p1, p2, control_visitors, treatment_visitors)
            
            # Statistical power (post-hoc)
            observed_effect = abs(p2 - p1)
            power = self.calculate_power(observed_effect, min(control_visitors, treatment_visitors))
            
            return {
                'control_rate': round(p1, 4),
                'treatment_rate': round(p2, 4),
                'z_score': round(z_score, 4),
                'p_value': round(p_value, 4),
                'effect_size_percent': round(effect_size, 2),
                'is_significant': p_value < self.alpha,
                'confidence_interval': confidence_interval,
                'statistical_power': round(power, 3),
                'sample_size_control': control_visitors,
                'sample_size_treatment': treatment_visitors
            }
            
        except Exception as e:
            logger.error(f"Error in statistical test: {e}")
            return {
                'error': str(e),
                'control_rate': 0,
                'treatment_rate': 0,
                'p_value': 1.0,
                'is_significant': False
            }
    
    def calculate_confidence_interval(self, p1: float, p2: float, n1: int, n2: int, 
                                    confidence: float = 0.95) -> Dict:
        """Calculate confidence interval for difference in proportions"""
        try:
            diff = p2 - p1
            se = np.sqrt((p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2))
            z = stats.norm.ppf(1 - (1 - confidence) / 2)
            
            lower = diff - z * se
            upper = diff + z * se
            
            return {
                'lower': round(lower, 4),
                'upper': round(upper, 4),
                'difference': round(diff, 4),
                'confidence_level': confidence
            }
        except Exception as e:
            logger.error(f"Error calculating confidence interval: {e}")
            return {'lower': 0, 'upper': 0, 'difference': 0, 'confidence_level': confidence}
    
    def calculate_power(self, effect_size: float, sample_size: int) -> float:
        """Calculate statistical power for given effect size and sample size"""
        try:
            z_alpha = stats.norm.ppf(1 - self.alpha/2)
            z_beta = (effect_size * np.sqrt(sample_size/2)) - z_alpha
            power = stats.norm.cdf(z_beta)
            return max(0, min(1, power))
        except Exception as e:
            logger.error(f"Error calculating power: {e}")
            return 0.8  # Default assumption
    
    def generate_workflow_steps(self, question: str, hypothesis: str, metric: str, 
                              effect_size: float, baseline_rate: float = 10) -> Dict:
        """Generate 5-step experiment design workflow"""
        try:
            sample_size = self.calculate_sample_size(effect_size, baseline_rate/100)
            duration = self.calculate_test_duration(sample_size)
            
            return {
                'step1': {
                    'title': 'Define Objective',
                    'description': f'Research Question: {question}',
                    'details': f'Primary metric: {metric.replace("-", " ").title()}'
                },
                'step2': {
                    'title': 'Hypothesis Formation',
                    'description': hypothesis,
                    'details': f'Expected effect size: {effect_size}% improvement'
                },
                'step3': {
                    'title': 'Sample Size Calculation',
                    'description': f'Required sample size: {sample_size:,} users per group',
                    'details': f'Total users needed: {sample_size * 2:,} (α={self.alpha}, Power={self.power*100:.0f}%)'
                },
                'step4': {
                    'title': 'Randomization Strategy',
                    'description': '50/50 split between control and treatment groups',
                    'details': 'Stratified randomization to ensure balanced assignment'
                },
                'step5': {
                    'title': 'Statistical Analysis Plan',
                    'description': f'Two-tailed test with α={self.alpha}, Power={self.power*100:.0f}%',
                    'details': f'Recommended test duration: {duration} days'
                }
            }
        except Exception as e:
            logger.error(f"Error generating workflow: {e}")
            return {'error': f'Failed to generate workflow: {str(e)}'}
    
    def generate_recommendations(self, results: Dict) -> str:
        """Generate actionable recommendations based on test results"""
        try:
            if results.get('error'):
                return "Unable to generate recommendations due to data issues."
            
            p_value = results.get('p_value', 1.0)
            effect_size = results.get('effect_size_percent', 0)
            is_significant = results.get('is_significant', False)
            power = results.get('statistical_power', 0)
            
            if is_significant:
                if effect_size > 0:
                    if abs(effect_size) > 20:
                        return "🎉 Strong Evidence: Implement treatment immediately. Large, statistically significant improvement detected."
                    elif abs(effect_size) > 10:
                        return "✅ Moderate Evidence: Implement treatment. Statistically significant improvement with practical impact."
                    else:
                        return "⚠️ Weak Evidence: Consider implementation. Statistically significant but small practical effect."
                else:
                    return "❌ Treatment Underperforms: Keep control. Treatment shows statistically significant decrease."
            else:
                if power < 0.8:
                    return "📊 Insufficient Power: Extend test duration or increase sample size for conclusive results."
                elif p_value < 0.1:
                    return "🤔 Marginally Significant: Consider extending test or investigating further before deciding."
                else:
                    return "➡️ No Clear Winner: No statistically significant difference detected. Consider other factors."
                    
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return "Unable to generate recommendations."

# Initialize the AB test designer
ab_designer = ABTestDesigner()

@app.route('/')
def home():
    """Serve the main application page"""
    try:
        # Try to serve index.html if it exists
        if os.path.exists('index.html'):
            with open('index.html', 'r', encoding='utf-8') as f:
                return render_template_string(f.read())
        else:
            # Return API information if no frontend file
            return jsonify({
                'name': 'A/B Test Designer Robot API',
                'version': '1.0.0',
                'description': 'Intelligent statistical analysis & experiment design platform',
                'endpoints': {
                    'GET /': 'This page',
                    'POST /api/design': 'Design A/B test experiment',
                    'POST /api/analyze': 'Analyze A/B test results',
                    'GET /api/data-template': 'Get sample data template',
                    'GET /api/health': 'Health check endpoint'
                },
                'documentation': 'https://github.com/spadekkk/ab-test-designer'
            })
    except Exception as e:
        logger.error(f"Error serving home page: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/design', methods=['POST'])
def design_experiment():
    """Design an A/B test experiment"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Validate required fields
        required_fields = ['question', 'hypothesis', 'metric', 'effect_size']
        missing_fields = [field for field in required_fields if not data.get(field)]
        
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        question = str(data.get('question', '')).strip()
        hypothesis = str(data.get('hypothesis', '')).strip()
        metric = str(data.get('metric', 'conversion-rate')).strip()
        effect_size = float(data.get('effect_size', 15))
        baseline_rate = float(data.get('baseline_rate', 10))
        
        # Validate effect size
        if effect_size <= 0 or effect_size > 1000:
            return jsonify({'error': 'Effect size must be between 0 and 1000 percent'}), 400
        
        if baseline_rate <= 0 or baseline_rate > 100:
            return jsonify({'error': 'Baseline rate must be between 0 and 100 percent'}), 400
        
        # Generate workflow
        workflow = ab_designer.generate_workflow_steps(
            question, hypothesis, metric, effect_size, baseline_rate
        )
        
        if 'error' in workflow:
            return jsonify({'error': workflow['error']}), 500
        
        # Calculate sample size and duration
        sample_size = ab_designer.calculate_sample_size(effect_size, baseline_rate/100)
        duration = ab_designer.calculate_test_duration(sample_size)
        
        # Calculate minimum detectable effect
        min_detectable_effect = (2 * np.sqrt(2 * (baseline_rate/100) * (1 - baseline_rate/100) / sample_size)) * 100
        
        response = {
            'success': True,
            'workflow': workflow,
            'sample_size_per_group': sample_size,
            'total_sample_size': sample_size * 2,
            'duration_days': duration,
            'minimum_detectable_effect': round(min_detectable_effect, 2),
            'statistical_parameters': {
                'alpha': ab_designer.alpha,
                'power': ab_designer.power,
                'baseline_rate': baseline_rate
            },
            'generated_at': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Experiment designed: {sample_size} users per group, {duration} days")
        return jsonify(response)
        
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Error in design_experiment: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_results():
    """Analyze A/B test results"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Extract and validate data
        control_conversions = int(data.get('control_conversions', 0))
        control_visitors = int(data.get('control_visitors', 0))
        treatment_conversions = int(data.get('treatment_conversions', 0))
        treatment_visitors = int(data.get('treatment_visitors', 0))
        
        # Validate inputs
        if any(x < 0 for x in [control_conversions, control_visitors, treatment_conversions, treatment_visitors]):
            return jsonify({'error': 'All values must be non-negative'}), 400
        
        if control_visitors == 0 or treatment_visitors == 0:
            return jsonify({'error': 'Visitor counts must be greater than zero'}), 400
        
        if control_conversions > control_visitors or treatment_conversions > treatment_visitors:
            return jsonify({'error': 'Conversions cannot exceed visitors'}), 400
        
        # Perform statistical analysis
        results = ab_designer.perform_statistical_test(
            control_conversions, control_visitors,
            treatment_conversions, treatment_visitors
        )
        
        if 'error' in results:
            return jsonify({'error': results['error']}), 400
        
        # Generate recommendations
        recommendation = ab_designer.generate_recommendations(results)
        
        # Calculate additional metrics
        control_rate = results['control_rate']
        treatment_rate = results['treatment_rate']
        
        response = {
            'success': True,
            'results': results,
            'recommendation': recommendation,
            'summary': {
                'control_conversion_rate': f"{control_rate:.2%}",
                'treatment_conversion_rate': f"{treatment_rate:.2%}",
                'relative_improvement': f"{results['effect_size_percent']:.1f}%",
                'significance': 'Significant' if results['is_significant'] else 'Not Significant',
                'confidence_level': '95%'
            },
            'analyzed_at': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Analysis completed: p={results['p_value']:.4f}, significant={results['is_significant']}")
        return jsonify(response)
        
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Error in analyze_results: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/data-template', methods=['GET'])
def get_data_template():
    """Get sample data template for A/B testing"""
    try:
        experiment_type = request.args.get('type', 'button_color')
        sample_size = min(50, max(5, int(request.args.get('sample_size', 20))))
        
        # Generate sample users data
        users_data = []
        for i in range(1, sample_size + 1):
            users_data.append({
                'user_id': f'U{str(i).zfill(6)}',
                'username': f'user_{i}@example.com'
            })
        
        # Generate interactions data with random assignment
        interactions_data = []
        ads_variants = ['AD_CONTROL', 'AD_TREATMENT']
        
        np.random.seed(42)  # For reproducible results
        for user in users_data:
            ads_id = np.random.choice(ads_variants)
            interactions_data.append({
                'user_id': user['user_id'],
                'ads_id': ads_id
            })
        
        # Generate ads data based on experiment type
        experiment_configs = {
            'button_color': [
                {'ads_id': 'AD_CONTROL', 'ads_name': 'Control - Blue Button'},
                {'ads_id': 'AD_TREATMENT', 'ads_name': 'Treatment - Red Button'}
            ],
            'headline': [
                {'ads_id': 'AD_CONTROL', 'ads_name': 'Control - Original Headline'},
                {'ads_id': 'AD_TREATMENT', 'ads_name': 'Treatment - Optimized Headline'}
            ],
            'pricing': [
                {'ads_id': 'AD_CONTROL', 'ads_name': 'Control - Standard Pricing'},
                {'ads_id': 'AD_TREATMENT', 'ads_name': 'Treatment - Discount Pricing'}
            ],
            'layout': [
                {'ads_id': 'AD_CONTROL', 'ads_name': 'Control - Current Layout'},
                {'ads_id': 'AD_TREATMENT', 'ads_name': 'Treatment - New Layout'}
            ]
        }
        
        ads_data = experiment_configs.get(experiment_type, experiment_configs['button_color'])
        
        # Calculate distribution statistics
        control_count = sum(1 for interaction in interactions_data if interaction['ads_id'] == 'AD_CONTROL')
        treatment_count = len(interactions_data) - control_count
        
        response = {
            'success': True,
            'experiment_type': experiment_type,
            'table1_users': users_data,
            'table2_interactions': interactions_data,
            'table3_ads': ads_data,
            'metadata': {
                'total_users': len(users_data),
                'control_group_size': control_count,
                'treatment_group_size': treatment_count,
                'split_ratio': f"{control_count}:{treatment_count}",
                'generated_at': datetime.utcnow().isoformat()
            },
            'schema': {
                'table1_users': {
                    'user_id': 'string - Unique user identifier',
                    'username': 'string - User email or username'
                },
                'table2_interactions': {
                    'user_id': 'string - Foreign key to users table',
                    'ads_id': 'string - Foreign key to ads table'
                },
                'table3_ads': {
                    'ads_id': 'string - Unique ad variant identifier',
                    'ads_name': 'string - Descriptive name of the ad variant'
                }
            }
        }
        
        logger.info(f"Data template generated: {sample_size} users, {experiment_type} experiment")
        return jsonify(response)
        
    except ValueError as e:
        return jsonify({'error': f'Invalid parameter: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Error generating data template: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': ['/api/design', '/api/analyze', '/api/data-template', '/api/health']
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle method not allowed errors"""
    return jsonify({'error': 'Method not allowed for this endpoint'}), 405

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

# Static files serving (for development)
@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

if __name__ == '__main__':
    # Get configuration from environment
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting A/B Test Designer API on {host}:{port}")
    logger.info(f"Debug mode: {debug}")
    
    app.run(host=host, port=port, debug=debug)