#!/usr/bin/env python3
"""
Comprehensive Test Suite for A/B Test Designer Robot
Tests all components: API, Statistical Engine, Data Generator
"""

import unittest
import json
import sys
import os
import tempfile
import logging
from unittest.mock import patch, MagicMock
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
try:
    from app import app, ab_designer
    from statistical_engine import StatisticalEngine, TestResult, TestType, EffectSizeType
    from data_generator import DataGenerator
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Configure test logging
logging.basicConfig(level=logging.WARNING)

class TestABTestDesignerAPI(unittest.TestCase):
    """Test the Flask API endpoints"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.app = app.test_client()
        self.app.testing = True
        
    def test_home_endpoint(self):
        """Test the home endpoint returns correct information"""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        
        # Check if it returns HTML (frontend) or JSON (API info)
        content_type = response.headers.get('Content-Type', '')
        self.assertTrue('text/html' in content_type or 'application/json' in content_type)
        
    def test_health_endpoint(self):
        """Test the health check endpoint"""
        response = self.app.get('/api/health')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('timestamp', data)
        self.assertIn('version', data)
        
    def test_design_endpoint_valid_data(self):
        """Test the experiment design endpoint with valid data"""
        payload = {
            'question': 'Does red button increase clicks?',
            'hypothesis': 'Red button will increase CTR by 15%',
            'metric': 'click-through-rate',
            'effect_size': 15,
            'baseline_rate': 10
        }
        
        response = self.app.post('/api/design', 
                               data=json.dumps(payload),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Check response structure
        self.assertTrue(data['success'])
        self.assertIn('workflow', data)
        self.assertIn('sample_size_per_group', data)
        self.assertIn('duration_days', data)
        self.assertIn('statistical_parameters', data)
        
        # Check workflow has all 5 steps
        workflow = data['workflow']
        for i in range(1, 6):
            self.assertIn(f'step{i}', workflow)
            self.assertIn('title', workflow[f'step{i}'])
            self.assertIn('description', workflow[f'step{i}'])
            
        # Verify sample size is reasonable
        self.assertGreater(data['sample_size_per_group'], 0)
        self.assertLess(data['sample_size_per_group'], 100000)  # Sanity check
        self.assertGreater(data['duration_days'], 0)
        
    def test_design_endpoint_missing_fields(self):
        """Test design endpoint with missing required fields"""
        payload = {
            'question': 'Test question',
            # Missing required fields
        }
        
        response = self.app.post('/api/design',
                               data=json.dumps(payload),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertIn('Missing required fields', data['error'])
        
    def test_design_endpoint_invalid_effect_size(self):
        """Test design endpoint with invalid effect size"""
        payload = {
            'question': 'Test question',
            'hypothesis': 'Test hypothesis',
            'metric': 'conversion-rate',
            'effect_size': -5  # Invalid negative value
        }
        
        response = self.app.post('/api/design',
                               data=json.dumps(payload),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        
    def test_analyze_endpoint_valid_data(self):
        """Test the results analysis endpoint with valid data"""
        payload = {
            'control_conversions': 100,
            'control_visitors': 1000,
            'treatment_conversions': 120,
            'treatment_visitors': 1000
        }
        
        response = self.app.post('/api/analyze',
                               data=json.dumps(payload),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Check response structure
        self.assertTrue(data['success'])
        self.assertIn('results', data)
        self.assertIn('recommendation', data)
        self.assertIn('summary', data)
        
        # Check results structure
        results = data['results']
        required_fields = ['control_rate', 'treatment_rate', 'p_value', 
                         'is_significant', 'confidence_interval']
        for field in required_fields:
            self.assertIn(field, results)
            
        # Verify statistical calculations
        self.assertGreaterEqual(results['p_value'], 0)
        self.assertLessEqual(results['p_value'], 1)
        self.assertIsInstance(results['is_significant'], bool)
        
        # Check confidence interval structure
        ci = results['confidence_interval']
        self.assertIn('lower', ci)
        self.assertIn('upper', ci)
        self.assertLessEqual(ci['lower'], ci['upper'])
        
    def test_analyze_endpoint_invalid_data(self):
        """Test analyze endpoint with invalid data"""
        # Test with negative values
        payload = {
            'control_conversions': -10,
            'control_visitors': 1000,
            'treatment_conversions': 120,
            'treatment_visitors': 1000
        }
        
        response = self.app.post('/api/analyze',
                               data=json.dumps(payload),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        
        # Test with conversions > visitors
        payload = {
            'control_conversions': 1200,
            'control_visitors': 1000,
            'treatment_conversions': 120,
            'treatment_visitors': 1000
        }
        
        response = self.app.post('/api/analyze',
                               data=json.dumps(payload),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        
    def test_data_template_endpoint(self):
        """Test the data template generation endpoint"""
        response = self.app.get('/api/data-template')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        
        # Check all required tables are present
        self.assertIn('table1_users', data)
        self.assertIn('table2_interactions', data)
        self.assertIn('table3_ads', data)
        self.assertIn('metadata', data)
        
        # Verify data structure
        users = data['table1_users']
        self.assertGreater(len(users), 0)
        self.assertIn('user_id', users[0])
        self.assertIn('username', users[0])
        
        ads = data['table3_ads']
        self.assertEqual(len(ads), 2)  # Control and treatment
        self.assertIn('ads_id', ads[0])
        self.assertIn('ads_name', ads[0])
        
        interactions = data['table2_interactions']
        self.assertEqual(len(interactions), len(users))
        
        # Check metadata
        metadata = data['metadata']
        self.assertIn('total_users', metadata)
        self.assertIn('generated_at', metadata)
        
    def test_data_template_with_parameters(self):
        """Test data template with query parameters"""
        response = self.app.get('/api/data-template?type=pricing&sample_size=10')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertEqual(data['experiment_type'], 'pricing')
        self.assertLessEqual(len(data['table1_users']), 10)
        
    def test_404_endpoint(self):
        """Test 404 handling for non-existent endpoints"""
        response = self.app.get('/api/nonexistent')
        self.assertEqual(response.status_code, 404)
        
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertIn('available_endpoints', data)
        
    def test_method_not_allowed(self):
        """Test 405 handling for wrong HTTP methods"""
        response = self.app.get('/api/design')  # Should be POST
        self.assertEqual(response.status_code, 405)


class TestStatisticalEngine(unittest.TestCase):
    """Test the statistical analysis engine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = StatisticalEngine(alpha=0.05, beta=0.2)
        
    def test_initialization(self):
        """Test engine initialization with valid parameters"""
        engine = StatisticalEngine(alpha=0.05, beta=0.2)
        self.assertEqual(engine.alpha, 0.05)
        self.assertEqual(engine.beta, 0.2)
        self.assertEqual(engine.power, 0.8)
        
    def test_initialization_invalid_parameters(self):
        """Test engine initialization with invalid parameters"""
        with self.assertRaises(ValueError):
            StatisticalEngine(alpha=1.5, beta=0.2)  # Alpha > 1
            
        with self.assertRaises(ValueError):
            StatisticalEngine(alpha=0.05, beta=-0.1)  # Negative beta
            
    def test_sample_size_calculation(self):
        """Test sample size calculation for different effect sizes"""
        # Test with normal effect size
        sample_size = self.engine.sample_size_calculator(effect_size=15)
        self.assertGreater(sample_size, 0)
        self.assertIsInstance(sample_size, int)
        
        # Larger effect sizes should require smaller samples
        large_effect = self.engine.sample_size_calculator(effect_size=50)
        small_effect = self.engine.sample_size_calculator(effect_size=5)
        self.assertLess(large_effect, small_effect)
        
        # Test edge cases
        zero_effect = self.engine.sample_size_calculator(effect_size=0)
        self.assertGreater(zero_effect, 0)  # Should return default
        
    def test_proportion_test_valid_data(self):
        """Test two-proportion z-test with valid data"""
        result = self.engine.run_proportion_test(
            control_conversions=100,
            control_visitors=1000,
            treatment_conversions=120,
            treatment_visitors=1000
        )
        
        # Check result type and structure
        self.assertIsInstance(result, TestResult)
        self.assertGreater(result.p_value, 0)
        self.assertLess(result.p_value, 1)
        self.assertIsInstance(result.is_significant, bool)
        self.assertIsInstance(result.recommendation, str)
        self.assertIsInstance(result.interpretation, str)
        
        # Check confidence interval
        ci = result.confidence_interval
        self.assertLess(ci[0], ci[1])  # Lower < Upper
        
        # Check effect size
        self.assertIsInstance(result.effect_size, float)
        self.assertEqual(result.effect_size_type, EffectSizeType.COHENS_H.value)
        
    def test_proportion_test_invalid_data(self):
        """Test proportion test with invalid data"""
        # Test with negative values
        result = self.engine.run_proportion_test(
            control_conversions=-10,
            control_visitors=1000,
            treatment_conversions=120,
            treatment_visitors=1000
        )
        
        self.assertEqual(result.test_type, "error")
        self.assertIn("Error", result.recommendation)
        
        # Test with zero visitors
        result = self.engine.run_proportion_test(
            control_conversions=10,
            control_visitors=0,
            treatment_conversions=120,
            treatment_visitors=1000
        )
        
        self.assertEqual(result.test_type, "error")
        
    def test_ttest_valid_data(self):
        """Test independent t-test with valid data"""
        import numpy as np
        
        # Generate test data
        np.random.seed(42)
        control_data = list(np.random.normal(100, 15, 100))
        treatment_data = list(np.random.normal(105, 15, 100))
        
        result = self.engine.run_ttest(control_data, treatment_data)
        
        self.assertIsInstance(result, TestResult)
        self.assertEqual(result.test_type, TestType.TWO_SAMPLE_TTEST.value)
        self.assertEqual(result.effect_size_type, EffectSizeType.COHENS_D.value)
        self.assertGreater(result.sample_size_control, 0)
        self.assertGreater(result.sample_size_treatment, 0)
        
    def test_ttest_empty_data(self):
        """Test t-test with empty data"""
        result = self.engine.run_ttest([], [1, 2, 3])
        self.assertEqual(result.test_type, "error")
        
        result = self.engine.run_ttest([1, 2, 3], [])
        self.assertEqual(result.test_type, "error")
        
    def test_power_analysis(self):
        """Test statistical power analysis"""
        power = self.engine.power_analysis(
            effect_size=0.5, 
            sample_size=100, 
            test_type=TestType.PROPORTION_ZTEST
        )
        
        self.assertGreaterEqual(power, 0)
        self.assertLessEqual(power, 1)
        self.assertIsInstance(power, float)
        
    def test_bayesian_probability(self):
        """Test Bayesian probability calculation"""
        result = self.engine.bayesian_probability(
            control_conversions=100,
            control_visitors=1000,
            treatment_conversions=120,
            treatment_visitors=1000
        )
        
        required_keys = ['probability_treatment_wins', 'probability_control_wins', 
                        'expected_lift_percent', 'risk_of_loss']
        for key in required_keys:
            self.assertIn(key, result)
            
        # Probabilities should sum to 1
        self.assertAlmostEqual(
            result['probability_treatment_wins'] + result['probability_control_wins'],
            1.0, places=3
        )
        
    def test_sequential_testing_boundary(self):
        """Test sequential testing boundary calculation"""
        boundaries = self.engine.sequential_testing_boundary(n_looks=5)
        
        self.assertEqual(len(boundaries), 5)
        self.assertTrue(all(isinstance(b, float) for b in boundaries))
        self.assertTrue(all(b > 0 for b in boundaries))
        
        # Test Pocock method
        pocock_boundaries = self.engine.sequential_testing_boundary(
            n_looks=3, method="pocock"
        )
        self.assertEqual(len(pocock_boundaries), 3)
        
    def test_minimum_detectable_effect(self):
        """Test minimum detectable effect calculation"""
        mde = self.engine.minimum_detectable_effect(sample_size=1000)
        
        self.assertGreater(mde, 0)
        self.assertIsInstance(mde, float)
        
        # Larger sample sizes should have smaller MDE
        mde_large = self.engine.minimum_detectable_effect(sample_size=5000)
        mde_small = self.engine.minimum_detectable_effect(sample_size=500)
        self.assertLess(mde_large, mde_small)
        
    def test_comprehensive_report_generation(self):
        """Test comprehensive report generation"""
        result = self.engine.run_proportion_test(100, 1000, 120, 1000)
        report = self.engine.generate_comprehensive_report(result, "Test Report")
        
        required_sections = ['test_name', 'summary', 'statistical_details', 
                           'interpretation', 'metadata']
        for section in required_sections:
            self.assertIn(section, report)
            
        # Check summary structure
        summary = report['summary']
        self.assertIn('is_significant', summary)
        self.assertIn('p_value', summary)
        self.assertIn('effect_size', summary)


class TestDataGenerator(unittest.TestCase):
    """Test the data generation functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.generator = DataGenerator(seed=42)  # Fixed seed for reproducibility
        
    def test_initialization(self):
        """Test generator initialization"""
        generator = DataGenerator(seed=123)
        self.assertIsInstance(generator.first_names, list)
        self.assertIsInstance(generator.last_names, list)
        self.assertIsInstance(generator.email_domains, list)
        self.assertGreater(len(generator.first_names), 0)
        
    def test_user_id_generation(self):
        """Test user ID generation in different formats"""
        # Standard format
        user_id = self.generator.generate_user_id(1, "standard")
        self.assertEqual(user_id, "U000001")
        
        # UUID format
        uuid_id = self.generator.generate_user_id(1, "uuid")
        self.assertEqual(len(uuid_id), 36)  # Standard UUID length
        self.assertIn('-', uuid_id)
        
        # Hash format
        hash_id = self.generator.generate_user_id(1, "hash")
        self.assertTrue(hash_id.startswith("user_"))
        
    def test_username_generation(self):
        """Test username generation in different formats"""
        # Email format
        email = self.generator.generate_username(1, "email")
        self.assertIn('@', email)
        self.assertTrue(any(domain in email for domain in self.generator.email_domains))
        
        # Username format
        username = self.generator.generate_username(1, "username")
        self.assertNotIn('@', username)
        
    def test_users_table_generation(self):
        """Test users table generation"""
        users_df = self.generator.generate_users_table(n_users=100)
        
        # Check structure
        self.assertEqual(len(users_df), 100)
        self.assertIn('user_id', users_df.columns)
        self.assertIn('username', users_df.columns)
        
        # Check for unique user IDs
        self.assertEqual(len(users_df['user_id'].unique()), len(users_df))
        
        # Check data types
        self.assertTrue(all(isinstance(uid, str) for uid in users_df['user_id']))
        self.assertTrue(all(isinstance(uname, str) for uname in users_df['username']))
        
    def test_ads_table_generation(self):
        """Test ads table generation"""
        # Test default experiment type
        ads_df = self.generator.generate_ads_table()
        self.assertEqual(len(ads_df), 2)  # Control and treatment
        self.assertIn('ads_id', ads_df.columns)
        self.assertIn('ads_name', ads_df.columns)
        
        # Test specific experiment type
        ads_df_pricing = self.generator.generate_ads_table('pricing')
        self.assertTrue(any('Pricing' in name for name in ads_df_pricing['ads_name']))
        
        # Test custom variants
        custom_variants = {
            'control': 'Custom Control',
            'treatment': 'Custom Treatment'
        }
        ads_df_custom = self.generator.generate_ads_table(custom_variants=custom_variants)
        self.assertIn('Custom Control', ads_df_custom['ads_name'].values)
        self.assertIn('Custom Treatment', ads_df_custom['ads_name'].values)
        
    def test_interactions_table_generation(self):
        """Test interactions table generation"""
        users_df = self.generator.generate_users_table(n_users=100)
        ads_df = self.generator.generate_ads_table()
        interactions_df = self.generator.generate_interactions_table(users_df, ads_df)
        
        # Check structure
        self.assertEqual(len(interactions_df), 100)
        self.assertIn('user_id', interactions_df.columns)
        self.assertIn('ads_id', interactions_df.columns)
        
        # Check that all ads_ids are valid
        valid_ads = set(ads_df['ads_id'])
        interaction_ads = set(interactions_df['ads_id'])
        self.assertTrue(interaction_ads.issubset(valid_ads))
        
        # Check that all user_ids are valid
        valid_users = set(users_df['user_id'])
        interaction_users = set(interactions_df['user_id'])
        self.assertTrue(interaction_users.issubset(valid_users))
        
    def test_conversion_data_generation(self):
        """Test conversion data generation"""
        users_df = self.generator.generate_users_table(n_users=50)
        ads_df = self.generator.generate_ads_table()
        interactions_df = self.generator.generate_interactions_table(users_df, ads_df)
        
        conversions_df = self.generator.generate_conversion_data(
            interactions_df, 
            control_rate=0.1, 
            treatment_rate=0.15
        )
        
        # Check structure
        self.assertEqual(len(conversions_df), 50)
        required_columns = ['user_id', 'ads_id', 'converted', 'conversion_value', 'timestamp']
        for col in required_columns:
            self.assertIn(col, conversions_df.columns)
            
        # Check data types
        self.assertTrue(all(isinstance(conv, bool) for conv in conversions_df['converted']))
        self.assertTrue(all(isinstance(val, float) for val in conversions_df['conversion_value']))
        
        # Check conversion logic
        converted_users = conversions_df[conversions_df['converted']]
        non_converted_users = conversions_df[~conversions_df['converted']]
        
        # Converted users should have positive conversion values
        self.assertTrue(all(val > 0 for val in converted_users['conversion_value']))
        # Non-converted users should have zero conversion values
        self.assertTrue(all(val == 0 for val in non_converted_users['conversion_value']))
        
    def test_complete_dataset_generation(self):
        """Test complete dataset generation"""
        datasets = self.generator.generate_complete_dataset(
            n_users=50,
            experiment_type="button_color",
            include_engagement=True
        )
        
        # Check all required datasets are present
        expected_datasets = ['users', 'ads', 'interactions', 'conversions', 'engagement']
        for dataset_name in expected_datasets:
            self.assertIn(dataset_name, datasets)
            self.assertGreater(len(datasets[dataset_name]), 0)
            
        # Check referential integrity
        user_ids = set(datasets['users']['user_id'])
        interaction_user_ids = set(datasets['interactions']['user_id'])
        conversion_user_ids = set(datasets['conversions']['user_id'])
        
        self.assertTrue(interaction_user_ids.issubset(user_ids))
        self.assertTrue(conversion_user_ids.issubset(user_ids))
        
    def test_api_sample_generation(self):
        """Test API sample data generation"""
        sample = self.generator.generate_api_sample(n_users=10)
        
        required_keys = ['table1_users', 'table2_interactions', 'table3_ads', 'metadata']
        for key in required_keys:
            self.assertIn(key, sample)
            
        # Check data structure
        self.assertIsInstance(sample['table1_users'], list)
        self.assertIsInstance(sample['table2_interactions'], list)
        self.assertIsInstance(sample['table3_ads'], list)
        
        # Check metadata
        metadata = sample['metadata']
        self.assertIn('generated_at', metadata)
        self.assertIn('total_users', metadata)
        self.assertEqual(metadata['total_users'], 10)
        
    def test_export_functionality(self):
        """Test CSV export functionality"""
        datasets = self.generator.generate_complete_dataset(n_users=20)
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                self.generator.export_to_csv(datasets, temp_dir)
                
                # Check that files were created
                expected_files = ['users_data.csv', 'ads_data.csv', 
                                'interactions_data.csv', 'conversions_data.csv']
                for filename in expected_files:
                    filepath = os.path.join(temp_dir, filename)
                    self.assertTrue(os.path.exists(filepath))
                    self.assertGreater(os.path.getsize(filepath), 0)
                    
            except Exception as e:
                self.fail(f"Export failed: {e}")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.app = app.test_client()
        self.app.testing = True
        self.generator = DataGenerator(seed=42)
        
    def test_full_workflow_integration(self):
        """Test the complete A/B test workflow from design to analysis"""
        # Step 1: Design experiment
        design_payload = {
            'question': 'Integration test: Does new design increase conversions?',
            'hypothesis': 'New design will improve conversions by 20%',
            'metric': 'conversion-rate',
            'effect_size': 20,
            'baseline_rate': 8
        }
        
        design_response = self.app.post('/api/design',
                                      data=json.dumps(design_payload),
                                      content_type='application/json')
        
        self.assertEqual(design_response.status_code, 200)
        design_data = json.loads(design_response.data)
        self.assertTrue(design_data['success'])
        
        # Extract sample size for analysis
        sample_size = design_data['sample_size_per_group']
        
        # Step 2: Get data template
        template_response = self.app.get('/api/data-template?sample_size=20')
        self.assertEqual(template_response.status_code, 200)
        template_data = json.loads(template_response.data)
        self.assertTrue(template_data['success'])
        
        # Step 3: Simulate data collection and analyze results
        analysis_payload = {
            'control_conversions': int(sample_size * 0.08),    # 8% baseline rate
            'control_visitors': sample_size,
            'treatment_conversions': int(sample_size * 0.096), # 20% improvement
            'treatment_visitors': sample_size
        }
        
        analysis_response = self.app.post('/api/analyze',
                                        data=json.dumps(analysis_payload),
                                        content_type='application/json')
        
        self.assertEqual(analysis_response.status_code, 200)
        analysis_data = json.loads(analysis_response.data)
        self.assertTrue(analysis_data['success'])
        
        # Verify the complete workflow worked
        self.assertIn('workflow', design_data)
        self.assertIn('table1_users', template_data)
        self.assertIn('recommendation', analysis_data)
        
        # Log workflow completion
        print(f"Integration test completed:")
        print(f"  Sample size: {sample_size}")
        print(f"  P-value: {analysis_data['results']['p_value']}")
        print(f"  Significant: {analysis_data['results']['is_significant']}")
        
    @patch('app.ab_designer')
    def test_error_handling_integration(self, mock_designer):
        """Test error handling across the system"""
        # Mock a failure in the AB designer
        mock_designer.generate_workflow_steps.side_effect = Exception("Test error")
        
        payload = {
            'question': 'Error test question',
            'hypothesis': 'Error test hypothesis',
            'metric': 'conversion-rate',
            'effect_size': 15
        }
        
        response = self.app.post('/api/design',
                               data=json.dumps(payload),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 500)
        data = json.loads(response.data)
        self.assertIn('error', data)
        
    def test_concurrent_requests(self):
        """Test system behavior under concurrent requests"""
        import threading
        import time
        
        results = []
        
        def make_request():
            response = self.app.get('/api/health')
            results.append(response.status_code)
            
        # Create multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            
        # Start all threads
        for thread in threads:
            thread.start()
            
        # Wait for completion
        for thread in threads:
            thread.join(timeout=5)
            
        # Check results
        self.assertEqual(len(results), 10)
        self.assertTrue(all(code == 200 for code in results))
        
    def test_data_consistency(self):
        """Test data consistency across different components"""
        # Generate data with data generator
        datasets = self.generator.generate_complete_dataset(n_users=100)
        
        # Extract conversion data for statistical analysis
        conversions = datasets['conversions']
        control_mask = conversions['ads_id'].str.contains('CONTROL')
        treatment_mask = ~control_mask
        
        control_conversions = conversions[control_mask]['converted'].sum()
        control_total = control_mask.sum()
        treatment_conversions = conversions[treatment_mask]['converted'].sum()
        treatment_total = treatment_mask.sum()
        
        # Analyze using API
        analysis_payload = {
            'control_conversions': int(control_conversions),
            'control_visitors': int(control_total),
            'treatment_conversions': int(treatment_conversions),
            'treatment_visitors': int(treatment_total)
        }
        
        response = self.app.post('/api/analyze',
                               data=json.dumps(analysis_payload),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Verify consistency
        api_control_rate = data['results']['control_rate']
        expected_control_rate = control_conversions / control_total
        self.assertAlmostEqual(api_control_rate, expected_control_rate, places=4)


class TestPerformance(unittest.TestCase):
    """Performance tests for the A/B Test Designer"""
    
    def setUp(self):
        """Set up performance test fixtures"""
        self.app = app.test_client()
        self.app.testing = True
        
    def test_api_response_time(self):
        """Test API response times are reasonable"""
        import time
        
        # Test health endpoint
        start_time = time.time()
        response = self.app.get('/api/health')
        end_time = time.time()
        
        self.assertEqual(response.status_code, 200)
        self.assertLess(end_time - start_time, 1.0)  # Should respond within 1 second
        
        # Test design endpoint
        payload = {
            'question': 'Performance test question',
            'hypothesis': 'Performance test hypothesis',
            'metric': 'conversion-rate',
            'effect_size': 15
        }
        
        start_time = time.time()
        response = self.app.post('/api/design',
                               data=json.dumps(payload),
                               content_type='application/json')
        end_time = time.time()
        
        self.assertEqual(response.status_code, 200)
        self.assertLess(end_time - start_time, 3.0)  # Should complete within 3 seconds
        
    def test_large_dataset_handling(self):
        """Test handling of large datasets"""
        generator = DataGenerator()
        
        # Test with moderately large dataset
        start_time = time.time()
        datasets = generator.generate_complete_dataset(n_users=1000)
        end_time = time.time()
        
        self.assertLess(end_time - start_time, 10.0)  # Should complete within 10 seconds
        self.assertEqual(len(datasets['users']), 1000)
        
    def test_memory_usage(self):
        """Test memory usage doesn't grow excessively"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Perform multiple operations
        for i in range(10):
            response = self.app.get('/api/data-template?sample_size=100')
            self.assertEqual(response.status_code, 200)
            
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (less than 100MB)
        self.assertLess(memory_growth, 100 * 1024 * 1024)


if __name__ == '__main__':
    # Configure test runner
    unittest.main(
        verbosity=2,
        failfast=False,
        buffer=True,
        warnings='ignore'
    )