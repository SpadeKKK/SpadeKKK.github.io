#!/usr/bin/env python3
"""
Data Generator for A/B Test Designer
Generates realistic demo data for testing and demonstration purposes
"""

import pandas as pd
import numpy as np
import random
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import string
import uuid

# Configure logging
logger = logging.getLogger(__name__)

class DataGenerator:
    """
    Generates realistic demo data for A/B testing scenarios
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize data generator with random seed for reproducibility
        
        Args:
            seed: Random seed for reproducible data generation
        """
        np.random.seed(seed)
        random.seed(seed)
        
        # Sample data pools for realistic generation
        self.first_names = [
            "Emma", "Liam", "Olivia", "Noah", "Ava", "William", "Sophia", "James",
            "Isabella", "Oliver", "Charlotte", "Benjamin", "Amelia", "Lucas", "Mia",
            "Henry", "Harper", "Alexander", "Evelyn", "Mason", "Abigail", "Michael",
            "Emily", "Ethan", "Elizabeth", "Sofia", "Avery", "Logan", "Ella", "Jackson",
            "Scarlett", "Sebastian", "Madison", "Jack", "Eleanor", "Aiden", "Grace",
            "Owen", "Chloe", "Samuel", "Victoria", "Matthew", "Riley", "Joseph", "Aria"
        ]
        
        self.last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
            "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
            "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
            "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
            "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
            "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green",
            "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell"
        ]
        
        self.email_domains = [
            "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "company.com",
            "email.com", "test.com", "example.org", "domain.net", "service.io",
            "business.com", "corporation.net", "startup.co", "tech.ai", "work.org"
        ]
        
        self.companies = [
            "TechCorp", "InnovateInc", "DataSolutions", "CloudSystems", "AIStartup",
            "DigitalFlow", "SmartTech", "FutureLabs", "CodeCraft", "WebDynamics"
        ]
        
        # Experiment type configurations
        self.experiment_types = {
            'button_color': {
                'control': 'Control - Blue Button',
                'treatment': 'Treatment - Red Button',
                'description': 'Testing button color impact on conversions'
            },
            'headline': {
                'control': 'Control - Original Headline',
                'treatment': 'Treatment - Optimized Headline',
                'description': 'Testing headline effectiveness'
            },
            'pricing': {
                'control': 'Control - Standard Pricing',
                'treatment': 'Treatment - Discount Pricing',
                'description': 'Testing pricing strategy impact'
            },
            'layout': {
                'control': 'Control - Current Layout',
                'treatment': 'Treatment - Streamlined Layout',
                'description': 'Testing page layout optimization'
            },
            'form_fields': {
                'control': 'Control - Long Form',
                'treatment': 'Treatment - Short Form',
                'description': 'Testing form length impact on completion'
            },
            'call_to_action': {
                'control': 'Control - "Submit"',
                'treatment': 'Treatment - "Get Started"',
                'description': 'Testing CTA text effectiveness'
            },
            'product_images': {
                'control': 'Control - Stock Photos',
                'treatment': 'Treatment - Custom Images',
                'description': 'Testing image type impact on engagement'
            },
            'page_load_speed': {
                'control': 'Control - Standard Speed',
                'treatment': 'Treatment - Optimized Speed',
                'description': 'Testing page speed impact on conversions'
            }
        }
    
    def generate_user_id(self, index: int, format_type: str = "standard") -> str:
        """
        Generate realistic user IDs in various formats
        
        Args:
            index: User index number
            format_type: Type of ID format ('standard', 'uuid', 'hash')
            
        Returns:
            Generated user ID string
        """
        try:
            if format_type == "uuid":
                return str(uuid.uuid4())
            elif format_type == "hash":
                return f"user_{hash(f'user_{index}') % 1000000:06d}"
            else:  # standard
                return f"U{str(index).zfill(6)}"
        except Exception as e:
            logger.warning(f"Error generating user ID: {e}")
            return f"U{str(index).zfill(6)}"
    
    def generate_username(self, index: int, format_type: str = "email") -> str:
        """
        Generate realistic usernames in various formats
        
        Args:
            index: User index
            format_type: Format type ('email', 'username', 'mixed')
            
        Returns:
            Generated username string
        """
        try:
            first_name = random.choice(self.first_names).lower()
            last_name = random.choice(self.last_names).lower()
            domain = random.choice(self.email_domains)
            
            if format_type == "email":
                formats = [
                    f"{first_name}.{last_name}@{domain}",
                    f"{first_name}{last_name}@{domain}",
                    f"{first_name}_{last_name}@{domain}",
                    f"{first_name}{random.randint(10, 999)}@{domain}",
                    f"{first_name[0]}{last_name}@{domain}",
                    f"{first_name}.{last_name}{random.randint(10, 99)}@{domain}"
                ]
            elif format_type == "username":
                formats = [
                    f"{first_name}_{last_name}",
                    f"{first_name}{last_name}{random.randint(10, 999)}",
                    f"{first_name[0]}{last_name}",
                    f"{first_name}.{last_name}",
                    f"user_{first_name}_{index}"
                ]
            else:  # mixed
                formats = [
                    f"{first_name}.{last_name}@{domain}",
                    f"{first_name}_{last_name}",
                    f"user_{index}@{domain}"
                ]
            
            return random.choice(formats)
            
        except Exception as e:
            logger.warning(f"Error generating username: {e}")
            return f"user_{index}@example.com"
    
    def generate_users_table(self, n_users: int = 1000, id_format: str = "standard", 
                           username_format: str = "email") -> pd.DataFrame:
        """
        Generate Table 1: Users (user_id, username)
        
        Args:
            n_users: Number of users to generate
            id_format: Format for user IDs
            username_format: Format for usernames
            
        Returns:
            DataFrame with user data
        """
        try:
            users_data = []
            
            for i in range(1, n_users + 1):
                user_id = self.generate_user_id(i, id_format)
                username = self.generate_username(i, username_format)
                
                users_data.append({
                    'user_id': user_id,
                    'username': username
                })
            
            df = pd.DataFrame(users_data)
            
            # Ensure uniqueness
            df = df.drop_duplicates(subset=['user_id'])
            df = df.reset_index(drop=True)
            
            logger.info(f"Generated {len(df)} unique users")
            return df
            
        except Exception as e:
            logger.error(f"Error generating users table: {e}")
            # Return minimal fallback data
            return pd.DataFrame([
                {'user_id': 'U000001', 'username': 'user1@example.com'},
                {'user_id': 'U000002', 'username': 'user2@example.com'}
            ])
    
    def generate_ads_table(self, experiment_type: str = "button_color", 
                          custom_variants: Optional[Dict] = None) -> pd.DataFrame:
        """
        Generate Table 3: Ad Variants (ads_id, ads_name)
        
        Args:
            experiment_type: Type of experiment
            custom_variants: Custom variant definitions
            
        Returns:
            DataFrame with ad variant data
        """
        try:
            if custom_variants:
                ads_data = [
                    {"ads_id": "AD_CONTROL", "ads_name": custom_variants.get('control', 'Control Variant')},
                    {"ads_id": "AD_TREATMENT", "ads_name": custom_variants.get('treatment', 'Treatment Variant')}
                ]
            else:
                config = self.experiment_types.get(experiment_type, self.experiment_types['button_color'])
                ads_data = [
                    {"ads_id": "AD_CONTROL", "ads_name": config['control']},
                    {"ads_id": "AD_TREATMENT", "ads_name": config['treatment']}
                ]
            
            df = pd.DataFrame(ads_data)
            logger.info(f"Generated ads table for {experiment_type} experiment")
            return df
            
        except Exception as e:
            logger.error(f"Error generating ads table: {e}")
            return pd.DataFrame([
                {"ads_id": "AD_CONTROL", "ads_name": "Control Variant"},
                {"ads_id": "AD_TREATMENT", "ads_name": "Treatment Variant"}
            ])
    
    def generate_interactions_table(self, users_df: pd.DataFrame, ads_df: pd.DataFrame,
                                   treatment_split: float = 0.5, 
                                   stratify_by: Optional[str] = None) -> pd.DataFrame:
        """
        Generate Table 2: User-Ad Interactions (user_id, ads_id)
        
        Args:
            users_df: DataFrame with user data
            ads_df: DataFrame with ad variant data
            treatment_split: Proportion assigned to treatment (0.5 = 50/50 split)
            stratify_by: Column to stratify randomization by
            
        Returns:
            DataFrame with interaction data
        """
        try:
            interactions_data = []
            control_ad_id = ads_df.iloc[0]['ads_id']
            treatment_ad_id = ads_df.iloc[1]['ads_id']
            
            # Simple randomization
            if stratify_by is None:
                for _, user in users_df.iterrows():
                    if np.random.random() < treatment_split:
                        ads_id = treatment_ad_id
                    else:
                        ads_id = control_ad_id
                    
                    interactions_data.append({
                        'user_id': user['user_id'],
                        'ads_id': ads_id
                    })
            
            # Stratified randomization (placeholder implementation)
            else:
                # For demo purposes, we'll implement simple stratification
                for _, user in users_df.iterrows():
                    # Hash-based stratification for consistent assignment
                    user_hash = hash(user['user_id']) % 100
                    if user_hash < treatment_split * 100:
                        ads_id = treatment_ad_id
                    else:
                        ads_id = control_ad_id
                    
                    interactions_data.append({
                        'user_id': user['user_id'],
                        'ads_id': ads_id
                    })
            
            df = pd.DataFrame(interactions_data)
            
            # Log assignment statistics
            control_count = (df['ads_id'] == control_ad_id).sum()
            treatment_count = (df['ads_id'] == treatment_ad_id).sum()
            actual_split = treatment_count / len(df) if len(df) > 0 else 0
            
            logger.info(f"Generated interactions: {control_count} control, {treatment_count} treatment (split: {actual_split:.2%})")
            return df
            
        except Exception as e:
            logger.error(f"Error generating interactions table: {e}")
            # Return minimal fallback
            return pd.DataFrame([
                {'user_id': users_df.iloc[0]['user_id'] if len(users_df) > 0 else 'U000001', 
                 'ads_id': 'AD_CONTROL'},
                {'user_id': users_df.iloc[1]['user_id'] if len(users_df) > 1 else 'U000002', 
                 'ads_id': 'AD_TREATMENT'}
            ])
    
    def generate_conversion_data(self, interactions_df: pd.DataFrame, 
                               control_rate: float = 0.10, treatment_rate: float = 0.12,
                               add_noise: bool = True, seasonal_effect: bool = False) -> pd.DataFrame:
        """
        Generate realistic conversion data with configurable parameters
        
        Args:
            interactions_df: DataFrame with user-ad interactions
            control_rate: Base conversion rate for control group
            treatment_rate: Base conversion rate for treatment group
            add_noise: Whether to add realistic noise to conversion rates
            seasonal_effect: Whether to simulate seasonal effects
            
        Returns:
            DataFrame with conversion data
        """
        try:
            conversion_data = []
            
            for _, interaction in interactions_df.iterrows():
                # Determine base conversion rate
                if "CONTROL" in interaction['ads_id']:
                    base_rate = control_rate
                else:
                    base_rate = treatment_rate
                
                # Add noise to make data more realistic
                if add_noise:
                    # Add user-level variation (some users more likely to convert)
                    user_effect = np.random.normal(0, 0.02)  # ±2% variation
                    
                    # Add time-based variation
                    time_effect = 0
                    if seasonal_effect:
                        day_of_month = np.random.randint(1, 31)
                        # Simple seasonal pattern (higher conversions mid-month)
                        time_effect = 0.01 * np.sin(2 * np.pi * day_of_month / 30)
                    
                    actual_rate = base_rate + user_effect + time_effect
                    actual_rate = max(0, min(1, actual_rate))  # Clamp to [0, 1]
                else:
                    actual_rate = base_rate
                
                # Simulate conversion
                converted = np.random.random() < actual_rate
                
                # Generate conversion value (revenue) if converted
                if converted:
                    # Log-normal distribution for realistic revenue
                    conversion_value = np.random.lognormal(mean=3.2, sigma=0.6)  # ~$25 average
                    conversion_value = max(1.0, conversion_value)  # Minimum $1
                else:
                    conversion_value = 0.0
                
                # Generate timestamp
                conversion_timestamp = self._generate_realistic_timestamp()
                
                conversion_data.append({
                    'user_id': interaction['user_id'],
                    'ads_id': interaction['ads_id'],
                    'converted': converted,
                    'conversion_value': round(conversion_value, 2),
                    'timestamp': conversion_timestamp,
                    'session_duration': max(10, np.random.lognormal(4.5, 1.2)),  # Session time in seconds
                    'pages_viewed': np.random.poisson(3) + 1  # Number of pages viewed
                })
            
            df = pd.DataFrame(conversion_data)
            
            # Log conversion statistics
            control_mask = df['ads_id'].str.contains('CONTROL')
            treatment_mask = ~control_mask
            
            control_conversions = df[control_mask]['converted'].sum()
            control_total = control_mask.sum()
            treatment_conversions = df[treatment_mask]['converted'].sum()
            treatment_total = treatment_mask.sum()
            
            logger.info(f"Generated conversion data:")
            logger.info(f"  Control: {control_conversions}/{control_total} = {control_conversions/control_total:.2%}")
            logger.info(f"  Treatment: {treatment_conversions}/{treatment_total} = {treatment_conversions/treatment_total:.2%}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating conversion data: {e}")
            # Return minimal fallback
            return pd.DataFrame([
                {
                    'user_id': interactions_df.iloc[0]['user_id'] if len(interactions_df) > 0 else 'U000001',
                    'ads_id': 'AD_CONTROL',
                    'converted': False,
                    'conversion_value': 0.0,
                    'timestamp': datetime.now().isoformat()
                }
            ])
    
    def generate_engagement_data(self, interactions_df: pd.DataFrame,
                                control_engagement: Dict = None,
                                treatment_engagement: Dict = None) -> pd.DataFrame:
        """
        Generate engagement metrics (time spent, clicks, bounces, etc.)
        
        Args:
            interactions_df: DataFrame with user-ad interactions
            control_engagement: Control group engagement parameters
            treatment_engagement: Treatment group engagement parameters
            
        Returns:
            DataFrame with engagement data
        """
        try:
            # Default engagement parameters
            if control_engagement is None:
                control_engagement = {
                    'avg_time_spent': 120,  # seconds
                    'avg_clicks': 2.5,
                    'bounce_rate': 0.4,
                    'avg_pages': 2.8
                }
            
            if treatment_engagement is None:
                treatment_engagement = {
                    'avg_time_spent': 135,  # 12.5% increase
                    'avg_clicks': 3.0,      # 20% increase
                    'bounce_rate': 0.35,    # 12.5% decrease
                    'avg_pages': 3.2        # 14% increase
                }
            
            engagement_data = []
            
            for _, interaction in interactions_df.iterrows():
                # Choose parameters based on variant
                if "CONTROL" in interaction['ads_id']:
                    params = control_engagement
                else:
                    params = treatment_engagement
                
                # Generate metrics with realistic distributions
                time_spent = max(5, np.random.lognormal(
                    np.log(params['avg_time_spent']), 0.8
                ))
                
                clicks = max(0, np.random.poisson(params['avg_clicks']))
                
                bounced = np.random.random() < params['bounce_rate']
                
                if bounced:
                    pages_viewed = 1
                    time_spent = min(time_spent, 30)  # Bounced users spend less time
                else:
                    pages_viewed = max(1, np.random.poisson(params['avg_pages']))
                
                # Additional engagement metrics
                scroll_depth = np.random.beta(2, 3) if not bounced else np.random.beta(1, 4)
                
                engagement_data.append({
                    'user_id': interaction['user_id'],
                    'ads_id': interaction['ads_id'],
                    'time_spent_seconds': round(time_spent, 1),
                    'clicks': clicks,
                    'pages_viewed': pages_viewed,
                    'bounced': bounced,
                    'scroll_depth_percent': round(scroll_depth * 100, 1),
                    'return_visitor': np.random.random() < 0.3,  # 30% return visitors
                    'mobile_device': np.random.random() < 0.6    # 60% mobile traffic
                })
            
            df = pd.DataFrame(engagement_data)
            
            # Log engagement statistics
            control_mask = df['ads_id'].str.contains('CONTROL')
            treatment_mask = ~control_mask
            
            logger.info(f"Generated engagement data:")
            logger.info(f"  Control avg time: {df[control_mask]['time_spent_seconds'].mean():.1f}s")
            logger.info(f"  Treatment avg time: {df[treatment_mask]['time_spent_seconds'].mean():.1f}s")
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating engagement data: {e}")
            return pd.DataFrame([{
                'user_id': interactions_df.iloc[0]['user_id'] if len(interactions_df) > 0 else 'U000001',
                'ads_id': 'AD_CONTROL',
                'time_spent_seconds': 120.0,
                'clicks': 2,
                'pages_viewed': 3,
                'bounced': False
            }])
    
    def _generate_realistic_timestamp(self) -> str:
        """Generate realistic timestamp within the last 30 days"""
        try:
            start_date = datetime.now() - timedelta(days=30)
            end_date = datetime.now()
            
            # Add realistic patterns (more activity during business hours)
            random_days = np.random.randint(0, 30)
            base_time = start_date + timedelta(days=random_days)
            
            # Bias towards business hours (9 AM - 6 PM)
            if np.random.random() < 0.7:  # 70% during business hours
                hour = np.random.randint(9, 18)
            else:
                hour = np.random.randint(0, 24)
            
            minute = np.random.randint(0, 60)
            second = np.random.randint(0, 60)
            
            timestamp = base_time.replace(hour=hour, minute=minute, second=second)
            return timestamp.isoformat()
            
        except Exception as e:
            logger.warning(f"Error generating timestamp: {e}")
            return datetime.now().isoformat()
    
    def generate_complete_dataset(self, n_users: int = 1000, experiment_type: str = "button_color",
                                 control_rate: float = 0.10, treatment_rate: float = 0.12,
                                 treatment_split: float = 0.5, 
                                 include_engagement: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Generate complete dataset for A/B testing with all required tables
        
        Args:
            n_users: Number of users to generate
            experiment_type: Type of experiment to simulate
            control_rate: Control group conversion rate
            treatment_rate: Treatment group conversion rate
            treatment_split: Proportion assigned to treatment
            include_engagement: Whether to include engagement metrics
            
        Returns:
            Dictionary containing all generated DataFrames
        """
        try:
            logger.info(f"Generating complete dataset: {n_users} users, {experiment_type} experiment")
            
            # Generate core tables
            users_df = self.generate_users_table(n_users)
            ads_df = self.generate_ads_table(experiment_type)
            interactions_df = self.generate_interactions_table(users_df, ads_df, treatment_split)
            
            # Generate behavioral data
            conversions_df = self.generate_conversion_data(interactions_df, control_rate, treatment_rate)
            
            datasets = {
                'users': users_df,
                'ads': ads_df,
                'interactions': interactions_df,
                'conversions': conversions_df
            }
            
            # Optional engagement data
            if include_engagement:
                engagement_df = self.generate_engagement_data(interactions_df)
                datasets['engagement'] = engagement_df
            
            logger.info(f"Dataset generation completed successfully")
            return datasets
            
        except Exception as e:
            logger.error(f"Error generating complete dataset: {e}")
            # Return minimal fallback dataset
            return {
                'users': pd.DataFrame([{'user_id': 'U000001', 'username': 'user1@example.com'}]),
                'ads': pd.DataFrame([{'ads_id': 'AD_CONTROL', 'ads_name': 'Control'}]),
                'interactions': pd.DataFrame([{'user_id': 'U000001', 'ads_id': 'AD_CONTROL'}])
            }
    
    def export_to_csv(self, datasets: Dict[str, pd.DataFrame], output_dir: str = "./data/") -> None:
        """
        Export all datasets to CSV files
        
        Args:
            datasets: Dictionary of DataFrames to export
            output_dir: Directory to save CSV files
        """
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            for name, df in datasets.items():
                filename = os.path.join(output_dir, f"{name}_data.csv")
                df.to_csv(filename, index=False)
                logger.info(f"Exported {name} data to {filename} ({len(df)} rows)")
                
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
    
    def generate_api_sample(self, n_users: int = 20, experiment_type: str = "button_color") -> Dict:
        """
        Generate sample data optimized for API responses
        
        Args:
            n_users: Number of users (kept small for API efficiency)
            experiment_type: Type of experiment
            
        Returns:
            Dictionary with sample data ready for JSON serialization
        """
        try:
            datasets = self.generate_complete_dataset(
                n_users=n_users, 
                experiment_type=experiment_type,
                include_engagement=False  # Keep API responses lightweight
            )
            
            return {
                'table1_users': datasets['users'].to_dict('records'),
                'table2_interactions': datasets['interactions'].to_dict('records'),
                'table3_ads': datasets['ads'].to_dict('records'),
                'sample_conversions': datasets['conversions'].head(10).to_dict('records'),
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'total_users': len(datasets['users']),
                    'experiment_type': experiment_type,
                    'control_group_size': (datasets['interactions']['ads_id'] == 'AD_CONTROL').sum(),
                    'treatment_group_size': (datasets['interactions']['ads_id'] == 'AD_TREATMENT').sum()
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating API sample: {e}")
            return {
                'table1_users': [{'user_id': 'U000001', 'username': 'user1@example.com'}],
                'table2_interactions': [{'user_id': 'U000001', 'ads_id': 'AD_CONTROL'}],
                'table3_ads': [{'ads_id': 'AD_CONTROL', 'ads_name': 'Control'}],
                'metadata': {'error': str(e)}
            }

# Example usage and testing
if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(level=logging.INFO)
    
    # Initialize generator
    generator = DataGenerator(seed=42)
    
    # Generate complete dataset
    print("Generating complete A/B test dataset...")
    datasets = generator.generate_complete_dataset(
        n_users=500,
        experiment_type="button_color",
        control_rate=0.08,
        treatment_rate=0.11
    )
    
    # Display summary statistics
    print(f"\nDataset Summary:")
    for name, df in datasets.items():
        print(f"  {name}: {len(df)} rows, {len(df.columns)} columns")
    
    # Show sample data
    print(f"\nSample Users:")
    print(datasets['users'].head())
    
    print(f"\nSample Interactions:")
    print(datasets['interactions'].head())
    
    print(f"\nConversion Rates by Group:")
    conv_summary = datasets['conversions'].groupby('ads_id')['converted'].agg(['count', 'sum', 'mean'])
    print(conv_summary)
    
    # Export to files
    print(f"\nExporting to CSV files...")
    generator.export_to_csv(datasets)
    
    # Generate API sample
    print(f"\nGenerating API sample...")
    api_sample = generator.generate_api_sample(n_users=20)
    
    with open('sample_api_data.json', 'w') as f:
        json.dump(api_sample, f, indent=2)
    
    print("Data generation completed successfully!")