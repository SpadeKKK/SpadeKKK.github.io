#!/usr/bin/env python3
"""
AI Chat Assistant for A/B Test Designer
Intelligent conversation system to guide users through A/B test setup
"""

import re
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import random

# Configure logging
logger = logging.getLogger(__name__)

class ChatState(Enum):
    """Chat conversation states"""
    GREETING = "greeting"
    UNDERSTANDING_GOAL = "understanding_goal"
    COLLECTING_DETAILS = "collecting_details"
    CONFIRMING_PARAMETERS = "confirming_parameters"
    READY_FOR_ANALYSIS = "ready_for_analysis"
    PROVIDING_RESULTS = "providing_results"

class TestType(Enum):
    """Types of A/B tests"""
    CONVERSION_RATE = "conversion_rate"
    CLICK_THROUGH_RATE = "click_through_rate"
    REVENUE = "revenue"
    ENGAGEMENT = "engagement_time"
    RETENTION = "retention_rate"
    BOUNCE_RATE = "bounce_rate"
    CUSTOM = "custom"

@dataclass
class ChatContext:
    """Stores conversation context and extracted parameters"""
    state: ChatState = ChatState.GREETING
    test_type: Optional[TestType] = None
    question: str = ""
    hypothesis: str = ""
    metric: str = ""
    time_period: Optional[str] = None
    effect_size: Optional[float] = None
    baseline_rate: Optional[float] = None
    confidence_level: float = 0.95
    power: float = 0.8
    collected_params: Dict = None
    conversation_history: List[Dict] = None
    
    def __post_init__(self):
        if self.collected_params is None:
            self.collected_params = {}
        if self.conversation_history is None:
            self.conversation_history = []

class ABTestChatAssistant:
    """
    Intelligent chat assistant for A/B test design and analysis
    """
    
    def __init__(self):
        self.context = ChatContext()
        self.intent_patterns = self._load_intent_patterns()
        self.responses = self._load_responses()
        
    def _load_intent_patterns(self) -> Dict:
        """Load patterns for intent recognition"""
        return {
            'test_design': [
                r'design.*test', r'create.*experiment', r'set.*up.*ab.*test',
                r'new.*test', r'start.*testing', r'design.*experiment'
            ],
            'analyze_results': [
                r'analyze.*results', r'check.*significance', r'test.*results',
                r'is.*significant', r'compare.*data', r'analyze.*data'
            ],
            'conversion_rate': [
                r'conversion.*rate', r'conversions?', r'convert', r'purchase',
                r'sign.*up', r'subscribe', r'buy'
            ],
            'click_through': [
                r'click.*through', r'ctr', r'clicks?', r'click.*rate',
                r'button.*click', r'link.*click'
            ],
            'revenue': [
                r'revenue', r'money', r'sales?', r'profit', r'income',
                r'earnings', r'dollar', r'price', r'payment'
            ],
            'engagement': [
                r'engagement', r'time.*spent', r'session.*duration',
                r'page.*views', r'interaction', r'activity'
            ],
            'time_periods': {
                'week': [r'week', r'7.*days?', r'weekly'],
                'month': [r'month', r'30.*days?', r'monthly', r'4.*weeks?'],
                'quarter': [r'quarter', r'3.*months?', r'90.*days?'],
                'year': [r'year', r'12.*months?', r'annual', r'yearly']
            },
            'comparison_intent': [
                r'compare', r'vs\.?', r'versus', r'against', r'difference',
                r'better', r'worse', r'between'
            ]
        }
    
    def _load_responses(self) -> Dict:
        """Load response templates"""
        return {
            'greeting': [
                "👋 Hi! I'm your A/B Test Designer assistant. I'll help you set up and analyze your experiments!",
                "🤖 Hello! Ready to design some winning A/B tests? I'm here to guide you through the process!",
                "✨ Welcome! I'm your statistical analysis companion. Let's create some data-driven insights together!"
            ],
            'clarify_goal': [
                "What would you like to do today? Are you looking to:",
                "I can help you with several things:",
                "Let me know what you'd like to accomplish:"
            ],
            'goal_options': [
                "📊 Design a new A/B test experiment",
                "📈 Analyze existing A/B test results",
                "🔍 Get sample size recommendations",
                "💡 Brainstorm test ideas for your product"
            ],
            'ask_for_details': [
                "Great! Let me gather some details to help you design the perfect test.",
                "Perfect! I'll need a few details to create your experiment design.",
                "Excellent choice! Let's collect the information I need."
            ],
            'time_period_clarification': [
                "When you mention comparing data, what time period are you interested in?",
                "What timeframe would you like to analyze?",
                "Over what period should we compare the results?"
            ],
            'effect_size_explanation': [
                "How much improvement do you expect to see? For example:",
                "What's your expected effect size? This could be:",
                "How big of a change are you hoping for?"
            ],
            'ready_to_analyze': [
                "Perfect! I have all the information I need. Let me analyze this for you... 🔬",
                "Great! Everything looks good. Running your statistical analysis now... 📊",
                "Excellent! Processing your A/B test parameters... ⚡"
            ]
        }
    
    def process_message(self, user_message: str) -> Dict:
        """
        Process user message and return appropriate response
        
        Args:
            user_message: User's input message
            
        Returns:
            Dictionary with response and updated context
        """
        user_message = user_message.lower().strip()
        
        # Add to conversation history
        self.context.conversation_history.append({
            'role': 'user',
            'message': user_message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Process based on current state
        if self.context.state == ChatState.GREETING:
            response = self._handle_greeting(user_message)
        elif self.context.state == ChatState.UNDERSTANDING_GOAL:
            response = self._handle_goal_understanding(user_message)
        elif self.context.state == ChatState.COLLECTING_DETAILS:
            response = self._handle_detail_collection(user_message)
        elif self.context.state == ChatState.CONFIRMING_PARAMETERS:
            response = self._handle_parameter_confirmation(user_message)
        else:
            response = self._handle_general_query(user_message)
        
        # Add assistant response to history
        self.context.conversation_history.append({
            'role': 'assistant',
            'message': response['message'],
            'timestamp': datetime.now().isoformat()
        })
        
        return response
    
    def _handle_greeting(self, message: str) -> Dict:
        """Handle initial greeting and determine user intent"""
        # Check for immediate intent indicators
        intent = self._detect_intent(message)
        
        if intent == 'test_design':
            self.context.state = ChatState.COLLECTING_DETAILS
            return {
                'message': f"{random.choice(self.responses['greeting'])}\n\n{random.choice(self.responses['ask_for_details'])}\n\nWhat would you like to test? For example:\n• 'Does a red button convert better than blue?'\n• 'Will shorter forms increase signups?'\n• 'Does premium pricing affect revenue?'",
                'suggestions': ['Design new A/B test', 'Analyze existing results', 'Get sample size help'],
                'state': self.context.state.value
            }
        elif intent == 'analyze_results':
            self.context.state = ChatState.COLLECTING_DETAILS
            return {
                'message': f"{random.choice(self.responses['greeting'])}\n\nGreat! I'll help you analyze your A/B test results. Please share:\n• Your control group data (visitors & conversions)\n• Your treatment group data (visitors & conversions)\n• What you were testing",
                'suggestions': ['I have test results to analyze', 'Need help with sample size', 'Want to design new test'],
                'state': self.context.state.value
            }
        else:
            self.context.state = ChatState.UNDERSTANDING_GOAL
            return {
                'message': f"{random.choice(self.responses['greeting'])}\n\n{random.choice(self.responses['clarify_goal'])}\n\n" + "\n".join([f"• {option}" for option in self.responses['goal_options']]),
                'suggestions': ['Design A/B test', 'Analyze results', 'Sample size help', 'Test ideas'],
                'state': self.context.state.value
            }
    
    def _handle_goal_understanding(self, message: str) -> Dict:
        """Handle goal clarification"""
        intent = self._detect_intent(message)
        
        if intent == 'test_design' or any(word in message for word in ['design', 'create', 'new', 'test']):
            self.context.state = ChatState.COLLECTING_DETAILS
            return {
                'message': "Perfect! Let's design your A/B test. 🧪\n\nFirst, what would you like to test? Please describe your research question.\n\nExamples:\n• 'Does changing our pricing page layout increase conversions?'\n• 'Will personalized emails improve click rates?'\n• 'Does video content increase engagement time?'",
                'suggestions': ['Button color test', 'Email subject test', 'Pricing page test', 'Landing page test'],
                'state': self.context.state.value
            }
        elif intent == 'analyze_results' or any(word in message for word in ['analyze', 'results', 'significant']):
            self.context.state = ChatState.COLLECTING_DETAILS
            return {
                'message': "Excellent! I'll analyze your A/B test results. 📊\n\nPlease provide:\n1. **Control group**: How many visitors and conversions?\n2. **Treatment group**: How many visitors and conversions?\n3. **What were you testing?**\n\nExample: 'Control had 1000 visitors with 100 conversions, treatment had 1000 visitors with 120 conversions, testing button color'",
                'suggestions': ['I have conversion data', 'I have click data', 'I have revenue data'],
                'state': self.context.state.value
            }
        else:
            return {
                'message': "I'd love to help! Could you be more specific about what you'd like to do?\n\n🤔 Are you looking to:\n• **Design** a new A/B test?\n• **Analyze** existing test results?\n• Get **sample size** recommendations?\n• **Brainstorm** test ideas?",
                'suggestions': ['Design new test', 'Analyze results', 'Sample size help', 'Test ideas'],
                'state': self.context.state.value
            }
    
    def _handle_detail_collection(self, message: str) -> Dict:
        """Handle collection of test details and parameters"""
        # Extract parameters from the message
        extracted_params = self._extract_parameters(message)
        self.context.collected_params.update(extracted_params)
        
        # Check if we have data to analyze
        if self._has_analysis_data(message):
            return self._process_analysis_request(message)
        
        # Build research question if provided
        if not self.context.question and len(message) > 20:
            self.context.question = message
        
        # Determine what we still need
        missing_params = self._get_missing_parameters()
        
        if not missing_params:
            self.context.state = ChatState.CONFIRMING_PARAMETERS
            return self._generate_confirmation()
        else:
            return self._ask_for_missing_parameter(missing_params[0])
    
    def _handle_parameter_confirmation(self, message: str) -> Dict:
        """Handle parameter confirmation and proceed to analysis"""
        if any(word in message.lower() for word in ['yes', 'correct', 'right', 'good', 'proceed', 'continue']):
            self.context.state = ChatState.READY_FOR_ANALYSIS
            return {
                'message': f"{random.choice(self.responses['ready_to_analyze'])}\n\n{self._generate_analysis_results()}",
                'analysis_ready': True,
                'parameters': self.context.collected_params,
                'state': self.context.state.value
            }
        else:
            self.context.state = ChatState.COLLECTING_DETAILS
            return {
                'message': "No problem! Let's update the details. What would you like to change?",
                'suggestions': ['Change time period', 'Update effect size', 'Modify baseline rate', 'Restart setup'],
                'state': self.context.state.value
            }
    
    def _handle_general_query(self, message: str) -> Dict:
        """Handle general queries and provide helpful responses"""
        if any(word in message for word in ['help', '?', 'how', 'what', 'explain']):
            return {
                'message': "I'm here to help! 🤝\n\nI can assist you with:\n• **Designing A/B tests** - I'll guide you through the setup\n• **Analyzing results** - Statistical significance testing\n• **Sample size calculation** - How many users you need\n• **Best practices** - Tips for successful testing\n\nWhat would you like to know more about?",
                'suggestions': ['Design help', 'Analysis help', 'Sample size help', 'Best practices'],
                'state': self.context.state.value
            }
        else:
            return {
                'message': "I'm not sure I understand that completely. Could you rephrase or choose one of these options?",
                'suggestions': ['Design new A/B test', 'Analyze test results', 'Get help', 'Start over'],
                'state': self.context.state.value
            }
    
    def _detect_intent(self, message: str) -> Optional[str]:
        """Detect user intent from message"""
        for intent, patterns in self.intent_patterns.items():
            if isinstance(patterns, list):
                for pattern in patterns:
                    if re.search(pattern, message, re.IGNORECASE):
                        return intent
        return None
    
    def _extract_parameters(self, message: str) -> Dict:
        """Extract A/B test parameters from user message"""
        params = {}
        
        # Extract time periods
        for period, patterns in self.intent_patterns['time_periods'].items():
            for pattern in patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    params['time_period'] = period
                    break
        
        # Extract numbers (potential effect sizes, rates, etc.)
        numbers = re.findall(r'(\d+(?:\.\d+)?)\s*%?', message)
        if numbers:
            # Try to determine what the numbers represent
            if any(word in message for word in ['effect', 'increase', 'improve', 'lift']):
                try:
                    params['effect_size'] = float(numbers[0])
                except (ValueError, IndexError):
                    pass
            elif any(word in message for word in ['baseline', 'current', 'existing']):
                try:
                    params['baseline_rate'] = float(numbers[0])
                except (ValueError, IndexError):
                    pass
        
        # Extract test type
        if any(word in message for word in ['conversion', 'convert', 'purchase', 'signup']):
            params['test_type'] = TestType.CONVERSION_RATE
            params['metric'] = 'conversion-rate'
        elif any(word in message for word in ['click', 'ctr', 'button']):
            params['test_type'] = TestType.CLICK_THROUGH_RATE
            params['metric'] = 'click-through-rate'
        elif any(word in message for word in ['revenue', 'money', 'sales', 'price']):
            params['test_type'] = TestType.REVENUE
            params['metric'] = 'revenue'
        elif any(word in message for word in ['engagement', 'time', 'duration']):
            params['test_type'] = TestType.ENGAGEMENT
            params['metric'] = 'engagement-time'
        
        # Extract analysis data (control/treatment numbers)
        analysis_pattern = r'control.*?(\d+).*?(\d+).*?treatment.*?(\d+).*?(\d+)'
        analysis_match = re.search(analysis_pattern, message, re.IGNORECASE)
        if analysis_match:
            params['control_visitors'] = int(analysis_match.group(1))
            params['control_conversions'] = int(analysis_match.group(2))
            params['treatment_visitors'] = int(analysis_match.group(3))
            params['treatment_conversions'] = int(analysis_match.group(4))
        
        return params
    
    def _has_analysis_data(self, message: str) -> bool:
        """Check if message contains data for analysis"""
        required_keys = ['control_visitors', 'control_conversions', 'treatment_visitors', 'treatment_conversions']
        return all(key in self.context.collected_params for key in required_keys)
    
    def _process_analysis_request(self, message: str) -> Dict:
        """Process analysis request with provided data"""
        params = self.context.collected_params
        
        # Import statistical engine (assuming it's available)
        try:
            from statistical_engine import StatisticalEngine
            engine = StatisticalEngine()
            
            result = engine.run_proportion_test(
                control_conversions=params['control_conversions'],
                control_visitors=params['control_visitors'],
                treatment_conversions=params['treatment_conversions'],
                treatment_visitors=params['treatment_visitors']
            )
            
            # Format results
            control_rate = params['control_conversions'] / params['control_visitors']
            treatment_rate = params['treatment_conversions'] / params['treatment_visitors']
            improvement = ((treatment_rate - control_rate) / control_rate) * 100
            
            significance = "✅ SIGNIFICANT" if result.is_significant else "❌ NOT SIGNIFICANT"
            
            analysis_message = f"""📊 **Analysis Results**

**Control Group**: {params['control_conversions']:,} conversions out of {params['control_visitors']:,} visitors ({control_rate:.2%})
**Treatment Group**: {params['treatment_conversions']:,} conversions out of {params['treatment_visitors']:,} visitors ({treatment_rate:.2%})

**Improvement**: {improvement:+.1f}%
**P-value**: {result.p_value:.4f}
**Statistical Significance**: {significance}

**📋 Recommendation**: {result.recommendation}

**🎯 Interpretation**: {result.interpretation}"""

            return {
                'message': analysis_message,
                'analysis_complete': True,
                'results': {
                    'p_value': result.p_value,
                    'is_significant': result.is_significant,
                    'improvement_percent': improvement,
                    'recommendation': result.recommendation
                },
                'suggestions': ['Run another test', 'Design new experiment', 'Get more details'],
                'state': 'analysis_complete'
            }
            
        except ImportError:
            return {
                'message': "I can see your data! Let me analyze this:\n\n📊 **Your Test Data**:\n" + 
                          f"• Control: {params['control_conversions']}/{params['control_visitors']} = {params['control_conversions']/params['control_visitors']:.2%}\n" +
                          f"• Treatment: {params['treatment_conversions']}/{params['treatment_visitors']} = {params['treatment_conversions']/params['treatment_visitors']:.2%}\n\n" +
                          "For full statistical analysis, please use the main A/B Test Designer interface! 🚀",
                'analysis_data': params,
                'suggestions': ['Open main interface', 'Design new test', 'Get help'],
                'state': 'analysis_ready'
            }
    
    def _get_missing_parameters(self) -> List[str]:
        """Get list of missing required parameters"""
        required = []
        
        if not self.context.question and 'test_type' not in self.context.collected_params:
            required.append('research_question')
        
        if 'test_type' in self.context.collected_params and 'effect_size' not in self.context.collected_params:
            required.append('effect_size')
        
        if 'effect_size' in self.context.collected_params and 'baseline_rate' not in self.context.collected_params:
            required.append('baseline_rate')
        
        if 'baseline_rate' in self.context.collected_params and 'time_period' not in self.context.collected_params:
            required.append('time_period')
        
        return required
    
    def _ask_for_missing_parameter(self, param: str) -> Dict:
        """Ask for a specific missing parameter"""
        if param == 'research_question':
            return {
                'message': "What would you like to test? Please describe your research question.\n\n💡 **Examples**:\n• 'Does a red call-to-action button increase conversions compared to blue?'\n• 'Will personalized email subject lines improve open rates?'\n• 'Does showing customer reviews increase purchase rates?'",
                'suggestions': ['Button color test', 'Email optimization', 'Product page test', 'Pricing test'],
                'state': self.context.state.value
            }
        elif param == 'effect_size':
            return {
                'message': f"{random.choice(self.responses['effect_size_explanation'])}\n\n• **5-10%** - Small but meaningful improvement\n• **10-20%** - Moderate improvement (common goal)\n• **20%+** - Large improvement (ambitious but possible)\n\nWhat improvement do you expect to see?",
                'suggestions': ['5% improvement', '10% improvement', '15% improvement', '20% improvement'],
                'state': self.context.state.value
            }
        elif param == 'baseline_rate':
            return {
                'message': "What's your current baseline rate?\n\n🎯 **For example**:\n• Current conversion rate: 3%\n• Current click-through rate: 2%\n• Current signup rate: 8%\n\nThis helps me calculate the right sample size for your test.",
                'suggestions': ['2%', '5%', '10%', '15%'],
                'state': self.context.state.value
            }
        elif param == 'time_period':
            return {
                'message': f"{random.choice(self.responses['time_period_clarification'])}\n\n⏰ **Common options**:\n• **1 week** - Quick test for high-traffic sites\n• **1 month** - Standard duration for most tests\n• **1 quarter** - Long-term impact measurement\n\nHow long would you like to run this test?",
                'suggestions': ['1 week', '2 weeks', '1 month', '3 months'],
                'state': self.context.state.value
            }
        else:
            return {
                'message': "I need a bit more information. Could you provide more details about your test?",
                'suggestions': ['Tell me more', 'Start over', 'Get help'],
                'state': self.context.state.value
            }
    
    def _generate_confirmation(self) -> Dict:
        """Generate parameter confirmation message"""
        params = self.context.collected_params
        
        test_type = params.get('test_type', 'unknown').value if params.get('test_type') else 'conversion rate'
        effect_size = params.get('effect_size', 'not specified')
        baseline_rate = params.get('baseline_rate', 'not specified')
        time_period = params.get('time_period', 'not specified')
        
        confirmation_message = f"""🔍 **Let me confirm your A/B test parameters**:

**Research Question**: {self.context.question or 'Testing impact on ' + test_type}
**Test Type**: {test_type.replace('_', ' ').title()}
**Expected Effect**: {effect_size}% improvement
**Baseline Rate**: {baseline_rate}%
**Time Period**: {time_period}

Does this look correct? I'll use these parameters to calculate your sample size and create your experiment design."""

        return {
            'message': confirmation_message,
            'confirmation_required': True,
            'parameters': params,
            'suggestions': ['Yes, looks good!', 'Let me change something', 'Start over'],
            'state': self.context.state.value
        }
    
    def _generate_analysis_results(self) -> Dict:
        """Generate analysis results based on collected parameters"""
        params = self.context.collected_params
        
        # Simple sample size calculation
        effect_size = params.get('effect_size', 15)
        baseline_rate = params.get('baseline_rate', 10) / 100
        
        # Simplified sample size formula
        z_alpha = 1.96  # 95% confidence
        z_beta = 0.84   # 80% power
        
        p1 = baseline_rate
        p2 = p1 * (1 + effect_size / 100)
        p_pooled = (p1 + p2) / 2
        
        n = (2 * p_pooled * (1 - p_pooled) * (z_alpha + z_beta)**2) / (p2 - p1)**2
        sample_size = max(100, int(n))
        
        # Calculate duration
        daily_traffic = 500  # assumption
        duration_days = max(7, int((sample_size * 2) / daily_traffic))
        
        results_message = f"""🎯 **Your A/B Test Design**:

**Sample Size**: {sample_size:,} users per group ({sample_size*2:,} total)
**Test Duration**: {duration_days} days (at ~{daily_traffic} daily visitors)
**Success Metric**: {params.get('metric', 'conversion-rate').replace('-', ' ').title()}

**Statistical Parameters**:
• Confidence Level: 95%
• Statistical Power: 80%
• Expected Effect: {effect_size}% improvement

**Next Steps**:
1. 🚀 **Launch your test** using these parameters
2. 📊 **Collect data** for {duration_days} days
3. 📈 **Analyze results** when you have enough data
4. 🎯 **Make data-driven decisions** based on significance

Ready to launch your test? You can now use the main A/B Test Designer interface with these optimized parameters! 🚀"""

        return results_message
    
    def get_conversation_summary(self) -> Dict:
        """Get summary of the conversation and extracted parameters"""
        return {
            'state': self.context.state.value,
            'collected_parameters': self.context.collected_params,
            'research_question': self.context.question,
            'conversation_length': len(self.context.conversation_history),
            'ready_for_analysis': self.context.state in [ChatState.READY_FOR_ANALYSIS, ChatState.PROVIDING_RESULTS]
        }
    
    def reset_conversation(self):
        """Reset the conversation to start fresh"""
        self.context = ChatContext()
        
    def export_parameters_for_api(self) -> Dict:
        """Export collected parameters in format suitable for main API"""
        params = self.context.collected_params.copy()
        
        return {
            'question': self.context.question or f"Testing {params.get('test_type', 'conversion rate')} improvement",
            'hypothesis': f"Treatment will improve {params.get('metric', 'conversion rate')} by {params.get('effect_size', 15)}%",
            'metric': params.get('metric', 'conversion-rate'),
            'effect_size': params.get('effect_size', 15),
            'baseline_rate': params.get('baseline_rate', 10)
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize chat assistant
    assistant = ABTestChatAssistant()
    
    print("🤖 A/B Test Designer Chat Assistant Demo")
    print("=" * 50)
    
    # Simulate conversation
    test_messages = [
        "Hi there!",
        "I want to test if a red button converts better than blue",
        "I expect about 15% improvement",
        "Current conversion rate is about 8%",
        "Let's run it for one month",
        "Yes, looks good!"
    ]
    
    for message in test_messages:
        print(f"\n👤 User: {message}")
        response = assistant.process_message(message)
        print(f"🤖 Assistant: {response['message']}")
        
        if response.get('suggestions'):
            print(f"💡 Suggestions: {', '.join(response['suggestions'])}")
    
    print(f"\n📋 Conversation Summary:")
    summary = assistant.get_conversation_summary()
    print(json.dumps(summary, indent=2))
    
    if summary['ready_for_analysis']:
        print(f"\n🚀 API Parameters:")
        api_params = assistant.export_parameters_for_api()
        print(json.dumps(api_params, indent=2))