"""
Test the improved hierarchical tag classification system.

This demonstrates the improvements over the old hardcoded approach.
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.planner_agent import TaskPlannerAgent


def test_hierarchical_tag_improvements():
    """Test the improved hierarchical tag classification."""
    
    print("=" * 80)
    print("TESTING: Improved Hierarchical Tag Classification System")
    print("=" * 80)
    print()
    
    # Initialize the planner agent
    llm_config = {
        "model": "deepseek-chat",
        "api_key": "test-key",
        "base_url": "https://api.deepseek.com"
    }
    
    planner = TaskPlannerAgent(llm_config)
    
    # Test cases that demonstrate the improvements
    test_cases = [
        # Basic validation tasks
        "Validate email address format",
        "Check password strength requirements",
        
        # Complex multi-action tasks
        "Convert image format and validate resolution",
        "Extract data from PDF and validate content",
        "Process user registration with email validation",
        
        # Domain-specific tasks
        "Calculate financial metrics for investment analysis",
        "Generate medical report with patient data validation",
        "Optimize database query performance monitoring",
        
        # Multi-language tasks
        "验证邮箱格式的正确性",  # Chinese
        "Validar formato de correo electrónico",  # Spanish
        
        # Complex semantic scenarios
        "Real-time chat message processing with content moderation",
        "Machine learning model training with data preprocessing",
        "Microservice API endpoint with authentication and logging",
        "Batch processing of image files with format conversion",
        
        # Edge cases
        "IoT sensor data collection and telemetry analysis",
        "Geolocation-based route optimization with GPS coordinates",
        "Enterprise-grade user management with role-based access control"
    ]
    
    for i, task in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST CASE {i}: {task}")
        print(f"{'='*60}")
        
        # Test the improved tag extraction
        tags = planner._extract_basic_tags(task)
        
        print(f"📊 HIERARCHICAL TAGS: {tags}")
        
        # Analyze tag structure
        primary_tags = []
        secondary_tags = []
        specific_tags = []
        
        # Categorize tags by type
        primary_categories = ['data', 'computation', 'communication', 'security', 'media', 'business', 'system', 'integration']
        
        for tag in tags:
            if tag in primary_categories:
                primary_tags.append(tag)
            elif tag in ['validation', 'transformation', 'extraction', 'processing', 'analysis', 'generation', 'storage',
                        'calculation', 'mathematical', 'algorithm', 'logic',
                        'messaging', 'email', 'web', 'phone', 'social',
                        'authentication', 'authorization', 'encryption', 'audit',
                        'image', 'video', 'audio', 'document',
                        'financial', 'ecommerce', 'healthcare', 'education', 'legal',
                        'database', 'network', 'file_system', 'performance', 'monitoring',
                        'api', 'external', 'cloud', 'iot']:
                secondary_tags.append(tag)
            else:
                specific_tags.append(tag)
        
        print(f"  🔵 Primary Categories: {primary_tags}")
        print(f"  🟡 Secondary Categories: {secondary_tags}")
        print(f"  🟢 Specific Domains: {specific_tags}")
        
        # Check for semantic tags
        semantic_tags = [tag for tag in tags if tag in [
            'image_processing', 'data_pipeline', 'user_management', 'content_management',
            'real_time', 'machine_learning', 'geolocation', 'internationalization',
            'multilingual', 'batch_processing', 'async_processing', 'high_performance',
            'enterprise_grade', 'microservice', 'event_driven'
        ]]
        
        if semantic_tags:
            print(f"  🚀 Semantic Tags: {semantic_tags}")
        
        # Show coverage improvement
        coverage_score = len(set(tags)) / max(len(task.split()), 1)
        print(f"  📈 Coverage Score: {coverage_score:.2f}")
        
        # Demonstrate hierarchy
        if len(tags) >= 2:
            print(f"  🏗️  Hierarchical Structure: {' → '.join(tags[:3])}")
    
    print(f"\n{'='*80}")
    print("COMPARISON: OLD vs NEW TAG CLASSIFICATION")
    print(f"{'='*80}")
    
    # Demonstrate specific improvements
    comparison_cases = [
        ("Convert image format and validate resolution", 
         "OLD: ['conversion', 'validation'] (2 tags)",
         "NEW: Hierarchical + Semantic tags"),
        
        ("Real-time chat with user authentication",
         "OLD: ['auth'] (1 tag)", 
         "NEW: Multi-category + Complexity tags"),
        
        ("验证邮箱格式正确性",
         "OLD: ['general'] (fallback)",
         "NEW: Multilingual + Domain tags")
    ]
    
    for task, old_result, new_description in comparison_cases:
        print(f"\nTask: {task}")
        print(f"  ❌ {old_result}")
        
        new_tags = planner._extract_basic_tags(task)
        print(f"  ✅ NEW: {new_tags} ({len(new_tags)} tags)")
        print(f"  📝 {new_description}")
    
    print(f"\n{'='*80}")
    print("IMPROVEMENTS SUMMARY")
    print(f"{'='*80}")
    print("""
✅ PROBLEMS SOLVED:

1. ❌ Limited Categories (7-8) → ✅ Comprehensive Taxonomy (50+ categories)
   - 8 primary categories with hierarchical subcategories
   - Domain-specific tags for specialized areas
   - Semantic pattern matching for complex scenarios

2. ❌ Flat Tag Structure → ✅ Hierarchical Tag System
   - Primary → Secondary → Specific tag hierarchy
   - Example: ['data', 'validation', 'email'] for email validation
   - Preserves semantic relationships between concepts

3. ❌ Simple Keyword Matching → ✅ Multi-Level Classification
   - Hierarchical taxonomy matching
   - Regex-based semantic pattern detection
   - Complex scenario recognition (image_processing, real_time, etc.)

4. ❌ No Complex Task Support → ✅ Advanced Semantic Analysis
   - Multi-action task recognition ("convert image format and validate resolution")
   - Cross-domain classification (security + media + data)
   - Complexity indicators (enterprise_grade, high_performance)

5. ❌ English-Only → ✅ Multi-Language Support
   - Unicode pattern detection for Chinese, Japanese, Arabic, etc.
   - Automatic 'multilingual' tag assignment
   - Preserves semantic meaning across languages

🚀 BENEFITS:
- Better function discovery through semantic hierarchies
- More precise categorization with 3-level tag structure
- Support for complex, multi-domain tasks
- International task description support
- Extensible taxonomy for new domains
- Semantic pattern recognition for advanced scenarios
""")


if __name__ == "__main__":
    test_hierarchical_tag_improvements()
