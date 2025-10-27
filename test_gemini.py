"""
Test Gemini LLM Service
"""
from app.services.llm_service import gemini_service

print("=" * 60)
print("Testing Gemini LLM Service")
print("=" * 60)

# Test 1: Check if available
print("\n1️⃣ Checking LLM availability...")
is_available = gemini_service.is_available()
print(f"   LLM Available: {is_available}")

if not is_available:
    print("\n❌ Gemini not configured!")
    print("   Please set GOOGLE_API_KEY in .env file")
    exit(1)

# Test 2: Simple generation
print("\n2️⃣ Testing simple generation...")
prompt = "Explain what machine learning bias is in 2 sentences."
response = gemini_service.generate_summary(prompt)
print(f"   Response: {response[:200]}...")

# Test 3: Bias summary
print("\n3️⃣ Testing bias summary generation...")
bias_data = {
    "sensitive_attribute": "gender",
    "groups": ["Male", "Female"],
    "overall_metrics": {
        "disparate_impact": 0.67,
        "demographic_parity_difference": 0.20,
        "equal_opportunity_difference": 0.15
    },
    "group_metrics": {
        "Male": {
            "group_size": 50,
            "positive_rate": 0.60,
            "accuracy": 0.85,
            "true_positive_rate": 0.83
        },
        "Female": {
            "group_size": 50,
            "positive_rate": 0.40,
            "accuracy": 0.80,
            "true_positive_rate": 0.68
        }
    }
}

summaries = gemini_service.generate_combined_summary(bias_data=bias_data)
print(f"   Generated {len(summaries)} summary sections")
if "bias_summary" in summaries:
    print(f"\n   Bias Summary Preview:")
    print(f"   {summaries['bias_summary'][:300]}...")

print("\n" + "=" * 60)
print("✅ Gemini Testing Complete!")
print("=" * 60)