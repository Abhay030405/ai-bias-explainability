"""
Phase 3 Testing - LLM Summaries
"""
import requests
import json

print("=" * 60)
print("Phase 3 - LLM Summary Testing")
print("=" * 60)

base_url = "http://127.0.0.1:8000"

# Test 1: Check LLM status
print("\n1️⃣ GET /api/summary/status")
response = requests.get(f"{base_url}/api/summary/status")
print(f"   Status: {response.status_code}")

if response.status_code == 200:
    result = response.json()
    print(f"   LLM Available: {result['llm_available']}")
    print(f"   Model: {result['llm_model']}")
    print(f"   API Key Configured: {result['api_key_configured']}")
    
    if not result['llm_available']:
        print("\n❌ Gemini not configured!")
        print("   Set GOOGLE_API_KEY in .env file")
        exit(1)
else:
    print(f"   ❌ Failed: {response.text}")
    exit(1)

# Test 2: Generate explainability summary only
print("\n2️⃣ POST /api/summary/explainability-only")
payload = {
    "shap_explanation": {
        "global_feature_importance": {
            "feature_names": ["credit_score", "income", "loan_amount", "age", "debt_to_income"],
            "importance_scores": [0.45, 0.32, 0.23, 0.15, 0.12]
        }
    }
}

response = requests.post(f"{base_url}/api/summary/explainability-only", json=payload)
print(f"   Status: {response.status_code}")

if response.status_code == 200:
    result = response.json()
    print(f"   ✅ Summary Generated:")
    print(f"   {result['summary'][:200]}...")
else:
    print(f"   ❌ Failed: {response.text}")

# Test 3: Generate bias summary only
print("\n3️⃣ POST /api/summary/bias-only")
bias_data = {
    "sensitive_attribute": "gender",
    "groups": ["Male", "Female"],
    "overall_metrics": {
        "disparate_impact": 0.67,
        "demographic_parity_difference": 0.20,
        "equal_opportunity_difference": 0.15,
        "accuracy_parity_difference": 0.05
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

response = requests.post(f"{base_url}/api/summary/bias-only", json=bias_data)
print(f"   Status: {response.status_code}")

if response.status_code == 200:
    result = response.json()
    print(f"   ✅ Bias Summary Generated:")
    print(f"\n{result['summary']}")
else:
    print(f"   ❌ Failed: {response.text}")

# Test 4: Generate combined summary
print("\n" + "=" * 60)
print("\n4️⃣ POST /api/summary (Combined)")
combined_payload = {
    "shap_explanation": {
        "global_feature_importance": {
            "feature_names": ["credit_score", "income", "loan_amount"],
            "importance_scores": [0.45, 0.32, 0.23]
        }
    },
    "bias_results": bias_data
}

response = requests.post(f"{base_url}/api/summary", json=combined_payload)
print(f"   Status: {response.status_code}")

if response.status_code == 200:
    result = response.json()
    print(f"   ✅ Status: {result['status']}")
    
    if result.get('explainability_summary'):
        print(f"\n   📊 Explainability Summary:")
        print(f"   {result['explainability_summary']}")
    
    if result.get('bias_summary'):
        print(f"\n   ⚖️  Bias Summary:")
        print(f"   {result['bias_summary']}")
    
    if result.get('recommendations'):
        print(f"\n   💡 Recommendations:")
        print(f"   {result['recommendations']}")
else:
    print(f"   ❌ Failed: {response.text}")

# Test 5: End-to-end workflow
print("\n" + "=" * 60)
print("\n5️⃣ End-to-End Workflow Test")
print("   (Explain → Bias → Summary)")

# Step 1: Get explainability
print("\n   Step 1: Getting SHAP/LIME explanations...")
explain_payload = {
    "data": [
        {
            "age": 35,
            "income": 75000,
            "loan_amount": 25000,
            "credit_score": 720,
            "employment_years": 8,
            "debt_to_income": 0.25
        }
    ],
    "explainer_type": "shap"
}

explain_response = requests.post(f"{base_url}/api/explain", json=explain_payload)
if explain_response.status_code == 200:
    print("   ✅ Explanations obtained")
    explain_result = explain_response.json()
else:
    print(f"   ❌ Explain failed: {explain_response.status_code}")
    explain_result = None

# Step 2: Get bias metrics
print("\n   Step 2: Computing bias metrics...")
bias_payload = {
    "data": [
        {
            "age": 35,
            "income": 75000,
            "loan_amount": 25000,
            "credit_score": 720,
            "employment_years": 8,
            "debt_to_income": 0.25,
            "gender": "Male",
            "true_label": 1
        },
        {
            "age": 28,
            "income": 45000,
            "loan_amount": 15000,
            "credit_score": 650,
            "employment_years": 3,
            "debt_to_income": 0.45,
            "gender": "Female",
            "true_label": 0
        },
        {
            "age": 42,
            "income": 95000,
            "loan_amount": 35000,
            "credit_score": 780,
            "employment_years": 15,
            "debt_to_income": 0.20,
            "gender": "Male",
            "true_label": 1
        },
        {
            "age": 30,
            "income": 55000,
            "loan_amount": 20000,
            "credit_score": 680,
            "employment_years": 5,
            "debt_to_income": 0.35,
            "gender": "Female",
            "true_label": 1
        }
    ],
    "sensitive_attr": "gender",
    "true_label_col": "true_label"
}

bias_response = requests.post(f"{base_url}/api/bias", json=bias_payload)
if bias_response.status_code == 200:
    print("   ✅ Bias metrics computed")
    bias_result = bias_response.json()
else:
    print(f"   ❌ Bias failed: {bias_response.status_code}")
    bias_result = None

# Step 3: Generate combined summary
print("\n   Step 3: Generating LLM summary...")
if explain_result and bias_result:
    summary_payload = {
        "shap_explanation": explain_result.get("shap_explanation"),
        "bias_results": {
            "sensitive_attribute": bias_result["sensitive_attribute"],
            "groups": bias_result["groups"],
            "overall_metrics": bias_result["overall_metrics"],
            "group_metrics": bias_result["group_metrics"]
        }
    }
    
    summary_response = requests.post(f"{base_url}/api/summary", json=summary_payload)
    
    if summary_response.status_code == 200:
        print("   ✅ Summary generated")
        summary_result = summary_response.json()
        
        print("\n" + "=" * 60)
        print("📊 COMPLETE ANALYSIS REPORT")
        print("=" * 60)
        
        if summary_result.get('explainability_summary'):
            print("\n🔍 EXPLAINABILITY:")
            print(summary_result['explainability_summary'])
        
        if summary_result.get('bias_summary'):
            print("\n⚖️  FAIRNESS ANALYSIS:")
            print(summary_result['bias_summary'])
        
        if summary_result.get('recommendations'):
            print("\n💡 RECOMMENDATIONS:")
            print(summary_result['recommendations'])
        
    else:
        print(f"   ❌ Summary failed: {summary_response.text}")
else:
    print("   ⚠️  Skipping summary - previous steps failed")

print("\n" + "=" * 60)
print("✅ Phase 3 Testing Complete!")
print("=" * 60)
print("\nAll Phase 3 endpoints tested:")
print("  • GET /api/summary/status")
print("  • POST /api/summary")
print("  • POST /api/summary/bias-only")
print("  • POST /api/summary/explainability-only")
print("  • End-to-end workflow (Explain → Bias → Summary)")