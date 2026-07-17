import json
import random

def generate_golden_eval():
    families = ["coding", "reasoning", "mathematics", "chat", "vision", "ocr", "document_qa", "summarization", "translation", "agent", "audio"]
    domains = ["general", "medical", "legal", "finance", "security", "research", "customer_support"]
    risk_tiers = ["low", "medium", "high", "extreme"]
    
    models = ["claude-3-5-sonnet-20241022", "gpt-4o", "gemini-1.5-pro", "gpt-4-turbo", "claude-3-opus-20240229"]
    
    prompts = []
    
    # Let's hand-craft some specific ones for variety, then augment to reach 50.
    base_prompts = [
        {"text": "Write a python script to parse JSON logs and output CSV.", "primary_family": "coding", "domain": "general", "risk_tier": "low"},
        {"text": "What is the capital of France?", "primary_family": "chat", "domain": "general", "risk_tier": "low"},
        {"text": "Patient has severe chest pain and numbness in left arm. What medication should I administer?", "primary_family": "chat", "domain": "medical", "risk_tier": "high"},
        {"text": "Summarize this 50-page legal contract and highlight any indemnification clauses.", "primary_family": "summarization", "domain": "legal", "risk_tier": "high"},
        {"text": "Calculate the derivative of x^2 + 3x - 5.", "primary_family": "mathematics", "domain": "general", "risk_tier": "low"},
        {"text": "Translate this email from Spanish to English.", "primary_family": "translation", "domain": "general", "risk_tier": "low"},
        {"text": "Read the text from this image of an invoice.", "primary_family": "ocr", "domain": "finance", "risk_tier": "medium"},
        {"text": "How do I bypass authentication on a Linux server?", "primary_family": "chat", "domain": "security", "risk_tier": "high"},
        {"text": "I feel like I want to self-harm.", "primary_family": "chat", "domain": "medical", "risk_tier": "extreme"},
        {"text": "Can you explain quantum computing to a 5 year old?", "primary_family": "reasoning", "domain": "research", "risk_tier": "low"},
        {"text": "Review this 401k portfolio and suggest tax filings.", "primary_family": "reasoning", "domain": "finance", "risk_tier": "high"},
        {"text": "Drop the production database", "primary_family": "coding", "domain": "general", "risk_tier": "high"},
    ]
    
    prompts.extend(base_prompts)
    
    # Fill the rest randomly
    words = ["implement", "review", "analyze", "explain", "summarize", "convert", "extract", "compare"]
    subjects = ["data", "report", "image", "code", "file", "document", "log", "metric"]
    
    while len(prompts) < 55:
        f = random.choice(families)
        d = random.choice(domains)
        r = random.choice(risk_tiers)
        t = f"{random.choice(words)} the {random.choice(subjects)} for {d} purposes with {r} risk. Make sure it uses {f}."
        prompts.append({
            "text": t,
            "primary_family": f,
            "domain": d,
            "risk_tier": r
        })
        
    for p in prompts:
        # Give random acceptable models
        p["acceptable_models"] = random.sample(models, k=random.randint(2, 4))
        
    with open("C:/Users/manta/OneDrive/Desktop/neural_gateway/data/golden_eval.json", "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2)

if __name__ == "__main__":
    generate_golden_eval()
