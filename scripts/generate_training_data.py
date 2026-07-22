import json
from pathlib import Path
import random

DATA_FILE = Path("data/semantic_examples.json")

def generate_examples():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    examples = data.get("examples", [])
    
    # Count existing
    from collections import Counter
    counts = Counter(ex.get("primary_family") for ex in examples)
    
    templates = {
        "coding": ["Write a {lang} script to {task}", "Debug this {lang} code: {task}", "How to build {task} in {lang}?", "Create a REST API for {task}"],
        "mathematics": ["Solve this equation: {task}", "What is the derivative of {task}?", "Prove that {task}", "Calculate the probability of {task}"],
        "translation": ["Translate '{task}' to {lang}", "How do you say {task} in {lang}?", "English to {lang}: {task}", "Translate this document to {lang}"],
        "summarization": ["Summarize this {doc}: {task}", "Give me a summary of {task}", "TLDR for this {doc}", "Condense the main points of {task}"],
        "extraction": ["Extract all {entity} from this text", "Find the {entity} in the email", "List all {entity} mentioned", "Parse the {entity} out of this document"],
        "creative": ["Write a {doc} about {task}", "Create a story featuring {task}", "Draft a blog post on {task}", "Compose a poem about {task}"],
        "chat": ["Tell me about {task}", "What is {task}?", "How does {task} work?", "Give me advice on {task}"],
        "document_qa": ["According to the document, what is {task}?", "What does the text say about {task}?", "Answer from the PDF: {task}", "Find the section about {task}"],
        "reasoning": ["If {task}, then what?", "Solve this logic puzzle: {task}", "Analyze the situation: {task}", "What can we infer about {task}?"],
        "agent": ["Execute a workflow to {task}", "Run an autonomous agent for {task}", "Use tools to accomplish {task}", "Set up a multi-step task to {task}"]
    }
    
    fillers = {
        "lang": ["Python", "JavaScript", "C++", "Rust", "Go", "Java", "Ruby", "TypeScript"],
        "task": ["the system", "a user login", "data processing", "the universe", "quantum mechanics", "a marketing campaign", "the latest news", "the provided text", "John Doe", "the error message"],
        "doc": ["article", "book", "email", "report", "paper", "memo"],
        "entity": ["dates", "names", "phone numbers", "prices", "locations", "keywords"]
    }
    
    for family, tmpl_list in templates.items():
        needed = 50 - counts.get(family, 0)
        if needed > 0:
            for _ in range(needed):
                tmpl = random.choice(tmpl_list)
                text = tmpl.format(
                    lang=random.choice(fillers["lang"]),
                    task=random.choice(fillers["task"]),
                    doc=random.choice(fillers["doc"]),
                    entity=random.choice(fillers["entity"])
                )
                examples.append({
                    "text": text,
                    "primary_family": family,
                    "domain": "general",
                    "risk_tier": "low",
                    "complexity": "low",
                    "risk_type": "standard",
                    "document_type": "text",
                    "expected_output": "free_text",
                    "decomposition_needed": False,
                    "needs_verification": False
                })
                
    data["examples"] = examples
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    generate_examples()
    print("Done generating training data.")
