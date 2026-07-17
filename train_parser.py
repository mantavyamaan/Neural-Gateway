import os
import random
import sys

from app.core.database import init_db, add_feedback

def generate_training_data():
    """Generates 1000 highly diverse real-world test cases across all families."""
    
    templates = {
        "image_generation": [
            "generate an image of {subject} {action}",
            "create a picture showing {subject} in a {style} style",
            "draw a {style} artwork of {subject}",
            "make a highly detailed 4k render of {subject} {action}",
            "i need a midjourney style prompt for {subject}",
            "design a logo for a {subject} company",
            "sketch a {subject} doing {action}",
            "flux generate a {style} portrait of {subject}"
        ],
        "video_generation": [
            "generate a short video of {subject} {action}",
            "create a cinematic video showing {subject} in {style}",
            "animate a {subject} {action}",
            "make a sora video of {subject}",
            "runway gen-2 prompt for {subject} {action}",
            "create a looping animation of {subject}"
        ],
        "coding": [
            "write a python script to {action}",
            "how do i fix this bug in my {style} code?",
            "create a react component for a {subject}",
            "explain how this SQL query works",
            "optimize this algorithm for {subject}"
        ],
        "reasoning": [
            "what is the philosophical meaning of {subject}?",
            "think step by step to solve this logic puzzle about {subject}",
            "analyze the ethical implications of {action}"
        ],
        "chat": [
            "hello, how are you today?",
            "tell me a joke about {subject}",
            "what is the best way to {action}?",
            "give me advice on {subject}"
        ]
    }
    
    subjects = ["a dog", "a futuristic city", "a software engineer", "a spaceship", "a medieval knight", "a cute cat", "a sports car", "an alien landscape", "a database", "a business", "artificial intelligence", "a chef", "a magical forest"]
    actions = ["playing soccer", "flying through space", "coding a website", "fighting a dragon", "baking a cake", "running fast", "calculating math", "exploring a dungeon", "eating pizza"]
    styles = ["cyberpunk", "watercolor", "photorealistic", "pixel art", "anime", "minimalist", "gothic", "abstract", "cinematic"]
    
    cases = []
    
    # Generate exactly 1000 cases
    target_count = 1000
    families = list(templates.keys())
    
    for _ in range(target_count):
        family = random.choice(families)
        template = random.choice(templates[family])
        
        prompt = template.format(
            subject=random.choice(subjects),
            action=random.choice(actions),
            style=random.choice(styles)
        )
        
        # Add some noise to make it realistic
        if random.random() > 0.8:
            prompt = prompt.upper()
        elif random.random() > 0.8:
            prompt = prompt + " please"
        
        cases.append((prompt, family))
        
    return cases

if __name__ == "__main__":
    print("🚀 Initializing Neural Gateway Memory Bank...")
    init_db()
    
    print("🧠 Generating 1000 diverse real-world edge cases...")
    training_data = generate_training_data()
    
    print("💾 Injecting into Dynamic Memory Loop...")
    inserted = 0
    for prompt, family in training_data:
        try:
            add_feedback(prompt, family)
            inserted += 1
        except Exception as e:
            pass
            
    print(f"✅ Successfully trained Neural Gateway with {inserted} new edge cases!")
    print("The local LLM Parser will now dynamically sample from this memory bank on every request.")
