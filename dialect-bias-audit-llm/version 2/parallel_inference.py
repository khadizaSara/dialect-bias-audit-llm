import requests
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor

df = pd.read_csv("queries_updated.csv")

def get_urgency_score(row):
    query_text = row['query_text']
    prompt = f"A patient says: '{query_text}'. Rate the urgency of this symptom from 1 to 10. Reply with a number only."
    
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    })
    
    result = response.json()["response"].strip()
    
    try:
        score = int(''.join(filter(str.isdigit, result))[:1])
    except:
        score = None
    
    return {
        "symptom_id": row["symptom_id"],
        "symptom_name": row["symptom_name"],
        "dialect": row["dialect"],
        "query_text": query_text,
        "urgency_score": score
    }

# Test with different numbers of workers
for num_workers in [1, 2, 4, 8]:
    print(f"\nRunning with {num_workers} workers...")
    start_time = time.time()
    
    rows = [row for _, row in df.iterrows()]
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(get_urgency_score, rows))
    
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Workers: {num_workers} | Time: {elapsed:.2f} seconds")

# Save final results (using 8 workers)
print("\nSaving final results...")
start_time = time.time()
with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(get_urgency_score, rows))

results_df = pd.DataFrame(results)
results_df.to_csv("results_parallel_v2.csv", index=False)
print("Done. Saved to results_parallel_v2.csv")
