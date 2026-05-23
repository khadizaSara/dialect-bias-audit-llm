import requests
import pandas as pd
import time

df = pd.read_csv("queries.csv")

def get_urgency_score(query_text):
    prompt = f"A patient says: '{query_text}'. Rate the urgency of this symptom from 1 to 10. Reply with a number only, nothing else."
    
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    })
    
    result = response.json()["response"].strip()
    
    try:
        score = int(''.join(filter(str.isdigit, result))[:1])
        return score
    except:
        return None

print("Starting serial inference...")
start_time = time.time()

scores = []
for index, row in df.iterrows():
    print(f"Processing {index+1}/60: {row['symptom_name']} - {row['dialect']}")
    score = get_urgency_score(row['query_text'])
    scores.append(score)

end_time = time.time()
serial_time = end_time - start_time

df['urgency_score'] = scores
df.to_csv("results_serial.csv", index=False)

print(f"\nDone!")
print(f"Serial time: {serial_time:.2f} seconds")
print("Results saved to results_serial.csv")