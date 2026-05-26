# wer_only.py — run this separately on existing outputs
import whisper
import numpy as np
import csv, os

model = whisper.load_model("base")

SENTENCES = [
    "Hello, how are you doing today?",
    "The weather is beautiful this morning.",
    "I would like to order a coffee please.",
    "Can you tell me the time right now?",
    "She sells seashells by the seashore.",
    "The quick brown fox jumps over the lazy dog.",
    "Peter Piper picked a peck of pickled peppers.",
    "How much wood would a woodchuck chuck?",
    "To be or not to be, that is the question.",
    "All that glitters is not gold.",
    "A journey of a thousand miles begins with a single step.",
    "The early bird catches the worm.",
    "Actions speak louder than words.",
    "Better late than never.",
    "Every cloud has a silver lining.",
    "I am going to the market to buy vegetables.",
    "Please send me the report by tomorrow morning.",
    "The meeting has been scheduled for three o'clock.",
    "We need to finish this project before the deadline.",
    "Could you please help me with this task?",
    "Technology is changing the way we live and work.",
    "Artificial intelligence is transforming every industry.",
    "Machine learning models require large amounts of data.",
    "Deep learning has revolutionized computer vision.",
    "Natural language processing enables human-machine interaction.",
    "The sun rises in the east and sets in the west.",
    "Water boils at one hundred degrees Celsius.",
    "The human body contains approximately thirty-seven trillion cells.",
    "Mount Everest is the highest peak in the world.",
    "The Amazon River is the largest river by discharge.",
    "I enjoy reading books on science and philosophy.",
    "Music has the power to change your mood instantly.",
    "Exercise regularly to maintain good health and fitness.",
    "A balanced diet includes proteins, carbohydrates, and fats.",
    "Sleep is essential for memory consolidation and recovery.",
    "The project deadline is approaching very quickly.",
    "We should review the code before the final submission.",
    "Please make sure to save your work frequently.",
    "The database needs to be updated with new records.",
    "Security vulnerabilities must be patched immediately.",
    "One two three four five six seven eight nine ten.",
    "The year two thousand and twenty five was eventful.",
    "She was born on the fifteenth of August.",
    "The price increased by twenty three percent last quarter.",
    "There are approximately eight billion people on Earth.",
    "Good morning everyone, welcome to today's presentation.",
    "Thank you for your patience and understanding.",
    "I sincerely apologize for the inconvenience caused.",
    "Congratulations on your outstanding achievement.",
    "Have a wonderful day and take care of yourself.",
    "The cat sat on the mat near the window.",
    "Birds fly south during the cold winter months.",
    "Children love playing in the park on sunny days.",
    "The old library had thousands of interesting books.",
    "Fresh fruits and vegetables are good for health.",
    "The engineer designed a new bridge over the river.",
    "Scientists discovered a new species in the rainforest.",
    "The chef prepared a delicious meal for the guests.",
    "Students gathered in the hall for the annual event.",
    "The pilot landed the aircraft safely on the runway.",
    "Innovation drives progress in modern society.",
    "Collaboration leads to better outcomes in teamwork.",
    "Patience is a virtue that everyone should cultivate.",
    "Creativity is the foundation of artistic expression.",
    "Knowledge is power when applied with wisdom.",
    "The conference call starts at nine in the morning.",
    "Please review the attached document and provide feedback.",
    "The server is down and needs immediate attention.",
    "Our team successfully completed the sprint goals.",
    "The client was satisfied with the final deliverable.",
    "Can we schedule a meeting for later this week?",
    "The temperature dropped significantly overnight.",
    "Heavy rainfall is expected throughout the weekend.",
    "The storm caused widespread power outages in the city.",
    "Rescue teams worked tirelessly to help those affected.",
    "Emergency services responded quickly to the situation.",
    "The stock market showed strong gains this week.",
    "Investors are optimistic about the economic recovery.",
    "The company reported record profits for the quarter.",
    "New regulations will impact the financial sector.",
    "Consumer spending increased significantly last month.",
    "The film received critical acclaim from reviewers.",
    "The author published her third novel this year.",
    "The concert was attended by thousands of fans.",
    "The art exhibition showcased works from local artists.",
    "The museum opened a new wing dedicated to history.",
    "She spoke confidently during the entire presentation.",
    "He carefully analyzed the data before concluding.",
    "They collaborated effectively to solve the problem.",
    "The team worked late into the night to meet the deadline.",
    "Everyone contributed their best efforts to the project.",
    "In the morning I wake up and have breakfast.",
    "After lunch I usually take a short walk outside.",
    "In the evening the family gathers for dinner.",
    "Before sleeping I like to read for thirty minutes.",
    "On weekends we visit family and friends nearby.",
    "This is a test of the voice cloning system.",
    "The system should reproduce my voice accurately.",
    "Voice cloning technology has advanced significantly.",
    "This evaluation will measure the quality of cloning.",
    "Thank you for using the VoiceForge cloning system.",
]

OUTPUT_DIR = "evaluation/outputs"

def compute_wer(ref, hyp):
    r, h = ref.lower().split(), hyp.lower().split()
    d = [[0]*(len(h)+1) for _ in range(len(r)+1)]
    for i in range(len(r)+1): d[i][0] = i
    for j in range(len(h)+1): d[0][j] = j
    for i in range(1,len(r)+1):
        for j in range(1,len(h)+1):
            d[i][j] = d[i-1][j-1] if r[i-1]==h[j-1] else 1+min(d[i-1][j],d[i][j-1],d[i-1][j-1])
    return d[len(r)][len(h)] / len(r)

wers, precs, recs, f1s = [], [], [], []

for i, sentence in enumerate(SENTENCES):
    path = f"{OUTPUT_DIR}/sample_{i+1:03d}.wav"
    if not os.path.exists(path):
        continue
    result = model.transcribe(path, language="en")
    hyp = result["text"].strip().lower()
    ref = sentence.strip().lower()

    wer = compute_wer(ref, hyp)
    ref_w, hyp_w = set(ref.split()), set(hyp.split())
    tp = len(ref_w & hyp_w)
    p = tp / (len(hyp_w) + 1e-8)
    r = tp / (len(ref_w) + 1e-8)
    f1 = 2*p*r/(p+r+1e-8)

    wers.append(wer); precs.append(p); recs.append(r); f1s.append(f1)
    print(f"[{i+1:3d}] WER={wer:.3f} P={p:.3f} R={r:.3f} F1={f1:.3f} | {hyp[:50]}")

print("\n── FINAL METRICS ──────────────────────")
print(f"WER       : {np.mean(wers):.4f}  (lower is better)")
print(f"Accuracy  : {1-np.mean(wers):.4f}")
print(f"Precision : {np.mean(precs):.4f}")
print(f"Recall    : {np.mean(recs):.4f}")
print(f"F1 Score  : {np.mean(f1s):.4f}")