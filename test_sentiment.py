"""
Quick script to test sentiment analysis on your own text.
Run from project root: python test_sentiment.py
"""

import sys
sys.path.insert(0, 'backend')

from app.models import analyze_sentiment, analyze_batch

# Test single text
print("=" * 60)
print("SINGLE TEXT ANALYSIS")
print("=" * 60)

text = "Tesla stock surged 15% on strong earnings beat"
result = analyze_sentiment(text, return_all_scores=True)

print(f"\nText: {text}")
print(f"Label: {result['label']}")
print(f"Confidence: {result['score']:.2%}")
print(f"\nAll scores:")
for label, score in result['scores'].items():
    print(f"  {label:>8}: {score:.2%}")

# Test batch
print("\n" + "=" * 60)
print("BATCH ANALYSIS")
print("=" * 60)

texts = [
    "Market crashes amid recession fears",
    "Fed maintains interest rates unchanged",
    "Tech stocks rally on AI optimism",
    "Company announces massive layoffs",
    "Earnings meet Wall Street expectations"
]

results = analyze_batch(texts)

print(f"\nAnalyzed {len(results)} texts:\n")
for text, result in zip(texts, results):
    sentiment = result['label'].upper()
    score = result['score']
    print(f"[{sentiment:>8}] {score:.2%} - {text}")

# Your own text
print("\n" + "=" * 60)
print("TRY YOUR OWN TEXT")
print("=" * 60)
print("\nUncomment the lines below and add your own text:")
print()
print("# my_text = \"Your financial text here...\"")
print("# my_result = analyze_sentiment(my_text)")
print("# print(f\"Sentiment: {my_result['label']} ({my_result['score']:.2%})\")")
