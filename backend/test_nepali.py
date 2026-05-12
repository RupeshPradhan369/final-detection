import requests
import json

text = "अहिलेको सरकार अमेरिकाको कठपुतली हो : मोहनविक्रम सिंह पिहलेका सत्ताधारी पनि अमेरिकापरस्त थिए, तर चीन विरोधी थिएनन्"

# Test debug endpoint
response = requests.post(
    'http://127.0.0.1:8000/api/debug-nepali/',
    json={'text': text},
    headers={'Content-Type': 'application/json'}
)
print("DEBUG NEPALI:")
print(json.dumps(response.json(), ensure_ascii=False, indent=2))

print("\n" + "="*50 + "\n")

# Test predict endpoint
response2 = requests.post(
    'http://127.0.0.1:8000/api/predict/',
    json={'text': text},
    headers={'Content-Type': 'application/json'}
)
print("PREDICT:")
print(json.dumps(response2.json(), ensure_ascii=False, indent=2))

import feedparser
import unicodedata

print("\nTesting Online Khabar RSS directly:")
feed = feedparser.parse('https://www.onlinekhabar.com/feed')
for entry in feed.entries[:10]:
    title = entry.get('title', '')
    print(f"  Title: {title}")
    
    # Check word overlap
    article_words = set(
        unicodedata.normalize('NFC', w) 
        for w in text.split() if len(w) > 2
    )
    headline_words = set(
        unicodedata.normalize('NFC', w) 
        for w in title.split() if len(w) > 2
    )
    common = article_words & headline_words
    if common:
        print(f"  ✅ Common words: {common}")


text2 = "सरकारी कार्यालय र शैक्षिक संस्थाहरूमा शनिबार र आइतबार बिदा पेट्रोलियम पदार्थमा उत्पन्न असहज अवस्थाका कारण सरकारी कार्यालय र शैक्षिक संस्थाहरूमा शनिबार र आइतबार बिदा दिने निर्णय गरिएको सरकारका प्रवक्ता सिस्मित पोखरेलले जानकारी दिए"

response3 = requests.post(
    'http://127.0.0.1:8000/api/predict/',
    json={'text': text2},
    headers={'Content-Type': 'application/json'}
)
import json
data = response3.json()
print("Unified score:", data['unified_score'])
print("RSS score:", data['rss_verification']['rss_score'])
print("Matched sources:")
for s in data['rss_verification']['matched_sources']:
    print(f"  - {s['source']} | is_credible_nepali: {s.get('is_credible_nepali', 'MISSING')}")