import torch
import numpy as np
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .ml_model import ModelSingleton
import requests
import feedparser
from keybert import KeyBERT
from django.conf import settings

# Initialize KeyBERT once
kw_model = KeyBERT()

def get_keywords(text, n=3):
    keywords = kw_model.extract_keywords(
        text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=n
    )
    return [kw[0] for kw in keywords]

def check_google_factcheck(keywords):
    if not settings.GOOGLE_FACT_CHECK_API_KEY:
        return {'found': False, 'results': []}
    try:
        query = ' '.join(keywords)
        url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {'query': query, 'key': settings.GOOGLE_FACT_CHECK_API_KEY}
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        claims = data.get('claims', [])
        results = []
        for claim in claims[:3]:
            review = claim.get('claimReview', [{}])[0]
            results.append({
                'claim': claim.get('text', ''),
                'publisher': review.get('publisher', {}).get('name', ''),
                'rating': review.get('textualRating', ''),
                'url': review.get('url', '')
            })
        return {'found': len(results) > 0, 'results': results}
    except Exception as e:
        return {'found': False, 'results': [], 'error': str(e)}

def check_rss_feeds(keywords):
    rss_feeds = [
        'https://feeds.bbci.co.uk/news/rss.xml',
        'https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml',
        'https://www.theguardian.com/world/rss',
        'https://kathmandupost.com/rss',
        'https://feeds.online.khabar24.com/rss/news',
    ]
    matched_sources = []

    individual_words = set()
    for kw in keywords:
        for word in kw.lower().split():
            if len(word) > 4:
                individual_words.add(word)

    for feed_url in rss_feeds:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:30]:
                title = entry.get('title', '').lower()
                summary = entry.get('summary', '').lower()
                combined = title + ' ' + summary
                if any(word in combined for word in individual_words):
                    matched_sources.append({
                        'source': feed.feed.get('title', feed_url),
                        'title': entry.get('title', ''),
                        'link': entry.get('link', '')
                    })
                    break
        except Exception:
            continue

    rss_score = min(len(matched_sources) / len(rss_feeds), 1.0)
    return {
        'matched_sources': matched_sources,
        'rss_score': round(rss_score, 2),
        'coverage': f"{len(matched_sources)}/{len(rss_feeds)} sources"
    }

def compute_unified_score(fake_prob, rss_score, api_found):
    alpha, beta, gamma = 0.5, 0.3, 0.2
    fake_prob_normalized = fake_prob / 100
    rss_component = 1 - rss_score
    api_component = 0.5 if not api_found else 0.3
    if not api_found:
        alpha, beta = 0.6, 0.4
        gamma = 0
    score = (alpha * fake_prob_normalized +
             beta * rss_component +
             gamma * api_component)
    return round(score * 100, 2)

@api_view(['POST'])
def predict(request):
    text = request.data.get('text', '').strip()
    if not text:
        return Response(
            {'error': 'No text provided'},
            status=status.HTTP_400_BAD_REQUEST
        )
    if len(text) < 20:
        return Response(
            {'error': 'Text too short. Please provide a full news article.'},
            status=status.HTTP_400_BAD_REQUEST
        )

    try:
        ml = ModelSingleton.get_instance()
        prediction = ml.predict(text)
        keywords = get_keywords(text)
        api_result = check_google_factcheck(keywords)
        rss_result = check_rss_feeds(keywords)
        unified_score = compute_unified_score(
            prediction['fake_probability'],
            rss_result['rss_score'],
            api_result['found']
        )

        return Response({
            'prediction': prediction,
            'keywords': keywords,
            'unified_score': unified_score,
            'verdict': 'FAKE' if unified_score > 50 else 'REAL',
            'api_verification': api_result,
            'rss_verification': rss_result,
        })

    except Exception as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['POST'])
def explain(request):
    text = request.data.get('text', '').strip()
    if not text or len(text) < 20:
        return Response(
            {'error': 'Text too short'},
            status=status.HTTP_400_BAD_REQUEST
        )

    try:
        from lime.lime_text import LimeTextExplainer

        ml = ModelSingleton.get_instance()

        def predict_proba(texts):
            results = []
            for t in texts:
                inputs = ml.tokenizer(
                    t, truncation=True, max_length=256,
                    padding='max_length', return_tensors='pt'
                )
                input_ids = inputs['input_ids'].to(ml.device)
                attention_mask = inputs['attention_mask'].to(ml.device)
                with torch.no_grad():
                    outputs = ml.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    probs = torch.softmax(outputs.logits, dim=1)
                results.append(probs[0].cpu().numpy())
            return np.array(results)

        explainer = LimeTextExplainer(class_names=['REAL', 'FAKE'])
        exp = explainer.explain_instance(
            text,
            predict_proba,
            num_features=10,
            num_samples=100
        )

        explanation = [
            {'word': word, 'weight': round(float(weight), 4)}
            for word, weight in exp.as_list()
        ]

        return Response({'explanation': explanation})

    except Exception as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
def health_check(request):
    return Response({
        'status': 'ok',
        'message': 'Fake News Detection API is running'
    })

@api_view(['GET'])
def debug_rss(request):
    results = {}
    feeds = [
        'https://feeds.bbci.co.uk/news/rss.xml',
        'https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml',
        'https://www.theguardian.com/world/rss',
        'https://kathmandupost.com/rss',
        'https://feeds.online.khabar24.com/rss/news',
    ]
    for feed_url in feeds:
        try:
            feed = feedparser.parse(feed_url)
            entries = [e.get('title', '') for e in feed.entries[:3]]
            results[feed_url] = {
                'status': 'ok',
                'title': feed.feed.get('title', 'unknown'),
                'entries': entries
            }
        except Exception as e:
            results[feed_url] = {'status': 'error', 'error': str(e)}
    return Response(results)