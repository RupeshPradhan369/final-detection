import torch
import numpy as np
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .ml_model import ModelSingleton
from .models import (
    Article,
    ClassificationResult,
    APIVerification,
    RSSVerification,
    LIMEExplanation
)
import requests
import feedparser
from keybert import KeyBERT
from django.conf import settings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import unicodedata

# ─── Initialize KeyBERT once ─────────────────────────────────────────────────
kw_model = KeyBERT()

# ─── RSS Cache ────────────────────────────────────────────────────────────────
_rss_cache = {}
_rss_cache_time = {}
RSS_CACHE_DURATION = 600  # 10 minutes

def get_cached_feed(feed_url):
    now = time.time()
    if (feed_url in _rss_cache and
            now - _rss_cache_time.get(feed_url, 0) < RSS_CACHE_DURATION):
        return _rss_cache[feed_url]
    feed = feedparser.parse(feed_url)
    _rss_cache[feed_url] = feed
    _rss_cache_time[feed_url] = now
    return feed

# ─── Credible Nepali Sources ──────────────────────────────────────────────────
# Fix 1: Added ronbpost, ratopati full variants, and more Nepali sources
CREDIBLE_NEPALI_SOURCES = [
    # Already existing
    'online khabar', 'onlinekhabar',
    'kathmandu post',
    'setopati',
    'nepal khabar', 'nepalkhabar',
    'gorkhapatra',
    'ratopati',
    'myrepublica', 'republica',
    'the himalayan times', 'himalayan',
    'rising nepal',

    # Newly confirmed
    'nagarik', 'nagarik dainik',          # ✅
    'bbc nepali', 'bbc news nepali',       # ✅
    'nepali times',                        # ✅
    'naya patrika', 'nayapatrika',         # ✅
    'baahrakhari',                         # ✅
    'lokantar', 'lokaantar',               # ✅
    'rajdhani', 'rajdhani daily',          # ✅
    'kantipur', 'ekantipur',               # ✅
    'makalu khabar', 'makalukhabar',       # ✅
    'osnepal',                             # ✅
    'ronb', 'ronbpost',                    # ⚠️ keep — widely known
]
# ─── RSS Feed List ────────────────────────────────────────────────────────────
# Fix 2: Added ronbpost and more Nepali RSS feeds
RSS_FEEDS = [
    # ── International (confirmed working) ─────────────────────────────────────
    'https://feeds.bbci.co.uk/news/rss.xml',
    'https://feeds.bbci.co.uk/news/world/rss.xml',
    'https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml',
    'https://www.theguardian.com/world/rss',
    'http://rss.cnn.com/rss/edition.rss',
    'https://www.npr.org/rss/rss.php?id=1001',
    'https://www.aljazeera.com/xml/rss/all.xml',
    'https://rss.dw.com/rdf/rss-en-all',
    'https://feeds.skynews.com/feeds/rss/world.xml',
    'https://www.independent.co.uk/news/world/rss',
    'https://feeds.washingtonpost.com/rss/world',

    # ── Nepali (all confirmed working from FeedSpot) ──────────────────────────
    'https://kathmandupost.com/rss',                      # ✅ Kathmandu Post
    'https://www.onlinekhabar.com/feed',                  # ✅ Online Khabar
    'https://www.setopati.com/feed',                      # ✅ Setopati
    'https://www.nepalkhabar.com/feed',                   # ✅ Nepal Khabar
    'https://nagariknews.nagariknetwork.com/feed',        # ✅ Nagarik Dainik
    'https://www.ratopati.com/feed',                      # ✅ Ratopati
    'https://feeds.bbci.co.uk/nepali/rss.xml',           # ✅ BBC Nepali
    'https://www.nepalitimes.com/feed/',                  # ✅ Nepali Times
    'https://nayapatrikadaily.com/feed',                  # ✅ Naya Patrika
    'https://baahrakhari.com/feed',                       # ✅ Baahrakhari
    'https://lokaantar.com/feed',                         # ✅ Lokantar
    'https://rajdhanidaily.com/feed/',                    # ✅ Rajdhani Daily
    'https://news24nepal.tv/feed/',                       # ✅ News24 Nepal
    'https://makalukhabar.com/feed',                      # ✅ Makalu Khabar
    'https://www.osnepal.com/feed',                       # ✅ OSNepal
    'https://ekantipur.com/rss',                          # ✅ Kantipur/ekantipur
]


# ─── Language Detection ───────────────────────────────────────────────────────
def detect_language(text):
    """Detect if text is Nepali or English."""
    is_nepali = bool(re.search(r'[\u0900-\u097F]', text))
    return 'ne' if is_nepali else 'en'


# ─── Keyword Extraction ───────────────────────────────────────────────────────
def get_keywords(text, n=3):
    is_nepali = bool(re.search(r'[\u0900-\u097F]', text))

    if is_nepali:
        # Fix 3: Increased word length filter from 3 to 4
        # to avoid generic short words being used as keywords
        words = text.split()
        keywords = [w.strip('।,?!\' ') for w in words if len(w) > 4]
        seen = set()
        result = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                result.append(kw)
            if len(result) >= n:
                break
        return result
    else:
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(2, 3),
            stop_words='english',
            top_n=n,
            diversity=0.9
        )
        generic_words = {
            'start', 'thursday', 'friday', 'monday', 'tuesday',
            'wednesday', 'saturday', 'sunday', 'year', 'time',
            'people', 'said', 'says', 'make', 'made', 'take'
        }
        filtered = []
        for kw in keywords:
            word = kw[0].lower()
            words_in_kw = set(word.split())
            if not words_in_kw.issubset(generic_words):
                filtered.append(kw[0])
        return filtered[:3]


# ─── Google Fact Check API ────────────────────────────────────────────────────
def check_google_factcheck(keywords):
    if not settings.GOOGLE_FACT_CHECK_API_KEY:
        return {'found': False, 'results': []}
    try:
        query = ' '.join(keywords)
        if not query.strip():
            return {'found': False, 'results': []}

        queries_to_try = [query, keywords[0] if keywords else '']
        all_results = []

        for q in queries_to_try:
            if not q.strip():
                continue
            url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
            params = {
                'query': q,
                'key': settings.GOOGLE_FACT_CHECK_API_KEY,
                'pageSize': 5
            }
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            claims = data.get('claims', [])

            for claim in claims[:3]:
                review = claim.get('claimReview', [{}])[0]
                claim_text = claim.get('text', '').lower()

                # Fix 4: Check if API result is actually relevant
                # to the submitted article keywords
                query_words = set(q.lower().split())
                claim_words = set(claim_text.split())
                overlap = query_words & claim_words

                # Skip result if no keyword overlap with the claim
                # (prevents completely unrelated fact checks showing up)
                if len(query_words) > 2 and len(overlap) < 1:
                    continue

                result = {
                    'claim': claim.get('text', ''),
                    'publisher': review.get(
                        'publisher', {}
                    ).get('name', ''),
                    'rating': review.get('textualRating', ''),
                    'url': review.get('url', '')
                }
                if result not in all_results:
                    all_results.append(result)
            if all_results:
                break

        return {'found': len(all_results) > 0, 'results': all_results[:3]}
    except Exception as e:
        return {'found': False, 'results': [], 'error': str(e)}


# ─── RSS Feed Verification ────────────────────────────────────────────────────
def check_rss_feeds(keywords, original_text=""):
    is_nepali = bool(re.search(r'[\u0900-\u097F]', original_text))
    matched_sources = []

    def check_single_feed(feed_url):
        try:
            feed = get_cached_feed(feed_url)
            feed_title = feed.feed.get('title', '').lower()
            is_credible_nepali = any(
                s in feed_title for s in CREDIBLE_NEPALI_SOURCES
            )
            best_match = None
            best_score = 0

            for entry in feed.entries[:30]:
                title = entry.get('title', '')
                summary = entry.get('summary', '')
                combined = (title + ' ' + summary).strip()

                if not combined:
                    continue

                if is_nepali:
                    def normalize(t):
                        return unicodedata.normalize('NFC', t.strip())

                    # Fix 5: Increased word length filter from 2 to 3
                    # to avoid short common words causing false matches
                    article_words = set(
                        normalize(w) for w in original_text.split()
                        if len(w) > 3
                    )
                    headline_words = set(
                        normalize(w) for w in combined.split()
                        if len(w) > 3
                    )
                    common_words = article_words & headline_words

                    # Fix 6: Stricter matching thresholds
                    # AND relevance ratio check to prevent unrelated matches
                    min_match = (
                        4 if len(original_text.split()) < 30 else 6
                    )
                    # Common words must be at least 8% of article words
                    relevance_ratio = len(common_words) / max(
                        len(article_words), 1
                    )

                    if (len(common_words) >= min_match
                            and relevance_ratio >= 0.08):
                        score = len(common_words)
                        if score > best_score:
                            best_score = score
                            best_match = {
                                'source': feed.feed.get(
                                    'title', feed_url
                                ),
                                'title': title,
                                'link': entry.get('link', ''),
                                'match_score': score,
                                'is_credible_nepali': is_credible_nepali
                            }
                else:
                    # English: TF-IDF cosine similarity (unchanged)
                    try:
                        vectorizer = TfidfVectorizer()
                        tfidf = vectorizer.fit_transform(
                            [original_text.lower(), combined.lower()]
                        )
                        score = cosine_similarity(
                            tfidf[0:1], tfidf[1:2]
                        )[0][0]
                        if score > 0.30 and score > best_score:
                            best_score = score
                            best_match = {
                                'source': feed.feed.get(
                                    'title', feed_url
                                ),
                                'title': title,
                                'link': entry.get('link', ''),
                                'match_score': round(float(score), 3),
                                'is_credible_nepali': is_credible_nepali
                            }
                    except Exception:
                        continue

            return best_match
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(check_single_feed, url): url
            for url in RSS_FEEDS
        }
        for future in as_completed(futures):
            result = future.result()
            if result:
                matched_sources.append(result)

    if matched_sources:
        total_score = sum(
            m.get('match_score', 1) for m in matched_sources
        )
        rss_score = min(total_score / (len(RSS_FEEDS) * 2), 1.0)
    else:
        rss_score = 0.0

    return {
        'matched_sources': matched_sources,
        'rss_score': round(rss_score, 2),
        'coverage': f"{len(matched_sources)}/{len(RSS_FEEDS)} sources"
    }


# ─── Unified Score ────────────────────────────────────────────────────────────
def compute_unified_score(fake_prob, rss_score, api_found,
                          api_results=None, matched_sources=None):
    fake_prob_normalized = fake_prob / 100

    credible_nepali_count = 0
    if matched_sources:
        for source in matched_sources:
            if source.get('is_credible_nepali', False):
                credible_nepali_count += 1

    api_component = 0.5
    if api_found and api_results:
        true_ratings = ['true', 'correct', 'accurate', 'verified']
        false_ratings = [
            'false', 'fake', 'misleading',
            'distorts', 'no evidence', 'pinocchio'
        ]
        for result in api_results:
            rating = result.get('rating', '').lower()
            if any(r in rating for r in true_ratings):
                api_component = 0.1
            elif any(r in rating for r in false_ratings):
                api_component = 0.9

    # Fix 7: Lowered cap for 2+ credible Nepali sources from 35% to 25%
    # Makes real Nepali news score even lower (more confidently REAL)
    if credible_nepali_count >= 2:
        base_score = (
            fake_prob_normalized * 0.1 + (1 - rss_score) * 0.9
        )
        return round(min(max(base_score * 100, 0), 25), 2)

    # Fix 8: Lowered cap for 1 credible Nepali source from 49% to 40%
    if credible_nepali_count == 1:
        base_score = (
            fake_prob_normalized * 0.15 + (1 - rss_score) * 0.85
        )
        return round(min(max(base_score * 100, 0), 40), 2)

    if rss_score >= 0.3:
        alpha, beta, gamma = 0.3, 0.6, 0.1
    elif rss_score >= 0.15:
        alpha, beta, gamma = 0.4, 0.5, 0.1
    else:
        alpha, beta, gamma = 0.6, 0.3, 0.1

    if not api_found:
        total = alpha + beta
        alpha = alpha / total * (alpha + beta + gamma)
        beta = beta / total * (alpha + beta + gamma)
        gamma = 0

    rss_component = 1 - rss_score
    score = (
        alpha * fake_prob_normalized +
        beta * rss_component +
        gamma * api_component
    )

    return round(min(max(score * 100, 0), 100), 2)


# ─── Predict View ─────────────────────────────────────────────────────────────
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
        # ── Step 1: Detect language ───────────────────────────────────────────
        language = detect_language(text)

        # ── Step 2: Save article to database ─────────────────────────────────
        article = Article.objects.create(
            text=text,
            language=language
        )

        # ── Step 3: Model prediction ──────────────────────────────────────────
        ml = ModelSingleton.get_instance()
        prediction = ml.predict(text)

        # ── Step 4: Extract keywords ──────────────────────────────────────────
        keywords = get_keywords(text)

        # ── Step 5: Google Fact Check API ─────────────────────────────────────
        api_result = check_google_factcheck(keywords)

        # ── Step 6: RSS verification ──────────────────────────────────────────
        rss_result = check_rss_feeds(keywords, text)

        # ── Step 7: Unified score ─────────────────────────────────────────────
        unified_score = compute_unified_score(
            prediction['fake_probability'],
            rss_result['rss_score'],
            api_result['found'],
            api_result.get('results', []),
            rss_result.get('matched_sources', [])
        )

        verdict = 'FAKE' if unified_score > 50 else 'REAL'

        # ── Step 8: Save ClassificationResult to database ─────────────────────
        classification = ClassificationResult.objects.create(
            article=article,
            label=prediction['label'],
            confidence=prediction['confidence'],
            fake_probability=prediction['fake_probability'],
            real_probability=prediction['real_probability'],
            unified_score=unified_score,
            verdict=verdict
        )

        # ── Step 9: Save APIVerification to database ──────────────────────────
        APIVerification.objects.create(
            result=classification,
            found=api_result['found'],
            claims=api_result.get('results', [])
        )

        # ── Step 10: Save RSSVerification to database ─────────────────────────
        RSSVerification.objects.create(
            result=classification,
            matched_sources=rss_result.get('matched_sources', []),
            rss_score=rss_result['rss_score'],
            coverage=rss_result['coverage']
        )

        # ── Step 11: Return response ──────────────────────────────────────────
        return Response({
            'article_id': article.id,
            'prediction': prediction,
            'keywords': keywords,
            'unified_score': unified_score,
            'verdict': verdict,
            'api_verification': api_result,
            'rss_verification': rss_result,
        })

    except Exception as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# ─── Explain View (LIME) ──────────────────────────────────────────────────────
@api_view(['POST'])
def explain(request):
    text = request.data.get('text', '').strip()
    article_id = request.data.get('article_id', None)

    if not text or len(text) < 20:
        return Response(
            {'error': 'Text too short'},
            status=status.HTTP_400_BAD_REQUEST
        )

    try:
        from lime.lime_text import LimeTextExplainer

        is_nepali = bool(re.search(r'[\u0900-\u097F]', text))
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

        if is_nepali:
            explainer = LimeTextExplainer(
                class_names=['REAL', 'FAKE'],
                split_expression=r'\s+',
                bow=True
            )
        else:
            explainer = LimeTextExplainer(
                class_names=['REAL', 'FAKE']
            )

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

        if article_id:
            try:
                classification = ClassificationResult.objects.get(
                    article__id=article_id
                )
                LIMEExplanation.objects.update_or_create(
                    result=classification,
                    defaults={'word_scores': explanation}
                )
            except ClassificationResult.DoesNotExist:
                pass

        return Response({'explanation': explanation})

    except Exception as e:
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# ─── Health Check ─────────────────────────────────────────────────────────────
@api_view(['GET'])
def health_check(request):
    return Response({
        'status': 'ok',
        'message': 'Fake News Detection API is running'
    })


# ─── Debug RSS ────────────────────────────────────────────────────────────────
@api_view(['GET'])
def debug_rss(request):
    results = {}
    for feed_url in RSS_FEEDS:
        try:
            feed = get_cached_feed(feed_url)
            entries = [e.get('title', '') for e in feed.entries[:3]]
            results[feed_url] = {
                'status': 'ok',
                'title': feed.feed.get('title', 'unknown'),
                'entries': entries,
                'cached': feed_url in _rss_cache
            }
        except Exception as e:
            results[feed_url] = {
                'status': 'error',
                'error': str(e)
            }
    return Response(results)


# ─── Debug Nepali ─────────────────────────────────────────────────────────────
@api_view(['POST'])
def debug_nepali(request):
    text = request.data.get('text', '').strip()
    is_nepali = bool(re.search(r'[\u0900-\u097F]', text))
    words = text.split()
    filtered = [w.strip('।,?!\'') for w in words if len(w) > 3]
    return Response({
        'is_nepali': is_nepali,
        'total_words': len(words),
        'words': words[:10],
        'filtered_words': filtered[:10],
        'text_length': len(text)
    })# auto-sync test
# sync test
# test
# test2
