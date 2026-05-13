import logging
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

import feedparser
import numpy as np
import requests
import torch
from django.conf import settings
from django.core.cache import cache
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from keybert import KeyBERT

from .ml_model import ModelSingleton
from .models import (
    Article,
    ClassificationResult,
    APIVerification,
    RSSVerification,
    LIMEExplanation,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

RSS_CACHE_DURATION   = getattr(settings, 'RSS_CACHE_DURATION', 600)   # seconds
RSS_SIMILARITY_NEPALI = getattr(settings, 'RSS_SIM_NEPALI', 0.32)
RSS_SIMILARITY_OTHER  = getattr(settings, 'RSS_SIM_OTHER',  0.42)
FACTCHECK_THRESHOLD   = getattr(settings, 'FACTCHECK_THRESHOLD', 55)  # fake_prob %
FACTCHECK_MIN_SIM     = getattr(settings, 'FACTCHECK_MIN_SIM', 0.25)

NEPALI_STOPWORDS = frozenset({
    'काठमाडौं', 'महानगरले', 'लागि', 'खुलायो', 'गर्यो', 'भने',
    'रहेको', 'छन्', 'तथा', 'गरेको', 'भएको', 'सँग', 'मा', 'को',
    'का', 'र', 'छ', 'हो', 'गर्न', 'भएका', 'गरी',
})

CREDIBLE_NEPALI_SOURCES = frozenset({
    'online khabar', 'onlinekhabar', 'kathmandu post', 'setopati',
    'nepal khabar', 'nepalkhabar', 'gorkhapatra', 'ratopati',
    'ratopati.com', 'myrepublica', 'republica', 'the himalayan times',
    'himalayan times', 'himalayan', 'rising nepal', 'nagarik',
    'bbc nepali', 'nepali times', 'naya patrika', 'nayapatrika',
    'baahrakhari', 'lokantar', 'lokaantar', 'rajdhani', 'kantipur',
    'ekantipur', 'makalu khabar', 'makalukhabar', 'osnepal',
    'ronb', 'ronbpost', 'annapurna post', 'annapurnapost',
})

# Ratings that indicate a claim is verified true / false
_TRUE_RATINGS  = frozenset({'true', 'correct', 'accurate', 'verified'})
_FALSE_RATINGS = frozenset({'false', 'fake', 'misleading', 'distorts', 'no evidence'})

RSS_FEEDS = [
    # International
    'https://feeds.bbci.co.uk/news/rss.xml',
    'https://feeds.bbci.co.uk/news/world/rss.xml',
    'https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml',
    'https://www.theguardian.com/world/rss',
    'http://rss.cnn.com/rss/edition.rss',
    'https://www.npr.org/rss/rss.php?id=1001',
    'https://www.aljazeera.com/xml/rss/all.xml',
    # Nepali
    'https://kathmandupost.com/rss',
    'https://www.onlinekhabar.com/feed',
    'https://www.setopati.com/feed',
    'https://www.nepalkhabar.com/feed',
    'https://nagariknews.nagariknetwork.com/feed',
    'https://www.ratopati.com/feed',
    'https://feeds.bbci.co.uk/nepali/rss.xml',
    'https://www.nepalitimes.com/feed/',
    'https://nayapatrikadaily.com/feed',
    'https://baahrakhari.com/feed',
    'https://lokaantar.com/feed',
    'https://rajdhanidaily.com/feed/',
    'https://news24nepal.tv/feed/',
    'https://makalukhabar.com/feed',
    'https://www.osnepal.com/feed',
    'https://ekantipur.com/rss',
]

# ──────────────────────────────────────────────────────────────────────────────
# LAZY SINGLETONS
# ──────────────────────────────────────────────────────────────────────────────

_kw_model: KeyBERT | None = None


def get_kw_model() -> KeyBERT:
    """Return a lazily-initialised KeyBERT instance (one per process)."""
    global _kw_model
    if _kw_model is None:
        _kw_model = KeyBERT()
    return _kw_model


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _is_nepali(text: str) -> bool:
    return bool(re.search(r'[\u0900-\u097F]', text))


def detect_language(text: str) -> str:
    return 'ne' if _is_nepali(text) else 'en'


def clean_text(text: str) -> str:
    """Normalise, strip punctuation, collapse whitespace, lowercase."""
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'[^\u0900-\u097Fa-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


def _tfidf_similarity(text_a: str, text_b: str) -> float:
    """Return char-level TF-IDF cosine similarity between two strings."""
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
    try:
        tfidf = vectorizer.fit_transform([text_a, text_b])
        return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
    except ValueError:
        # Happens when one or both strings are empty after vectorisation
        return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# KEYWORDS
# ──────────────────────────────────────────────────────────────────────────────

def get_keywords(text: str, n: int = 5) -> list[str]:
    """
    Extract the top-n keywords from *text*.

    Nepali: frequency-ranked unique tokens after stopword removal.
    English: KeyBERT multi-word phrases with MMR diversity.
    """
    if _is_nepali(text):
        words = re.findall(r'[\u0900-\u097F]+', text)
        filtered = [w for w in words if len(w) > 2 and w not in NEPALI_STOPWORDS]

        # Rank by frequency so the most prominent tokens come first
        freq: dict[str, int] = {}
        for w in filtered:
            freq[w] = freq.get(w, 0) + 1

        seen: set[str] = set()
        ranked = []
        for w in sorted(freq, key=freq.get, reverse=True):
            if w not in seen:
                seen.add(w)
                ranked.append(w)
            if len(ranked) >= n:
                break
        return ranked

    kw = get_kw_model().extract_keywords(
        text,
        keyphrase_ngram_range=(2, 3),
        stop_words='english',
        top_n=n,
        use_mmr=True,
        diversity=0.8,
    )
    return [phrase for phrase, _ in kw]


# ──────────────────────────────────────────────────────────────────────────────
# RSS CACHE  (Django cache backend → Redis / Memcache in production)
# ──────────────────────────────────────────────────────────────────────────────

def get_cached_feed(feed_url: str):
    """
    Return a parsed feedparser Feed, using Django's cache backend so
    the data is shared across Gunicorn workers.
    """
    cache_key = f'rss_feed:{feed_url}'
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    feed = feedparser.parse(feed_url)
    cache.set(cache_key, feed, timeout=RSS_CACHE_DURATION)
    return feed


# ──────────────────────────────────────────────────────────────────────────────
# FACT-CHECK API
# ──────────────────────────────────────────────────────────────────────────────

def check_google_factcheck(keywords: list[str], original_text: str = '') -> dict:
    """
    Query the Google Fact Check Tools API and return claims that are
    similar enough to *original_text* (cosine similarity ≥ FACTCHECK_MIN_SIM).
    """
    api_key = getattr(settings, 'GOOGLE_FACT_CHECK_API_KEY', '')
    if not api_key:
        logger.debug('Google Fact Check API key not configured — skipping.')
        return {'found': False, 'results': []}

    query = ' '.join(keywords).strip()
    if not query:
        return {'found': False, 'results': []}

    url = 'https://factchecktools.googleapis.com/v1alpha1/claims:search'
    params = {'query': query, 'key': api_key, 'pageSize': 5}

    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        logger.warning('Fact Check API request failed: %s', exc)
        return {'found': False, 'results': [], 'error': str(exc)}

    cleaned_original = clean_text(original_text)
    all_results = []

    for claim in data.get('claims', []):
        review = claim.get('claimReview', [{}])[0]
        claim_text = clean_text(claim.get('text', ''))

        similarity = _tfidf_similarity(cleaned_original, claim_text)
        if similarity < FACTCHECK_MIN_SIM:
            continue

        all_results.append({
            'claim':      claim.get('text', ''),
            'publisher':  review.get('publisher', {}).get('name', ''),
            'rating':     review.get('textualRating', ''),
            'url':        review.get('url', ''),
            'similarity': round(similarity, 3),
        })

    # Return only the single best match (highest similarity)
    all_results.sort(key=lambda x: x['similarity'], reverse=True)
    return {'found': bool(all_results), 'results': all_results[:1]}


# ──────────────────────────────────────────────────────────────────────────────
# RSS VERIFICATION
# ──────────────────────────────────────────────────────────────────────────────

def _check_single_feed(feed_url: str, cleaned_original: str, is_nepali_text: bool) -> dict | None:
    """
    Check one RSS feed and return the best-matching entry or None.
    Isolated so ThreadPoolExecutor can run it in parallel.
    """
    try:
        feed = get_cached_feed(feed_url)
    except Exception as exc:
        logger.debug('Feed fetch failed (%s): %s', feed_url, exc)
        return None

    feed_title = feed.feed.get('title', '').lower()
    is_credible_nepali = any(src in feed_title for src in CREDIBLE_NEPALI_SOURCES)
    threshold = RSS_SIMILARITY_NEPALI if is_nepali_text else RSS_SIMILARITY_OTHER

    best_match: dict | None = None
    best_score = 0.0

    for entry in feed.entries[:30]:
        combined = clean_text(f"{entry.get('title', '')} {entry.get('summary', '')}")
        if not combined:
            continue

        score = _tfidf_similarity(cleaned_original, combined)

        if score >= threshold and score > best_score:
            best_score = score
            best_match = {
                'source':              feed.feed.get('title', feed_url),
                'title':               entry.get('title', ''),
                'link':                entry.get('link', ''),
                'match_score':         round(score, 3),
                'is_credible_nepali':  is_credible_nepali,
            }

    return best_match


def check_rss_feeds(keywords: list[str], original_text: str = '') -> dict:
    """
    Parallel-scan all RSS_FEEDS and return matched sources + a normalised score.
    Deduplication is done by article link domain to avoid near-duplicate entries.
    """
    is_nepali_text = _is_nepali(original_text)
    cleaned_original = clean_text(original_text)
    matched_sources: list[dict] = []
    seen_links: set[str] = set()

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(_check_single_feed, url, cleaned_original, is_nepali_text): url
            for url in RSS_FEEDS
        }
        for future in as_completed(futures):
            result = future.result()
            if result and result['link'] not in seen_links:
                seen_links.add(result['link'])
                matched_sources.append(result)

    if matched_sources:
        # Credible Nepali sources count 1.5×
        weighted_count = sum(
            1.5 if s.get('is_credible_nepali') else 1.0
            for s in matched_sources
        )
        rss_score = round(min(weighted_count / len(RSS_FEEDS), 1.0), 4)
    else:
        rss_score = 0.0

    return {
        'matched_sources': matched_sources,
        'rss_score':       rss_score,
        'coverage':        f'{len(matched_sources)}/{len(RSS_FEEDS)} sources',
    }


# ──────────────────────────────────────────────────────────────────────────────
# UNIFIED SCORE
# ──────────────────────────────────────────────────────────────────────────────

def compute_unified_score(
    fake_prob: float,
    rss_score: float,
    api_found: bool,
    api_results: list | None = None,
    matched_sources: list | None = None,
) -> float:
    """
    Combine the ML fake-probability, RSS coverage score, and API verdict into
    a single 0–100 unified score (higher → more likely fake).

    Weight strategy
    ───────────────
    • ≥2 credible Nepali sources matched  → trust RSS heavily, cap output at 25
    • 1 credible Nepali source matched    → cap output at 40
    • Fallback: dynamic alpha/beta based on rss_score magnitude
    • API verdict shifts api_component when a clear rating is available
    """
    fake_prob_norm = fake_prob / 100.0
    rss_component  = 1.0 - rss_score

    # ── API component ────────────────────────────────────────────────────────
    api_component = 0.5  # neutral default
    if api_found and api_results:
        for result in api_results:
            rating = result.get('rating', '').lower()
            if any(r in rating for r in _TRUE_RATINGS):
                api_component = 0.1   # independently verified true → reduce fake score
                break
            if any(r in rating for r in _FALSE_RATINGS):
                api_component = 0.9   # independently flagged false → increase fake score
                break

    # ── Credible Nepali source shortcuts ─────────────────────────────────────
    credible_count = sum(
        1 for s in (matched_sources or []) if s.get('is_credible_nepali')
    )

    if credible_count >= 2:
        raw = fake_prob_norm * 0.1 + rss_component * 0.9
        return round(min(max(raw * 100, 0), 25), 2)

    if credible_count == 1:
        raw = fake_prob_norm * 0.15 + rss_component * 0.85
        return round(min(max(raw * 100, 0), 40), 2)

    # ── Dynamic weighted fallback ─────────────────────────────────────────────
    if rss_score >= 0.3:
        alpha, beta, gamma = 0.30, 0.60, 0.10
    elif rss_score >= 0.15:
        alpha, beta, gamma = 0.40, 0.50, 0.10
    else:
        alpha, beta, gamma = 0.60, 0.30, 0.10

    # If no API result, redistribute gamma weight proportionally
    if not api_found:
        total = alpha + beta
        alpha, beta, gamma = alpha / total, beta / total, 0.0

    score = alpha * fake_prob_norm + beta * rss_component + gamma * api_component
    return round(min(max(score * 100, 0), 100), 2)


# ──────────────────────────────────────────────────────────────────────────────
# PREDICT
# ──────────────────────────────────────────────────────────────────────────────

@api_view(['POST'])
def predict(request):
    """
    POST { "text": "<news article>" }

    Returns ML prediction, keyword list, RSS verification, optional fact-check
    API result, and a unified fake-news score with a REAL / FAKE verdict.
    """
    text = request.data.get('text', '').strip()

    if not text:
        return Response(
            {'error': 'No text provided.'},
            status=status.HTTP_400_BAD_REQUEST,
        )
    if len(text) < 20:
        return Response(
            {'error': 'Text too short. Please provide a full news article.'},
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        language = detect_language(text)
        article  = Article.objects.create(text=text, language=language)

        ml         = ModelSingleton.get_instance()
        prediction = ml.predict(text)
        keywords   = get_keywords(text)

        # Only hit the paid Fact Check API when ML is already suspicious
        should_factcheck = prediction['fake_probability'] > FACTCHECK_THRESHOLD
        api_result = (
            check_google_factcheck(keywords, text)
            if should_factcheck
            else {'found': False, 'results': []}
        )

        rss_result    = check_rss_feeds(keywords, text)
        unified_score = compute_unified_score(
            prediction['fake_probability'],
            rss_result['rss_score'],
            api_result['found'],
            api_result.get('results', []),
            rss_result.get('matched_sources', []),
        )
        verdict = 'FAKE' if unified_score > 50 else 'REAL'

        # ── Persist results ──────────────────────────────────────────────────
        classification = ClassificationResult.objects.create(
            article=article,
            label=prediction['label'],
            confidence=prediction['confidence'],
            fake_probability=prediction['fake_probability'],
            real_probability=prediction['real_probability'],
            unified_score=unified_score,
            verdict=verdict,
        )
        APIVerification.objects.create(
            result=classification,
            found=api_result['found'],
            claims=api_result.get('results', []),
        )
        RSSVerification.objects.create(
            result=classification,
            matched_sources=rss_result.get('matched_sources', []),
            rss_score=rss_result['rss_score'],
            coverage=rss_result['coverage'],
        )

        return Response({
            'article_id':       article.id,
            'prediction':       prediction,
            'keywords':         keywords,
            'unified_score':    unified_score,
            'verdict':          verdict,
            'api_verification': api_result,
            'rss_verification': rss_result,
        })

    except Exception as exc:
        logger.exception('Unhandled error in predict view')
        return Response(
            {'error': str(exc)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# ──────────────────────────────────────────────────────────────────────────────
# EXPLAIN  (LIME)
# ──────────────────────────────────────────────────────────────────────────────

@api_view(['POST'])
def explain(request):
    """
    POST { "text": "...", "article_id": <int|null> }

    Returns per-word LIME weights indicating which tokens push the model
    towards REAL (negative weight) or FAKE (positive weight).

    Note: LIME runs ~100 forward passes — expect 5–15 s latency.
    """
    text       = request.data.get('text', '').strip()
    article_id = request.data.get('article_id')

    if not text or len(text) < 20:
        return Response(
            {'error': 'Text too short.'},
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        from lime.lime_text import LimeTextExplainer

        ml = ModelSingleton.get_instance()
        nepali = _is_nepali(text)

        def predict_proba(texts: list[str]) -> np.ndarray:
            """Batch-friendly wrapper for LIME."""
            results = []
            for t in texts:
                inputs = ml.tokenizer(
                    t,
                    truncation=True,
                    max_length=256,
                    padding='max_length',
                    return_tensors='pt',
                )
                with torch.no_grad():
                    outputs = ml.model(
                        input_ids=inputs['input_ids'].to(ml.device),
                        attention_mask=inputs['attention_mask'].to(ml.device),
                    )
                    probs = torch.softmax(outputs.logits, dim=1)
                results.append(probs[0].cpu().numpy())
            return np.array(results)

        explainer = LimeTextExplainer(
            class_names=['REAL', 'FAKE'],
            # Whitespace tokenisation works better for Devanagari
            split_expression=r'\s+' if nepali else None,
            bow=nepali,
        )
        exp = explainer.explain_instance(
            text,
            predict_proba,
            num_features=10,
            num_samples=100,
        )

        explanation = [
            {'word': word, 'weight': round(float(weight), 4)}
            for word, weight in exp.as_list()
        ]

        # Persist to DB if we have an article reference
        if article_id:
            try:
                classification = ClassificationResult.objects.get(
                    article__id=article_id
                )
                LIMEExplanation.objects.update_or_create(
                    result=classification,
                    defaults={'word_scores': explanation},
                )
            except ClassificationResult.DoesNotExist:
                logger.warning('explain: no ClassificationResult for article_id=%s', article_id)

        return Response({'explanation': explanation})

    except Exception as exc:
        logger.exception('Unhandled error in explain view')
        return Response(
            {'error': str(exc)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# ──────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ──────────────────────────────────────────────────────────────────────────────

@api_view(['GET'])
def health_check(request):
    return Response({
        'status':  'ok',
        'message': 'Fake News Detection API is running',
    })


# ──────────────────────────────────────────────────────────────────────────────
# DEBUG RSS  (dev / staff only)
# ──────────────────────────────────────────────────────────────────────────────

@api_view(['GET'])
def debug_rss(request):
    """
    Returns live status of every configured RSS feed.
    Restricted to staff users in non-debug environments.
    """
    if not settings.DEBUG and not getattr(request.user, 'is_staff', False):
        return Response(
            {'error': 'Forbidden'},
            status=status.HTTP_403_FORBIDDEN,
        )

    results = {}
    for feed_url in RSS_FEEDS:
        try:
            feed = get_cached_feed(feed_url)
            results[feed_url] = {
                'status':  'ok',
                'title':   feed.feed.get('title', 'unknown'),
                'entries': [e.get('title', '') for e in feed.entries[:3]],
                'cached':  bool(cache.get(f'rss_feed:{feed_url}')),
            }
        except Exception as exc:
            results[feed_url] = {'status': 'error', 'error': str(exc)}

    return Response(results)