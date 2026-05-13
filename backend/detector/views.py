import logging
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

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

RSS_CACHE_DURATION    = getattr(settings, 'RSS_CACHE_DURATION', 600)
RSS_SIMILARITY_NEPALI = getattr(settings, 'RSS_SIM_NEPALI', 0.38)
RSS_SIMILARITY_OTHER  = getattr(settings, 'RSS_SIM_OTHER', 0.45)
FACTCHECK_THRESHOLD   = getattr(settings, 'FACTCHECK_THRESHOLD', 55)
FACTCHECK_MIN_SIM     = getattr(settings, 'FACTCHECK_MIN_SIM', 0.35)
RSS_SCORE_DIVISOR     = getattr(settings, 'RSS_SCORE_DIVISOR', 10)

NEPALI_STOPWORDS = frozenset({
    'काठमाडौं', 'महानगरले', 'लागि', 'खुलायो', 'गर्यो', 'भने',
    'रहेको', 'छन्', 'तथा', 'गरेको', 'भएको', 'सँग', 'मा', 'को',
    'का', 'र', 'छ', 'हो', 'गर्न', 'भएका', 'गरी', 'नेपाल', 'नेपाली',
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
    'my republica', 'rising nepal',
})

TRUSTED_FACTCHECKERS = frozenset({
    'afp', 'snopes', 'politifact', 'bbc', 'reuters', 'factcheck.org',
    'altnews', 'boomlive', 'vishvasnews',
})

_TRUE_RATINGS  = frozenset({'true', 'correct', 'accurate', 'verified'})
_FALSE_RATINGS = frozenset({'false', 'fake', 'misleading', 'distorts', 'no evidence', 'scam'})

RSS_FEEDS = [
    # International
    'https://feeds.bbci.co.uk/news/rss.xml',
    'https://feeds.bbci.co.uk/news/world/rss.xml',
    'https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml',
    'https://www.theguardian.com/world/rss',
    'http://rss.cnn.com/rss/edition.rss',
    'https://www.npr.org/rss/rss.php?id=1001',
    'https://www.aljazeera.com/xml/rss/all.xml',
    'https://feeds.reuters.com/reuters/topNews',
    'https://apnews.com/rss',

    # Fact-check
    'https://www.snopes.com/feed/',
    'https://www.politifact.com/rss/all/',
    'https://www.factcheck.org/feed/',

    # South Asia
    'https://www.thehindu.com/feeder/default.rss',
    'https://feeds.feedburner.com/ndtvnews-top-stories',

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
    'https://www.myrepublica.com/feed/',
    'https://thehimalayantimes.com/feed/',
    'https://risingnepaldaily.com/feed',
]

# ──────────────────────────────────────────────────────────────────────────────
# LAZY SINGLETON — KeyBERT
# ──────────────────────────────────────────────────────────────────────────────

_kw_model: KeyBERT | None = None


def get_kw_model() -> KeyBERT:
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
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'[^\u0900-\u097Fa-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


def _tfidf_similarity(text_a: str, text_b: str) -> float:
    """Char-level TF-IDF cosine similarity. Returns 0.0 on any error."""
    try:
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
        tfidf = vectorizer.fit_transform([text_a, text_b])
        return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
    except Exception:
        return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# KEYWORDS
# ──────────────────────────────────────────────────────────────────────────────

def get_keywords(text: str, n: int = 5) -> List[str]:
    """
    Nepali : frequency-ranked unique tokens after stopword removal.
    English: KeyBERT multi-word phrases with MMR diversity.
    """
    if _is_nepali(text):
        words    = re.findall(r'[\u0900-\u097F]+', clean_text(text))
        filtered = [w for w in words if len(w) > 2 and w not in NEPALI_STOPWORDS]

        freq: Dict[str, int] = {}
        for w in filtered:
            freq[w] = freq.get(w, 0) + 1

        seen: set[str] = set()
        ranked: List[str] = []
        for w in sorted(freq, key=lambda x: freq[x], reverse=True):
            if w not in seen:
                seen.add(w)
                ranked.append(w)
            if len(ranked) >= n:
                break
        return ranked

    try:
        kw = get_kw_model().extract_keywords(
            text,
            keyphrase_ngram_range=(2, 3),
            stop_words='english',
            top_n=n,
            use_mmr=True,
            diversity=0.7,
        )
        return [phrase for phrase, _ in kw]
    except Exception:
        return []


# ──────────────────────────────────────────────────────────────────────────────
# RSS CACHE
# ──────────────────────────────────────────────────────────────────────────────

def get_cached_feed(feed_url: str):
    """
    Fetch and cache an RSS feed via Django's cache backend.
    Shared across Gunicorn workers when backed by Redis.
    """
    cache_key = f'rss_feed:{feed_url}'
    cached    = cache.get(cache_key)
    if cached is not None:
        return cached
    feed = feedparser.parse(feed_url)
    cache.set(cache_key, feed, timeout=RSS_CACHE_DURATION)
    return feed


# ──────────────────────────────────────────────────────────────────────────────
# FACT-CHECK API
# ──────────────────────────────────────────────────────────────────────────────

def check_google_factcheck(keywords: List[str], original_text: str = '') -> Dict[str, Any]:
    """
    Query Google Fact Check Tools API.
    Filters by similarity and boosts trusted publisher results.
    """
    api_key = getattr(settings, 'GOOGLE_FACT_CHECK_API_KEY', '')
    if not api_key:
        logger.debug('Google Fact Check API key not configured — skipping.')
        return {'found': False, 'results': []}

    query = ' '.join(keywords).strip()
    if not query:
        return {'found': False, 'results': []}

    try:
        response = requests.get(
            'https://factchecktools.googleapis.com/v1alpha1/claims:search',
            params={'query': query, 'key': api_key, 'pageSize': 5},
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        logger.warning('Fact Check API request failed: %s', exc)
        return {'found': False, 'results': [], 'error': str(exc)}

    cleaned_original = clean_text(original_text)
    all_results      = []

    for claim in data.get('claims', []):
        review     = claim.get('claimReview', [{}])[0]
        claim_text = clean_text(claim.get('text', ''))
        publisher  = review.get('publisher', {}).get('name', '')

        similarity = _tfidf_similarity(cleaned_original, claim_text)
        if similarity < FACTCHECK_MIN_SIM:
            continue

        trust_bonus = 0.05 if any(
            t in publisher.lower() for t in TRUSTED_FACTCHECKERS
        ) else 0.0

        all_results.append({
            'claim':       claim.get('text', ''),
            'publisher':   publisher,
            'rating':      review.get('textualRating', ''),
            'url':         review.get('url', ''),
            'similarity':  round(similarity, 3),
            'final_score': round(similarity + trust_bonus, 3),
        })

    all_results.sort(key=lambda x: x['final_score'], reverse=True)
    return {'found': bool(all_results), 'results': all_results[:1]}


# ──────────────────────────────────────────────────────────────────────────────
# RSS VERIFICATION
# ──────────────────────────────────────────────────────────────────────────────

def _check_single_feed(
    feed_url: str,
    cleaned_original: str,
    is_nepali_text: bool,
) -> Optional[Dict[str, Any]]:
    """Check one RSS feed; return the best-matching entry or None."""
    try:
        feed = get_cached_feed(feed_url)
    except Exception as exc:
        logger.debug('Feed fetch failed (%s): %s', feed_url, exc)
        return None

    feed_title         = feed.feed.get('title', '').lower()
    is_credible_nepali = any(src in feed_title for src in CREDIBLE_NEPALI_SOURCES)
    threshold          = RSS_SIMILARITY_NEPALI if is_nepali_text else RSS_SIMILARITY_OTHER

    best_match: Optional[Dict[str, Any]] = None
    best_score = 0.0

    for entry in feed.entries[:20]:
        combined = clean_text(
            f"{entry.get('title', '')} {entry.get('summary', '')}"
        )
        if not combined:
            continue

        score = _tfidf_similarity(cleaned_original, combined)

        if score >= threshold and score > best_score:
            best_score = score
            best_match = {
                'source':             feed.feed.get('title', feed_url),
                'title':              entry.get('title', ''),
                'link':               entry.get('link', ''),
                'match_score':        round(score, 3),
                'is_credible_nepali': is_credible_nepali,
            }

    return best_match


def check_rss_feeds(keywords: List[str], original_text: str = '') -> Dict[str, Any]:
    """
    Parallel-scan all RSS_FEEDS.
    Score = weighted unique sources / RSS_SCORE_DIVISOR (capped at 1.0).
    Credible Nepali sources count 1.5x.
    """
    is_nepali_text   = _is_nepali(original_text)
    cleaned_original = clean_text(original_text)
    matched_sources: List[Dict[str, Any]] = []
    seen_links: set[str]        = set()

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(
                _check_single_feed, url, cleaned_original, is_nepali_text
            ): url
            for url in RSS_FEEDS
        }
        for future in as_completed(futures):
            try:
                result = future.result(timeout=8)
                if result and result['link'] not in seen_links:
                    seen_links.add(result['link'])
                    matched_sources.append(result)
            except Exception:
                continue

    if matched_sources:
        weighted_count = sum(
            1.5 if s.get('is_credible_nepali') else 1.0
            for s in matched_sources
        )
        rss_score = round(min(weighted_count / RSS_SCORE_DIVISOR, 1.0), 4)
    else:
        rss_score = 0.0

    unique_source_count = len({s['source'] for s in matched_sources})

    return {
        'matched_sources': matched_sources,
        'rss_score':       rss_score,
        'coverage':        f'{unique_source_count}/{len(RSS_FEEDS)} sources',
    }


# ──────────────────────────────────────────────────────────────────────────────
# UNIFIED SCORE
# ──────────────────────────────────────────────────────────────────────────────

def compute_unified_score(
    fake_prob: float,
    rss_score: float,
    api_found: bool,
    api_results: Optional[List[Dict[str, Any]]] = None,
    matched_sources: Optional[List[Dict[str, Any]]] = None,
) -> float:
    """
    Combine ML fake-probability, RSS coverage, and API verdict
    into a single 0–100 score (higher = more likely fake).
    """
    fake_prob_norm = fake_prob / 100.0
    rss_component  = 1.0 - rss_score

    # API component
    api_component = 0.5
    if api_found and api_results:
        for result in api_results:
            rating = result.get('rating', '').lower()
            if any(r in rating for r in _TRUE_RATINGS):
                api_component = 0.1
                break
            if any(r in rating for r in _FALSE_RATINGS):
                api_component = 0.9
                break

    # Credible Nepali source override
    credible_count = sum(
        1 for s in (matched_sources or []) if s.get('is_credible_nepali')
    )

    if credible_count >= 2:
        raw = fake_prob_norm * 0.1 + rss_component * 0.9
        return round(min(max(raw * 100, 0), 20), 2)

    if credible_count == 1:
        raw = fake_prob_norm * 0.15 + rss_component * 0.85
        return round(min(max(raw * 100, 0), 35), 2)

    # Dynamic weighted fallback
    if rss_score >= 0.4:
        alpha, beta, gamma = 0.25, 0.65, 0.10
    elif rss_score >= 0.2:
        alpha, beta, gamma = 0.40, 0.50, 0.10
    else:
        alpha, beta, gamma = 0.60, 0.30, 0.10

    if not api_found:
        total          = alpha + beta
        alpha, beta, gamma = alpha / total, beta / total, 0.0

    score = (
        alpha * fake_prob_norm +
        beta  * rss_component  +
        gamma * api_component
    )
    return round(min(max(score * 100, 0), 100), 2)


# ──────────────────────────────────────────────────────────────────────────────
# PREDICT
# ──────────────────────────────────────────────────────────────────────────────

@api_view(['POST'])
def predict(request):
    """
    POST { "text": "<news article>" }

    Returns ML prediction, keywords, RSS + API verification,
    and a unified REAL / FAKE verdict with credibility score.
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

    low_confidence = len(text.split()) < 15

    try:
        language = detect_language(text)
        article  = Article.objects.create(text=text, language=language)

        ml         = ModelSingleton.get_instance()
        prediction = ml.predict(text)
        keywords   = get_keywords(text)

        should_factcheck = prediction.get('fake_probability', 0) > FACTCHECK_THRESHOLD
        api_result = (
            check_google_factcheck(keywords, text)
            if should_factcheck
            else {'found': False, 'results': []}
        )

        rss_result    = check_rss_feeds(keywords, text)
        unified_score = compute_unified_score(
            prediction.get('fake_probability', 0),
            rss_result['rss_score'],
            api_result.get('found', False),
            api_result.get('results', []),
            rss_result.get('matched_sources', []),
        )
        verdict = 'FAKE' if unified_score > 50 else 'REAL'

        classification = ClassificationResult.objects.create(
            article=article,
            label=prediction.get('label', ''),
            confidence=prediction.get('confidence', 0.0),
            fake_probability=prediction.get('fake_probability', 0.0),
            real_probability=prediction.get('real_probability', 0.0),
            unified_score=unified_score,
            verdict=verdict,
        )
        APIVerification.objects.create(
            result=classification,
            found=api_result.get('found', False),
            claims=api_result.get('results', []),
        )
        RSSVerification.objects.create(
            result=classification,
            matched_sources=rss_result.get('matched_sources', []),
            rss_score=rss_result['rss_score'],
            coverage=rss_result['coverage'],
        )

        response_data = {
            'article_id':       article.id,
            'prediction':       prediction,
            'keywords':         keywords,
            'unified_score':    unified_score,
            'verdict':          verdict,
            'api_verification': api_result,
            'rss_verification': rss_result,
        }
        if low_confidence:
            response_data['warning'] = (
                'Text is very short. Results may be unreliable. '
                'Please provide a full article for accurate analysis.'
            )

        return Response(response_data)

    except Exception as exc:
        logger.exception('Unhandled error in predict view')
        return Response(
            {'error': 'Internal server error.'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# ──────────────────────────────────────────────────────────────────────────────
# EXPLAIN  (LIME)
# ──────────────────────────────────────────────────────────────────────────────

@api_view(['POST'])
def explain(request):
    """
    POST { "text": "...", "article_id": <int|null> }

    Returns per-word LIME weights:
      Positive weight → pushes toward FAKE
      Negative weight → pushes toward REAL

    Notes:
      - Uses a robust predict_proba wrapper that guarantees shape (n,2).
      - Uses a split_expression that handles Nepali and English tokens.
      - Returns a clear error message if explanation fails.
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

        # Determine label ordering from model if possible
        # We want to return probabilities in order [P(REAL), P(FAKE)]
        label_order = None
        try:
            # try common attributes
            if hasattr(ml, 'id2label'):
                id2label = getattr(ml, 'id2label')
                if isinstance(id2label, dict):
                    label_order = [id2label.get(0, '').upper(), id2label.get(1, '').upper()]
            elif hasattr(ml, 'config') and hasattr(ml.config, 'id2label'):
                id2label = ml.config.id2label
                label_order = [id2label.get(0, '').upper(), id2label.get(1, '').upper()]
        except Exception:
            label_order = None

        def predict_proba(texts: List[str]) -> np.ndarray:
            """
            Called by LIME with many perturbed samples.
            MUST return shape (n_samples, 2): [P(REAL), P(FAKE)] per row.
            """
            results: List[np.ndarray] = []
            for t in texts:
                # minimal pre-clean to avoid tokenization surprises
                t_clean = unicodedata.normalize('NFC', t)
                inputs = ml.tokenizer(
                    t_clean,
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
                    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

                # Ensure we have two probabilities
                if probs.size == 2:
                    results.append(probs)
                else:
                    # If model returns different shape, try to coerce or log and fallback
                    logger.error('predict_proba: unexpected probs shape %s for text length %d', probs.shape, len(t))
                    # fallback: normalize whatever we have into two values
                    arr = np.asarray(probs).astype(float)
                    if arr.size == 1:
                        results.append(np.array([1.0 - arr[0], arr[0]]))
                    else:
                        # reduce or pad to 2
                        arr = arr.flatten()
                        if arr.size > 2:
                            arr = arr[:2]
                        elif arr.size < 2:
                            arr = np.pad(arr, (0, 2 - arr.size), constant_values=0.0)
                        s = arr.sum()
                        if s <= 0:
                            results.append(np.array([0.5, 0.5]))
                        else:
                            results.append(arr / s)

            arr = np.array(results)
            # Defensive checks
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] != 2:
                # Try to reorder columns if label_order suggests reversed mapping
                if label_order and label_order == ['FAKE', 'REAL']:
                    arr = arr[:, ::-1]
                else:
                    # As a last resort, normalize rows to sum to 1 and ensure 2 columns
                    try:
                        arr = arr.reshape(arr.shape[0], -1)
                        if arr.shape[1] > 2:
                            arr = arr[:, :2]
                        elif arr.shape[1] < 2:
                            arr = np.pad(arr, ((0, 0), (0, 2 - arr.shape[1])), constant_values=0.0)
                        row_sums = arr.sum(axis=1, keepdims=True)
                        row_sums[row_sums == 0] = 1.0
                        arr = arr / row_sums
                    except Exception as e:
                        logger.exception('predict_proba: cannot coerce probs to shape (n,2): %s', e)
                        # return a neutral distribution to avoid crashing LIME
                        return np.tile(np.array([0.5, 0.5]), (len(texts), 1))

            return arr

        # Use a split_expression that handles Nepali script and English words
        split_expr = r'[\u0900-\u097F]+|\w+'

        explainer = LimeTextExplainer(
            class_names=['REAL', 'FAKE'],
            split_expression=split_expr,
            bow=True,
        )

        # Increase num_samples for more stable explanations; handle timeouts gracefully
        try:
            exp = explainer.explain_instance(
                text,
                predict_proba,
                num_features=15,
                num_samples=500,
                top_labels=1,
            )
        except Exception as e:
            logger.exception('LIME explain_instance failed: %s', e)
            return Response(
                {'error': 'LIME explanation failed. Try a shorter or cleaner article.'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # Extract explanation for the top label (FAKE or REAL)
        try:
            # exp.as_list() returns list of (feature, weight) for the top label by default
            explanation = [
                {'word': word, 'weight': round(float(weight), 4)}
                for word, weight in exp.as_list()
            ]
        except Exception as e:
            logger.exception('Failed to parse LIME explanation: %s', e)
            explanation = []

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
                logger.warning(
                    'explain: no ClassificationResult for article_id=%s',
                    article_id,
                )

        return Response({'explanation': explanation})

    except Exception as exc:
        logger.exception('Unhandled error in explain view')
        return Response(
            {'error': 'Internal server error while generating explanation.'},
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
# DEBUG RSS  (staff / DEBUG only)
# ──────────────────────────────────────────────────────────────────────────────

@api_view(['GET'])
def debug_rss(request):
    """
    Live status of every configured RSS feed.
    Only accessible when DEBUG=True or by staff users.
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
