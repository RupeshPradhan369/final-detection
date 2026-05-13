# views.py

import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

import feedparser
import numpy as np
import requests
import torch
from django.conf import settings
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
    LIMEExplanation
)

# ──────────────────────────────────────────────────────────────────────────────
# INIT
# ──────────────────────────────────────────────────────────────────────────────

kw_model = KeyBERT()

_rss_cache = {}
_rss_cache_time = {}

RSS_CACHE_DURATION = 600  # 10 minutes

SIMILARITY_VECTORIZER = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(2, 4)
)

# ──────────────────────────────────────────────────────────────────────────────
# RSS CACHE
# ──────────────────────────────────────────────────────────────────────────────


def get_cached_feed(feed_url):

    now = time.time()

    if (
        feed_url in _rss_cache and
        now - _rss_cache_time.get(feed_url, 0)
        < RSS_CACHE_DURATION
    ):
        return _rss_cache[feed_url]

    feed = feedparser.parse(feed_url)

    _rss_cache[feed_url] = feed
    _rss_cache_time[feed_url] = now

    return feed


# ──────────────────────────────────────────────────────────────────────────────
# SOURCES
# ──────────────────────────────────────────────────────────────────────────────

CREDIBLE_NEPALI_SOURCES = [
    'online khabar',
    'onlinekhabar',
    'kathmandu post',
    'setopati',
    'nepal khabar',
    'nepalkhabar',
    'gorkhapatra',
    'ratopati',
    'myrepublica',
    'republica',
    'the himalayan times',
    'nagarik',
    'bbc nepali',
    'nepali times',
    'naya patrika',
    'baahrakhari',
    'lokantar',
    'rajdhani',
    'kantipur',
    'ekantipur',
    'makalu khabar',
    'osnepal',
    'ronb',
    'annapurna post',
]

RSS_FEEDS = [

    # INTERNATIONAL
    'https://feeds.bbci.co.uk/news/rss.xml',
    'https://feeds.bbci.co.uk/news/world/rss.xml',
    'https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml',
    'https://www.theguardian.com/world/rss',

    # NEPALI
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
# LANGUAGE DETECTION
# ──────────────────────────────────────────────────────────────────────────────


def detect_language(text):

    is_nepali = bool(
        re.search(r'[\u0900-\u097F]', text)
    )

    return 'ne' if is_nepali else 'en'


# ──────────────────────────────────────────────────────────────────────────────
# CLEAN TEXT
# ──────────────────────────────────────────────────────────────────────────────


def clean_text(text):

    text = unicodedata.normalize('NFC', text)

    text = re.sub(
        r'[^\u0900-\u097Fa-zA-Z0-9\s]',
        ' ',
        text
    )

    text = re.sub(r'\s+', ' ', text)

    return text.strip().lower()


# ──────────────────────────────────────────────────────────────────────────────
# SIMILARITY
# ──────────────────────────────────────────────────────────────────────────────


def compute_similarity(text1, text2):

    try:

        tfidf = SIMILARITY_VECTORIZER.fit_transform([
            clean_text(text1),
            clean_text(text2)
        ])

        score = cosine_similarity(
            tfidf[0:1],
            tfidf[1:2]
        )[0][0]

        return float(score)

    except Exception:
        return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# KEYWORDS
# ──────────────────────────────────────────────────────────────────────────────


def get_keywords(text, n=5):

    is_nepali = detect_language(text) == 'ne'

    if is_nepali:

        nepali_stopwords = {
            'काठमाडौं',
            'महानगरले',
            'लागि',
            'खुलायो',
            'गर्यो',
            'भने',
            'रहेको',
            'छन्',
            'तथा',
            'गरेको',
            'भएको',
            'सँग',
            'मा',
            'को',
            'का',
            'र',
            'छ',
            'हो',
            'गर्न',
            'भएका',
            'गरी',
            'नेपाल',
            'नेपाली',
        }

        words = re.findall(
            r'[\u0900-\u097F]+',
            clean_text(text)
        )

        filtered = [
            w for w in words
            if len(w) > 2 and w not in nepali_stopwords
        ]

        seen = set()
        keywords = []

        for word in filtered:

            if word not in seen:
                seen.add(word)
                keywords.append(word)

            if len(keywords) >= n:
                break

        return keywords

    else:

        try:

            keywords = kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(2, 3),
                stop_words='english',
                top_n=n,
                diversity=0.7
            )

            return [kw[0] for kw in keywords]

        except Exception:
            return []


# ──────────────────────────────────────────────────────────────────────────────
# FACT CHECK API
# ──────────────────────────────────────────────────────────────────────────────


def check_google_factcheck(keywords, original_text=""):

    if not settings.GOOGLE_FACT_CHECK_API_KEY:
        return {'found': False, 'results': []}

    try:

        query = ' '.join(keywords)

        if not query.strip():
            return {'found': False, 'results': []}

        url = (
            "https://factchecktools.googleapis.com/"
            "v1alpha1/claims:search"
        )

        params = {
            'query': query,
            'key': settings.GOOGLE_FACT_CHECK_API_KEY,
            'pageSize': 5
        }

        response = requests.get(
            url,
            params=params,
            timeout=5
        )

        data = response.json()

        claims = data.get('claims', [])

        all_results = []

        TRUSTED_FACTCHECKERS = [
            'AFP',
            'Snopes',
            'PolitiFact',
            'BBC',
            'Reuters',
            'FactCheck.org'
        ]

        for claim in claims:

            review = claim.get(
                'claimReview',
                [{}]
            )[0]

            claim_text = claim.get('text', '')

            similarity = compute_similarity(
                original_text,
                claim_text
            )

            if similarity < 0.35:
                continue

            publisher = review.get(
                'publisher',
                {}
            ).get('name', '')

            trust_bonus = 0

            for trusted in TRUSTED_FACTCHECKERS:

                if trusted.lower() in publisher.lower():
                    trust_bonus += 0.05

            final_score = similarity + trust_bonus

            result = {
                'claim': claim_text,
                'publisher': publisher,
                'rating': review.get(
                    'textualRating',
                    ''
                ),
                'url': review.get('url', ''),
                'similarity': round(similarity, 3),
                'final_score': round(final_score, 3)
            }

            all_results.append(result)

        sorted_results = sorted(
            all_results,
            key=lambda x: x['final_score'],
            reverse=True
        )

        return {
            'found': len(sorted_results) > 0,
            'results': sorted_results[:1]
        }

    except Exception as e:

        return {
            'found': False,
            'results': [],
            'error': str(e)
        }


# ──────────────────────────────────────────────────────────────────────────────
# RSS VERIFICATION
# ──────────────────────────────────────────────────────────────────────────────


def check_rss_feeds(keywords, original_text=""):

    is_nepali = detect_language(original_text) == 'ne'

    matched_sources = []

    def check_single_feed(feed_url):

        try:

            feed = get_cached_feed(feed_url)

            feed_title = feed.feed.get(
                'title',
                ''
            ).lower()

            is_credible_nepali = any(
                source in feed_title
                for source in CREDIBLE_NEPALI_SOURCES
            )

            best_match = None
            best_score = 0

            # only recent entries
            for entry in feed.entries[:10]:

                title = entry.get('title', '')
                summary = entry.get('summary', '')

                combined = f"{title} {summary}"

                score = compute_similarity(
                    original_text,
                    combined
                )

                threshold = 0.38 if is_nepali else 0.45

                if score > threshold and score > best_score:

                    best_score = score

                    best_match = {
                        'source': feed.feed.get(
                            'title',
                            feed_url
                        ),
                        'title': title,
                        'link': entry.get('link', ''),
                        'match_score': round(score, 3),
                        'is_credible_nepali': (
                            is_credible_nepali
                        )
                    }

            return best_match

        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=20) as executor:

        futures = {
            executor.submit(
                check_single_feed,
                url
            ): url
            for url in RSS_FEEDS
        }

        seen_titles = set()

        for future in as_completed(futures):

            try:

                result = future.result(timeout=5)

                if result:

                    title_key = clean_text(
                        result['title']
                    )

                    if title_key not in seen_titles:

                        seen_titles.add(title_key)

                        matched_sources.append(result)

            except Exception:
                continue

    unique_sources = set(
        m['source']
        for m in matched_sources
    )

    rss_score = min(
        len(unique_sources) / 8,
        1.0
    )

    return {
        'matched_sources': matched_sources,
        'rss_score': round(rss_score, 4),
        'coverage': (
            f"{len(unique_sources)}"
            f"/{len(RSS_FEEDS)} sources"
        )
    }


# ──────────────────────────────────────────────────────────────────────────────
# UNIFIED SCORE
# ──────────────────────────────────────────────────────────────────────────────


def compute_unified_score(
    fake_prob,
    rss_score,
    api_found,
    api_results=None,
    matched_sources=None
):

    fake_prob_normalized = fake_prob / 100

    credible_nepali_count = 0

    if matched_sources:

        for source in matched_sources:

            if source.get(
                'is_credible_nepali',
                False
            ):
                credible_nepali_count += 1

    api_component = 0.5

    if api_found and api_results:

        true_ratings = [
            'true',
            'correct',
            'accurate',
            'verified'
        ]

        false_ratings = [
            'false',
            'fake',
            'misleading',
            'distorts',
            'no evidence',
            'scam'
        ]

        for result in api_results:

            rating = result.get(
                'rating',
                ''
            ).lower()

            if any(r in rating for r in true_ratings):
                api_component = 0.1

            elif any(r in rating for r in false_ratings):
                api_component = 0.9

    # Strong trusted source confidence

    if credible_nepali_count >= 2:

        base_score = (
            fake_prob_normalized * 0.1 +
            (1 - rss_score) * 0.9
        )

        return round(
            min(max(base_score * 100, 0), 20),
            2
        )

    if credible_nepali_count == 1:

        base_score = (
            fake_prob_normalized * 0.15 +
            (1 - rss_score) * 0.85
        )

        return round(
            min(max(base_score * 100, 0), 35),
            2
        )

    # fallback weighted scoring

    if rss_score >= 0.4:
        alpha, beta, gamma = 0.25, 0.65, 0.10

    elif rss_score >= 0.2:
        alpha, beta, gamma = 0.40, 0.50, 0.10

    else:
        alpha, beta, gamma = 0.60, 0.30, 0.10

    if not api_found:

        total = alpha + beta

        alpha /= total
        beta /= total

        gamma = 0

    rss_component = 1 - rss_score

    score = (
        alpha * fake_prob_normalized +
        beta * rss_component +
        gamma * api_component
    )

    return round(
        min(max(score * 100, 0), 100),
        2
    )


# ──────────────────────────────────────────────────────────────────────────────
# PREDICT
# ──────────────────────────────────────────────────────────────────────────────


@api_view(['POST'])
def predict(request):

    text = request.data.get(
        'text',
        ''
    ).strip()

    if not text:

        return Response(
            {'error': 'No text provided'},
            status=status.HTTP_400_BAD_REQUEST
        )

    if len(text) < 20:

        return Response(
            {
                'error': (
                    'Text too short. '
                    'Provide a full article.'
                )
            },
            status=status.HTTP_400_BAD_REQUEST
        )

    try:

        language = detect_language(text)

        article = Article.objects.create(
            text=text,
            language=language
        )

        ml = ModelSingleton.get_instance()

        prediction = ml.predict(text)

        keywords = get_keywords(text)

        # only suspicious articles use factcheck api

        should_factcheck = (
            prediction['fake_probability'] > 55
        )

        api_result = (
            check_google_factcheck(
                keywords,
                text
            )
            if should_factcheck
            else {
                'found': False,
                'results': []
            }
        )

        rss_result = check_rss_feeds(
            keywords,
            text
        )

        unified_score = compute_unified_score(
            prediction['fake_probability'],
            rss_result['rss_score'],
            api_result['found'],
            api_result.get('results', []),
            rss_result.get('matched_sources', [])
        )

        verdict = (
            'FAKE'
            if unified_score > 50
            else 'REAL'
        )

        classification = (
            ClassificationResult.objects.create(
                article=article,
                label=prediction['label'],
                confidence=prediction['confidence'],
                fake_probability=(
                    prediction['fake_probability']
                ),
                real_probability=(
                    prediction['real_probability']
                ),
                unified_score=unified_score,
                verdict=verdict
            )
        )

        APIVerification.objects.create(
            result=classification,
            found=api_result['found'],
            claims=api_result.get('results', [])
        )

        RSSVerification.objects.create(
            result=classification,
            matched_sources=(
                rss_result['matched_sources']
            ),
            rss_score=rss_result['rss_score'],
            coverage=rss_result['coverage']
        )

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


# ──────────────────────────────────────────────────────────────────────────────
# EXPLAIN (LIME)
# ──────────────────────────────────────────────────────────────────────────────


@api_view(['POST'])
def explain(request):

    text = request.data.get(
        'text',
        ''
    ).strip()

    article_id = request.data.get(
        'article_id',
        None
    )

    if not text or len(text) < 20:

        return Response(
            {'error': 'Text too short'},
            status=status.HTTP_400_BAD_REQUEST
        )

    try:

        from lime.lime_text import LimeTextExplainer

        is_nepali = (
            detect_language(text) == 'ne'
        )

        ml = ModelSingleton.get_instance()

        def predict_proba(texts):

            results = []

            for t in texts:

                inputs = ml.tokenizer(
                    t,
                    truncation=True,
                    max_length=256,
                    padding='max_length',
                    return_tensors='pt'
                )

                input_ids = inputs[
                    'input_ids'
                ].to(ml.device)

                attention_mask = inputs[
                    'attention_mask'
                ].to(ml.device)

                with torch.no_grad():

                    outputs = ml.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )

                    probs = torch.softmax(
                        outputs.logits,
                        dim=1
                    )

                results.append(
                    probs[0].cpu().numpy()
                )

            return np.array(results)

        explainer = LimeTextExplainer(
            class_names=['REAL', 'FAKE'],
            split_expression=(
                r'\s+' if is_nepali else None
            ),
            bow=is_nepali
        )

        exp = explainer.explain_instance(
            text,
            predict_proba,
            num_features=10,
            num_samples=100
        )

        explanation = [
            {
                'word': word,
                'weight': round(float(weight), 4)
            }
            for word, weight in exp.as_list()
        ]

        if article_id:

            try:

                classification = (
                    ClassificationResult.objects.get(
                        article__id=article_id
                    )
                )

                LIMEExplanation.objects.update_or_create(
                    result=classification,
                    defaults={
                        'word_scores': explanation
                    }
                )

            except ClassificationResult.DoesNotExist:
                pass

        return Response({
            'explanation': explanation
        })

    except Exception as e:

        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# ──────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ──────────────────────────────────────────────────────────────────────────────


@api_view(['GET'])
def health_check(request):

    return Response({
        'status': 'ok',
        'message': 'Fake News Detection API is running'
    })


# ──────────────────────────────────────────────────────────────────────────────
# DEBUG RSS
# ──────────────────────────────────────────────────────────────────────────────


@api_view(['GET'])
def debug_rss(request):

    results = {}

    for feed_url in RSS_FEEDS:

        try:

            feed = get_cached_feed(feed_url)

            entries = [
                e.get('title', '')
                for e in feed.entries[:3]
            ]

            results[feed_url] = {
                'status': 'ok',
                'title': feed.feed.get(
                    'title',
                    'unknown'
                ),
                'entries': entries,
                'cached': (
                    feed_url in _rss_cache
                )
            }

        except Exception as e:

            results[feed_url] = {
                'status': 'error',
                'error': str(e)
            }

    return Response(results)