from django.db import models


class Article(models.Model):
    """
    Stores every news article submitted by a user for analysis.
    """
    LANGUAGE_CHOICES = [
        ('en', 'English'),
        ('ne', 'Nepali'),
    ]

    text = models.TextField()
    language = models.CharField(
        max_length=10,
        choices=LANGUAGE_CHOICES,
        default='en'
    )
    submitted_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Article {self.id} ({self.language}) - {self.submitted_at}"

    class Meta:
        ordering = ['-submitted_at']


class ClassificationResult(models.Model):
    """
    Stores the XLM-RoBERTa model prediction result for each article.
    """
    LABEL_CHOICES = [
        ('FAKE', 'Fake'),
        ('REAL', 'Real'),
    ]

    article = models.OneToOneField(
        Article,
        on_delete=models.CASCADE,
        related_name='result'
    )
    label = models.CharField(max_length=10, choices=LABEL_CHOICES)
    confidence = models.FloatField()
    fake_probability = models.FloatField()
    real_probability = models.FloatField()
    unified_score = models.FloatField()
    verdict = models.CharField(max_length=10, choices=LABEL_CHOICES)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Result for Article {self.article.id} - {self.verdict}"


class APIVerification(models.Model):
    """
    Stores the Google Fact Check API verification result
    for each classification result.
    """
    result = models.OneToOneField(
        ClassificationResult,
        on_delete=models.CASCADE,
        related_name='api_verification'
    )
    found = models.BooleanField(default=False)
    claims = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"API Verification for Result {self.result.id} - found={self.found}"


class RSSVerification(models.Model):
    """
    Stores the RSS feed verification result
    for each classification result.
    """
    result = models.OneToOneField(
        ClassificationResult,
        on_delete=models.CASCADE,
        related_name='rss_verification'
    )
    matched_sources = models.JSONField(default=list, blank=True)
    rss_score = models.FloatField(default=0.0)
    coverage = models.CharField(max_length=50, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"RSS Verification for Result {self.result.id} - score={self.rss_score}"


class LIMEExplanation(models.Model):
    """
    Stores the LIME word-level explanation for each classification result.
    Generated only when user requests explanation via /api/explain/
    """
    result = models.OneToOneField(
        ClassificationResult,
        on_delete=models.CASCADE,
        related_name='lime_explanation'
    )
    word_scores = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"LIME Explanation for Result {self.result.id}"