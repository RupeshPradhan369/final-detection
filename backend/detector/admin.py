from django.contrib import admin
from .models import (
    Article,
    ClassificationResult,
    APIVerification,
    RSSVerification,
    LIMEExplanation
)


# ─── Article Admin ────────────────────────────────────────────────────────────
@admin.register(Article)
class ArticleAdmin(admin.ModelAdmin):
    list_display = [
        'id',
        'language',
        'short_text',
        'submitted_at'
    ]
    list_filter = ['language', 'submitted_at']
    search_fields = ['text']
    readonly_fields = ['submitted_at']
    ordering = ['-submitted_at']

    def short_text(self, obj):
        return obj.text[:80] + '...' if len(obj.text) > 80 else obj.text
    short_text.short_description = 'Article Preview'


# ─── ClassificationResult Admin ───────────────────────────────────────────────
@admin.register(ClassificationResult)
class ClassificationResultAdmin(admin.ModelAdmin):
    list_display = [
        'id',
        'article_id',
        'label',
        'verdict',
        'confidence',
        'fake_probability',
        'real_probability',
        'unified_score',
        'created_at'
    ]
    list_filter = ['label', 'verdict', 'created_at']
    search_fields = ['article__text']
    readonly_fields = ['created_at']
    ordering = ['-created_at']

    def article_id(self, obj):
        return obj.article.id
    article_id.short_description = 'Article ID'


# ─── APIVerification Admin ────────────────────────────────────────────────────
@admin.register(APIVerification)
class APIVerificationAdmin(admin.ModelAdmin):
    list_display = [
        'id',
        'result_id',
        'found',
        'claims_count',
        'created_at'
    ]
    list_filter = ['found', 'created_at']
    readonly_fields = ['created_at']
    ordering = ['-created_at']

    def result_id(self, obj):
        return obj.result.id
    result_id.short_description = 'Result ID'

    def claims_count(self, obj):
        return len(obj.claims) if obj.claims else 0
    claims_count.short_description = 'Claims Found'


# ─── RSSVerification Admin ────────────────────────────────────────────────────
@admin.register(RSSVerification)
class RSSVerificationAdmin(admin.ModelAdmin):
    list_display = [
        'id',
        'result_id',
        'rss_score',
        'coverage',
        'matched_count',
        'created_at'
    ]
    list_filter = ['created_at']
    readonly_fields = ['created_at']
    ordering = ['-created_at']

    def result_id(self, obj):
        return obj.result.id
    result_id.short_description = 'Result ID'

    def matched_count(self, obj):
        return len(obj.matched_sources) if obj.matched_sources else 0
    matched_count.short_description = 'Sources Matched'


# ─── LIMEExplanation Admin ────────────────────────────────────────────────────
@admin.register(LIMEExplanation)
class LIMEExplanationAdmin(admin.ModelAdmin):
    list_display = [
        'id',
        'result_id',
        'top_word',
        'word_count',
        'created_at'
    ]
    readonly_fields = ['created_at']
    ordering = ['-created_at']

    def result_id(self, obj):
        return obj.result.id
    result_id.short_description = 'Result ID'

    def top_word(self, obj):
        if obj.word_scores and len(obj.word_scores) > 0:
            return obj.word_scores[0].get('word', '-')
        return '-'
    top_word.short_description = 'Top Word'

    def word_count(self, obj):
        return len(obj.word_scores) if obj.word_scores else 0
    word_count.short_description = 'Words Explained'