# app.py — TruthGuard with Gemini API integration
import os
import re
import random
import logging
import json
import threading
import hashlib
from datetime import datetime, timezone, timedelta
from functools import wraps, lru_cache
from logging.handlers import RotatingFileHandler
from sqlalchemy import text, inspect
import numpy as np
import pandas as pd
import nltk
import ssl
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

# --- Google Gemini Configuration ---
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold

    _HAS_GENAI = True
except ImportError:
    genai = None
    _HAS_GENAI = False

_GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
# New (working) setting
_GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')  # Changed to flash for faster responses
_GEMINI_AVAILABLE = False
_GEMINI_MODEL_INSTANCE = None

# Initialize Gemini
if _HAS_GENAI and _GEMINI_API_KEY and _GEMINI_API_KEY != 'your-gemini-api-key-here':
    try:
        genai.configure(api_key=_GEMINI_API_KEY)
        _GEMINI_MODEL_INSTANCE = genai.GenerativeModel(_GEMINI_MODEL)
        _GEMINI_AVAILABLE = True
        print(f"✓ Gemini AI initialized with model: {_GEMINI_MODEL}")
    except Exception as e:
        print(f"⚠ Gemini initialization error: {e}")
        _GEMINI_AVAILABLE = False
else:
    print(
        f"⚠ Gemini API {'key not configured' if not _GEMINI_API_KEY else 'SDK not installed'}. Using rule-based responses.")

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# --- Pre-load NLTK resources at startup ---
print("Pre-loading NLTK resources for immediate response...")
try:
    # Set NLTK data path to avoid re-downloads
    nltk_data_path = os.path.join(os.path.expanduser('~'), 'nltk_data')
    os.makedirs(nltk_data_path, exist_ok=True)
    nltk.data.path.append(nltk_data_path)

    # Download essential packages if missing (non-blocking)
    required_packages = ['punkt', 'stopwords', 'wordnet', 'vader_lexicon']
    for package in required_packages:
        try:
            if package == 'punkt':
                nltk.data.find('tokenizers/punkt')
            elif package == 'vader_lexicon':
                nltk.data.find('sentiment/vader_lexicon')
            else:
                nltk.data.find(f'corpora/{package}')
        except LookupError:
            try:
                nltk.download(package, quiet=True)
                print(f"✓ Downloaded NLTK package: {package}")
            except Exception as e:
                print(f"⚠ Could not download {package}: {e}")

    # Pre-load sentiment analyzer
    from nltk.sentiment import SentimentIntensityAnalyzer

    _ = SentimentIntensityAnalyzer()  # Force pre-load
    print("✓ NLTK resources pre-loaded successfully")

except Exception as e:
    print(f"⚠ NLTK initialization warning: {e}")

from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Flask app config ---
app = Flask(__name__, static_folder='static', template_folder='templates')


# --- Custom Jinja2 Filters ---
@app.template_filter()
def max_filter(a, b):
    """Custom max filter for Jinja2"""
    return max(a, b)


@app.template_filter()
def min_filter(a, b):
    """Custom min filter for Jinja2"""
    return min(a, b)


app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
# SQLite DB inside instance folder
os.makedirs(app.instance_path, exist_ok=True)
db_path = os.path.join(app.instance_path, 'truthguard.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{db_path}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('logs', exist_ok=True)

# --- Cache for immediate response ---
analysis_cache = {}
gemini_cache = {}  # Cache for Gemini responses

# --- Extensions ---
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'


# --- Models ---
class User(UserMixin, db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    avatar = db.Column(db.String(200), default='default.png')
    role = db.Column(db.String(20), default='user')
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    last_login = db.Column(db.DateTime)

    analyses = db.relationship('Analysis', backref='author', lazy=True, cascade='all, delete-orphan')
    chat_history = db.relationship('ChatHistory', backref='author', lazy=True, cascade='all, delete-orphan')


class Analysis(db.Model):
    __tablename__ = 'analyses'
    id = db.Column(db.Integer, primary_key=True)

    # Content & metadata
    title = db.Column(db.Text, default='')
    content = db.Column(db.Text, default='')  # raw text content (for display/search)
    article_data = db.Column(db.JSON, nullable=True)  # optional structured article extraction
    source_url = db.Column(db.Text, nullable=True)  # This should be nullable

    # Classification & scores
    classification = db.Column(db.String(20), nullable=False, default='UNKNOWN')  # RELIABLE, SUSPICIOUS, FAKE, etc.
    confidence_score = db.Column(db.Float, nullable=True)

    # Additional numeric scores for dashboard & analysis
    sentiment_score = db.Column(db.Float, nullable=True)
    sensationalism_score = db.Column(db.Float, nullable=True)
    credibility_score = db.Column(db.Float, nullable=True)

    # JSON/text fields to store structured findings
    key_findings = db.Column(db.Text, default='[]')  # JSON string
    recommendations = db.Column(db.Text, nullable=True)  # JSON string
    fact_checks = db.Column(db.Text, nullable=True)  # JSON string
    analysis_metadata = db.Column(db.Text, default='{}')  # JSON string (e.g. word count, etc.)

    # Quick analysis flag
    is_quick_analysis = db.Column(db.Boolean, default=False)

    # Relationships & timestamps
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc),
                           onupdate=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f'<Analysis {self.id}: {self.classification}>'


class ChatHistory(db.Model):
    __tablename__ = 'chat_history'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user_message = db.Column(db.Text, nullable=False)
    bot_response = db.Column(db.Text, nullable=False)
    message_type = db.Column(db.String(20), default='text')
    session_id = db.Column(db.String(100))
    source = db.Column(db.String(50), default='web')
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


class GeminiChat(db.Model):
    """Store Gemini AI chat conversations"""
    __tablename__ = 'gemini_chat'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)  # Optional for non-logged in users
    session_id = db.Column(db.String(100), nullable=False)
    user_message = db.Column(db.Text, nullable=False)
    gemini_response = db.Column(db.Text, nullable=False)
    model_used = db.Column(db.String(50), default='gemini-2.5-flash')
    tokens_used = db.Column(db.Integer, nullable=True)
    response_time = db.Column(db.Float, nullable=True)  # In seconds
    gemini_metadata = db.Column(db.JSON, default={})  # Renamed from 'metadata' to avoid conflict
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


# --- Gemini AI Assistant Class ---
class GeminiAssistant:
    """Gemini AI Assistant for misinformation detection and fact-checking"""

    def __init__(self):
        self.model = _GEMINI_MODEL_INSTANCE
        self.available = _GEMINI_AVAILABLE
        if self.available:
            print(f"✓ Gemini Assistant initialized with model: {_GEMINI_MODEL}")
        else:
            print("⚠ Gemini Assistant running in fallback mode")

    def generate_response(self, message, context=None, use_cache=True):
        """Generate response using Gemini AI"""
        start_time = datetime.now(timezone.utc)

        # Check cache first
        if use_cache:
            cache_key = hashlib.md5(f"{message}_{context}".encode()).hexdigest()
            if cache_key in gemini_cache:
                cached_response = gemini_cache[cache_key]
                return {
                    'success': True,
                    'response': cached_response['response'],
                    'model': cached_response['model'],
                    'cached': True,
                    'response_time': (datetime.now(timezone.utc) - start_time).total_seconds()
                }

        # Fallback to rule-based if Gemini not available
        if not self.available or not self.model:
            return self._fallback_response(message)

        try:
            # Prepare prompt with context
            prompt = self._build_prompt(message, context)

            # Generate response with safety settings
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }

            response = self.model.generate_content(
                prompt,
                safety_settings=safety_settings,
                generation_config={
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'top_k': 40,
                    'max_output_tokens': 1024,
                }
            )

            response_text = response.text.strip()
            response_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            # Cache the response
            if use_cache:
                gemini_cache[cache_key] = {
                    'response': response_text,
                    'model': _GEMINI_MODEL,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }

            return {
                'success': True,
                'response': response_text,
                'model': _GEMINI_MODEL,
                'cached': False,
                'response_time': response_time
            }

        except Exception as e:
            app.logger.error(f"Gemini API error: {e}")
            return self._fallback_response(message)

    def _build_prompt(self, message, context):
        """Build the prompt for Gemini"""
        system_prompt = """You are TruthGuard AI, an expert assistant for misinformation detection and fact-checking.

Your role:
1. Help users detect fake news and misinformation
2. Provide fact-checking guidance
3. Explain misinformation patterns
4. Suggest verification methods
5. Be helpful, accurate, and concise

Guidelines:
- Focus on practical advice
- Cite reliable sources when possible
- Explain reasoning clearly
- Warn about common misinformation tactics
- Provide actionable steps for verification

Format responses with clear sections using markdown-like formatting."""

        user_context = context or {}
        context_str = f"User context: {json.dumps(user_context)}" if user_context else ""

        return f"""{system_prompt}

{context_str}

User message: {message}

Please provide a helpful, accurate response:"""

    def _fallback_response(self, message):
        """Generate fallback response when Gemini is unavailable"""
        message_lower = message.lower()

        if any(word in message_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            response = "Hello! I'm TruthGuard Assistant. I can help you with misinformation detection and fact-checking."
        elif any(word in message_lower for word in ['fake', 'misinformation', 'hoax', 'rumor']):
            response = """**Fake News Detection Tips:**

1. **Check the Source:**
   • Verify website credibility
   • Look for "About Us" information
   • Check contact details

2. **Examine Content:**
   • Watch for emotional language
   • Check for spelling/grammar errors
   • Look for cited sources

3. **Cross-Verify:**
   • Use fact-checking websites
   • Search for other reports
   • Check dates and context

4. **Use Our Tools:**
   • Try our analysis tool for automated checking
   • Visit our educational resources"""
        elif any(word in message_lower for word in ['analyze', 'check', 'verify', 'scan']):
            response = """**How to Analyze Content:**

1. **Text Analysis:** Paste text into our analyzer
2. **URL Checking:** Enter website URLs
3. **Manual Verification:** Use fact-checking sites

**Recommended Fact-Checkers:**
• Snopes.com
• FactCheck.org
• PolitiFact.com
• Reuters Fact Check

Try our analysis tool for automated checking!"""
        else:
            response = f"I understand you're asking about: {message}\n\nAs your TruthGuard assistant, I specialize in misinformation detection. You can ask me about:\n• Fake news indicators\n• Fact-checking methods\n• Source verification\n• Content analysis\n\nOr use our analysis tool for automated checking."

        return {
            'success': True,
            'response': response,
            'model': 'fallback',
            'cached': False,
            'response_time': 0.1
        }


# Initialize Gemini Assistant
gemini_assistant = GeminiAssistant()


# Add time_ago filter
def time_ago(value):
    now = datetime.now(timezone.utc)
    diff = now - value

    if diff < timedelta(minutes=1):
        return 'just now'
    elif diff < timedelta(hours=1):
        minutes = int(diff.total_seconds() / 60)
        return f'{minutes} minute{"s" if minutes != 1 else ""} ago'
    elif diff < timedelta(days=1):
        hours = int(diff.total_seconds() / 3600)
        return f'{hours} hour{"s" if hours != 1 else ""} ago'
    elif diff < timedelta(days=30):
        days = diff.days
        return f'{days} day{"s" if days != 1 else ""} ago'
    elif diff < timedelta(days=365):
        months = diff.days // 30
        return f'{months} month{"s" if months != 1 else ""} ago'
    else:
        years = diff.days // 365
        return f'{years} year{"s" if years != 1 else ""} ago'


app.jinja_env.filters['time_ago'] = time_ago


# --- ULTRA-FAST Fake News Detector for Immediate Response ---
class FastNewsDetector:
    """Ultra-fast detector for immediate response (target: <100ms)"""

    def __init__(self):
        # Pre-compile patterns for speed
        self.url_pattern = re.compile(r'http\S+|www\S+|https\S+', re.MULTILINE | re.IGNORECASE)
        self.non_word_pattern = re.compile(r'[^a-zA-Z\s]')
        self.year_pattern = re.compile(r'\d{4}')
        self.research_pattern = re.compile(r'\b(research|study|data|evidence|peer.?reviewed)\b', re.IGNORECASE)

        # Indicators (lowercase for case-insensitive matching)
        self.fake_indicators = [
            'breaking news', 'shocking', "you won't believe", 'viral',
            'must read', 'experts say', 'studies show', "they don't want you to know",
            'mainstream media', 'fake news', 'cover-up', 'conspiracy', 'wake up',
            'the truth about', "they're hiding", 'secret', 'forbidden knowledge',
            'this will blow your mind', 'doctors hate this', 'big pharma',
            "government doesn't want you to know", 'mind-blowing', 'exposed',
            "the media won't tell you", 'hidden truth', 'censored'
        ]

        self.credible_indicators = [
            'according to research', 'peer-reviewed', 'study published',
            'data shows', 'statistics indicate', 'official report',
            'government data', 'academic study', 'scientific evidence',
            'clinical trial', 'research findings', 'according to experts'
        ]

        # Load sentiment analyzer once
        try:
            self.sia = SentimentIntensityAnalyzer()
        except:
            self.sia = None

        # Pre-compile indicator patterns
        self.fake_patterns = [re.compile(re.escape(indicator), re.IGNORECASE)
                              for indicator in self.fake_indicators]
        self.credible_patterns = [re.compile(re.escape(indicator), re.IGNORECASE)
                                  for indicator in self.credible_indicators]

    @lru_cache(maxsize=1000)
    def preprocess_text_cached(self, text):
        """Cached text preprocessing"""
        if not text:
            return ""
        # Remove URLs
        text = self.url_pattern.sub('', text)
        # Remove non-alphabetic characters
        text = self.non_word_pattern.sub('', text)
        # Convert to lowercase and normalize whitespace
        text = ' '.join(text.lower().split())
        return text

    def quick_classify(self, text, max_length=5000):
        """ULTRA-fast classification (target: <50ms)"""
        from datetime import datetime

        start_time = datetime.now(timezone.utc)

        if not text or len(text.strip()) < 50:
            return {
                'classification': 'INSUFFICIENT',
                'confidence': 0.0,
                'message': 'Text too short for analysis (minimum 50 characters).',
                'processing_ms': 0
            }

        # Limit text length for speed
        text = text[:max_length]
        text_lower = text.lower()

        # ULTRA-fast indicator counting using regex patterns
        fake_count = 0
        credible_count = 0

        # Use pre-compiled patterns for speed (check only first 12 patterns)
        for pattern in self.fake_patterns[:12]:
            if pattern.search(text_lower):
                fake_count += 1

        for pattern in self.credible_patterns[:12]:
            if pattern.search(text_lower):
                credible_count += 1

        # Fast sentiment analysis (cached, limited to 1000 chars)
        sentiment_score = 0.0
        if self.sia:
            try:
                sentiment = self.sia.polarity_scores(text[:1000])
                sentiment_score = sentiment.get('compound', 0.0)
            except:
                pass

        # Fast feature extraction
        words = text_lower.split()
        word_count = len(words)
        sentences = re.split(r'[.!?]+', text)
        sentence_count = max(1, len(sentences))

        # Count sensational elements
        exclamation_count = text.count('!')
        question_count = text.count('?')
        all_caps_words = sum(1 for word in words if word.isupper() and len(word) > 1)

        # Calculate sensationalism (simplified and faster)
        sensationalism = min(10,
                             fake_count * 0.5 +
                             min(exclamation_count / 3, 3) +
                             min(question_count / 3, 3) +
                             min(all_caps_words / 2, 3)
                             )

        # Calculate credibility (simplified and faster)
        credibility = min(10, 5 +
                          credible_count * 0.8 +
                          (2 if self.year_pattern.search(text) else 0) +
                          (2 if self.research_pattern.search(text_lower) else 0) +
                          (1 if abs(sentiment_score) < 0.3 else -0.5)
                          )

        # Ultra-quick classification logic
        score = 5.0  # Neutral starting point
        score -= sensationalism * 0.3
        score += credibility * 0.4
        score -= fake_count * 0.2
        score -= abs(sentiment_score) * 0.1

        # Normalize to 0-1 scale
        normalized_score = max(0.0, min(1.0, score / 10))

        # Determine classification
        if normalized_score >= 0.7:
            classification = 'RELIABLE'
            confidence = min(0.95, 0.7 + (normalized_score - 0.7) * 0.5)
        elif normalized_score >= 0.4:
            classification = 'SUSPICIOUS'
            confidence = 0.6 + (normalized_score - 0.4) * 0.3
        else:
            classification = 'FAKE'
            confidence = min(0.9, 0.5 + (0.4 - normalized_score) * 0.5)

        # Generate quick findings (limited for speed)
        key_findings = []
        if fake_count > 0:
            key_findings.append({
                'type': 'warning',
                'icon': '⚠️',
                'title': f'Found {fake_count} suspicious phrases',
                'description': 'Content contains language commonly used in misinformation'
            })

        if credible_count > 0:
            key_findings.append({
                'type': 'positive',
                'icon': '✅',
                'title': f'Found {credible_count} credible indicators',
                'description': 'Content includes references to research or data'
            })

        if sensationalism > 6:
            key_findings.append({
                'type': 'danger',
                'icon': '🚨',
                'title': 'High sensationalism',
                'description': 'Content shows exaggerated emotional language'
            })

        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        return {
            'classification': classification,
            'confidence': confidence,
            'processing_ms': round(processing_time, 1),
            'features': {
                'word_count': word_count,
                'sentence_count': sentence_count,
                'fake_indicators': fake_count,
                'credible_indicators': credible_count,
                'sensationalism_score': round(sensationalism, 1),
                'credibility_score': round(credibility, 1),
                'sentiment_compound': round(sentiment_score, 3),
                'exclamation_count': exclamation_count
            },
            'key_findings': key_findings[:2],  # Limit to 2 findings for speed
            'is_quick_analysis': True
        }


# --- URL content extractor with caching ---
@lru_cache(maxsize=100)
def extract_url_content_cached(url):
    """Cached URL content extraction"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)  # Reduced timeout to 5 seconds
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        main_content = soup.find('article') or soup.find('main') or soup.find('div',
                                                                              class_=re.compile(r'content|article|post',
                                                                                                re.I))
        text = main_content.get_text(separator=' ') if main_content else soup.get_text(separator=' ')
        text = ' '.join(text.split())
        title = soup.title.string if soup.title else None
        return {'content': text[:15000], 'title': title, 'success': True}  # Reduced to 15K chars
    except Exception as e:
        app.logger.error(f"Error extracting URL content: {e}")
        return {'content': '', 'title': None, 'success': False, 'error': str(e)}


def extract_url_content(url):
    """Wrapper for backward compatibility"""
    return extract_url_content_cached(url)


# --- Background Database Save Function ---
def save_analysis_background(user_id, content, result, url=None, title=None, is_quick=True):
    """Save analysis to database in background thread"""

    def save_task():
        try:
            # Minimal JSON serialization
            key_findings_json = json.dumps(result.get('key_findings', [])[:2]) if result.get('key_findings') else '[]'

            with app.app_context():
                analysis = Analysis(
                    user_id=user_id,
                    title=title or f"Analysis {datetime.now(timezone.utc).strftime('%H:%M')}",
                    content=content[:5000],  # Save more content to DB
                    source_url=url if url else None,
                    classification=result.get('classification', 'UNKNOWN'),
                    confidence_score=result.get('confidence', 0.0),
                    sentiment_score=result.get('features', {}).get('sentiment_compound'),
                    sensationalism_score=result.get('features', {}).get('sensationalism_score'),
                    credibility_score=result.get('features', {}).get('credibility_score'),
                    key_findings=key_findings_json,
                    is_quick_analysis=is_quick,
                    # Store metadata for history display
                    analysis_metadata=json.dumps({
                        'word_count': result.get('features', {}).get('word_count', 0),
                        'sentence_count': result.get('features', {}).get('sentence_count', 0),
                        'fake_indicators': result.get('features', {}).get('fake_indicators', 0),
                        'credible_indicators': result.get('features', {}).get('credible_indicators', 0),
                        'processing_ms': result.get('processing_ms', 0)
                    })
                )

                db.session.add(analysis)
                db.session.commit()
                app.logger.info(f"Analysis saved in background: {analysis.id}")
        except Exception as e:
            app.logger.error(f"Background save error: {e}")

    # Start background thread
    thread = threading.Thread(target=save_task)
    thread.daemon = True
    thread.start()
    return True


# --- Logging setup ---
def setup_logging():
    file_handler = RotatingFileHandler('logs/truthguard.log', maxBytes=10 * 1024 * 1024, backupCount=10,
                                       encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if app.debug else logging.INFO)

    class NoEmojiFilter(logging.Filter):
        def filter(self, record):
            try:
                msg = str(record.msg)
                emoji_pattern = re.compile("["
                                           u"\U0001F600-\U0001F64F"
                                           u"\U0001F300-\U0001F5FF"
                                           u"\U0001F680-\U0001F6FF"
                                           u"\U0001F1E0-\U0001F1FF"
                                           u"\U00002700-\U000027BF"
                                           "]+", flags=re.UNICODE)
                record.msg = emoji_pattern.sub('', msg)
            except Exception:
                pass
            return True

    console_handler.addFilter(NoEmojiFilter())
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    # Avoid adding handlers multiple times in reloader environments
    if not app.logger.handlers:
        app.logger.addHandler(file_handler)
        app.logger.addHandler(console_handler)

    app.logger.setLevel(logging.DEBUG if app.debug else logging.INFO)
    app.logger.info('TruthGuard application initialized')


# --- Login loader ---
@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(User, int(user_id))
    except Exception:
        return None


# --- Admin decorator ---
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'admin':
            flash('Access denied. Admin privileges required.', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)

    return decorated_function


# --- Context processor for templates ---
@app.context_processor
def inject_gemini_status():
    """Inject Gemini status into all templates"""
    return {
        'gemini_available': _GEMINI_AVAILABLE,
        'gemini_model': _GEMINI_MODEL,
        'gemini_api_key_set': bool(_GEMINI_API_KEY and _GEMINI_API_KEY != 'your-gemini-api-key-here')
    }


# --- Routes & APIs ---
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        name = request.form.get('name', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        if not all([email, name, password, confirm_password]):
            flash('Please fill in all fields', 'danger')
            return render_template('register.html')
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return render_template('register.html')
        if len(password) < 6:
            flash('Password must be at least 6 characters', 'danger')
            return render_template('register.html')
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'danger')
            return render_template('register.html')
        user = User(email=email, name=name, password=generate_password_hash(password, method='pbkdf2:sha256'))
        db.session.add(user)
        db.session.commit()
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        remember = 'remember' in request.form
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            if not user.is_active:
                flash('Account is disabled', 'danger')
                return render_template('login.html')
            login_user(user, remember=remember)
            user.last_login = datetime.now(timezone.utc)
            db.session.commit()
            flash(f'Welcome back, {user.name}!', 'success')
            return redirect(url_for('admin_dashboard') if user.role == 'admin' else url_for('dashboard'))
        flash('Invalid email or password', 'danger')
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))


@app.route('/dashboard')
@login_required
def dashboard():
    try:
        analyses = Analysis.query.filter_by(user_id=current_user.id).order_by(Analysis.created_at.desc()).limit(
            10).all()
    except Exception as e:
        app.logger.error(f"Failed to query analyses: {e}")
        analyses = []

    total = Analysis.query.filter_by(user_id=current_user.id).count()
    reliable = Analysis.query.filter_by(user_id=current_user.id, classification='RELIABLE').count()
    suspicious = Analysis.query.filter_by(user_id=current_user.id, classification='SUSPICIOUS').count()
    fake = Analysis.query.filter_by(user_id=current_user.id, classification='FAKE').count()

    # Calculate average confidence
    avg_confidence = db.session.query(db.func.avg(Analysis.confidence_score)).filter(
        Analysis.user_id == current_user.id).scalar() or 0
    if avg_confidence:
        avg_confidence = avg_confidence * 100  # Convert to percentage

    # Get recent count (last 7 days)
    from datetime import datetime, timedelta, timezone
    week_ago = datetime.now(timezone.utc) - timedelta(days=7)
    recent_count = Analysis.query.filter(
        Analysis.user_id == current_user.id,
        Analysis.created_at >= week_ago
    ).count()

    # Get maximum confidence
    max_confidence_result = db.session.query(db.func.max(Analysis.confidence_score)).filter(
        Analysis.user_id == current_user.id).scalar()
    max_confidence = (max_confidence_result or 0) * 100

    stats = {
        'total': total,
        'reliable': reliable,
        'suspicious': suspicious,
        'fake': fake,
        'reliable_percent': round((reliable / total * 100) if total > 0 else 0, 1),
        'fake_percent': round((fake / total * 100) if total > 0 else 0, 1)
    }

    return render_template('dashboard.html',
                           user=current_user,
                           analyses=analyses,
                           stats=stats,
                           avg_confidence=avg_confidence,
                           recent_count=recent_count,
                           max_confidence=max_confidence)


# --- GEMINI CHAT ENDPOINTS ---
@app.route('/api/gemini/chat', methods=['POST'])
def gemini_chat():
    """Chat endpoint using Gemini AI"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'})

        message = data.get('message', '').strip()
        session_id = data.get('session_id', 'default')

        if not message:
            return jsonify({'success': False, 'error': 'Message cannot be empty'})

        # Get response from Gemini
        response_data = gemini_assistant.generate_response(
            message=message,
            context={
                'user_id': current_user.id if current_user.is_authenticated else None,
                'session_id': session_id,
                'is_authenticated': current_user.is_authenticated,
                'is_admin': current_user.is_authenticated and current_user.role == 'admin'
            }
        )

        # Save to database if user is authenticated
        if current_user.is_authenticated:
            try:
                gemini_chat = GeminiChat(
                    user_id=current_user.id,
                    session_id=session_id,
                    user_message=message,
                    gemini_response=response_data['response'],
                    model_used=response_data['model'],
                    response_time=response_data.get('response_time', 0),
                    gemini_metadata={'cached': response_data.get('cached', False)}
                )
                db.session.add(gemini_chat)
                db.session.commit()
            except Exception as e:
                app.logger.error(f"Failed to save Gemini chat: {e}")

        return jsonify({
            'success': True,
            'response': response_data['response'],
            'model': response_data['model'],
            'cached': response_data.get('cached', False),
            'response_time': response_data.get('response_time', 0),
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

    except Exception as e:
        app.logger.error(f"Gemini chat error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        })


@app.route('/chat', methods=['POST'])
def chat_with_gemini():
    """Main chat endpoint for Gemini AI - works for both authenticated and non-authenticated users"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'})

        message = data.get('message', '').strip()
        session_id = data.get('session_id', 'default')

        if not message:
            return jsonify({'success': False, 'error': 'Message cannot be empty'})

        # Get user context if available
        user_context = {
            'is_logged_in': current_user.is_authenticated,
            'is_admin': current_user.is_authenticated and current_user.role == 'admin' if current_user.is_authenticated else False,
            'user_id': current_user.id if current_user.is_authenticated else None,
            'session_id': session_id
        }

        # Get response from Gemini
        response_data = gemini_assistant.generate_response(
            message=message,
            context=user_context
        )

        # Save to database if user is authenticated
        if current_user.is_authenticated:
            try:
                gemini_chat = GeminiChat(
                    user_id=current_user.id,
                    session_id=session_id,
                    user_message=message,
                    gemini_response=response_data['response'],
                    model_used=response_data['model'],
                    response_time=response_data.get('response_time', 0),
                    gemini_metadata={'cached': response_data.get('cached', False)}
                )
                db.session.add(gemini_chat)
                db.session.commit()
            except Exception as e:
                app.logger.error(f"Failed to save Gemini chat: {e}")

        return jsonify({
            'success': True,
            'response': response_data['response'],
            'model': response_data['model'],
            'cached': response_data.get('cached', False),
            'response_time': response_data.get('response_time', 0),
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

    except Exception as e:
        app.logger.error(f"Chat error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        })


@app.route('/api/chat/simple', methods=['POST'])
def simple_chat():
    """Simple chat endpoint that works without authentication"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'})

        message = data.get('message', '').strip()

        if not message:
            return jsonify({'success': False, 'error': 'Message cannot be empty'})

        # Use Gemini assistant (it has fallback built in)
        response_data = gemini_assistant.generate_response(message)

        return jsonify({
            'success': True,
            'response': response_data['response'],
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'ai_enhanced': response_data['model'] != 'fallback'
        })

    except Exception as e:
        app.logger.error(f"Simple chat error: {e}")
        return jsonify({
            'success': False,
            'error': 'Chat service unavailable',
            'timestamp': datetime.now(timezone.utc).isoformat()
        })


@app.route('/debug/gemini-status')
def debug_gemini_status():
    """Debug endpoint to check Gemini status"""
    return jsonify({
        'has_genai_sdk': _HAS_GENAI,
        'gemini_api_key_configured': bool(_GEMINI_API_KEY and _GEMINI_API_KEY != 'your-gemini-api-key-here'),
        'gemini_api_key_length': len(_GEMINI_API_KEY) if _GEMINI_API_KEY else 0,
        'gemini_available': _GEMINI_AVAILABLE,
        'gemini_model_instance': 'Set' if _GEMINI_MODEL_INSTANCE else 'None',
        'gemini_model': _GEMINI_MODEL,
        'gemini_assistant_available': gemini_assistant.available
    })


@app.route('/api/gemini/status')
def api_gemini_status():
    """API endpoint to check Gemini status"""
    return jsonify({
        'available': _GEMINI_AVAILABLE,
        'model': _GEMINI_MODEL,
        'initialized': bool(_GEMINI_MODEL_INSTANCE),
        'assistant_ready': gemini_assistant.available
    })


@app.route('/api/gemini/analyze', methods=['POST'])
@login_required
def gemini_analyze():
    """Enhanced analysis using Gemini AI"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'})

        content = data.get('content', '').strip()
        url = data.get('url', '').strip()

        if not content and not url:
            return jsonify({'success': False, 'error': 'Please provide content or URL'})

        if url and not content:
            url_result = extract_url_content_cached(url)
            if not url_result['success']:
                return jsonify(
                    {'success': False, 'error': f"Failed to fetch URL: {url_result.get('error', 'Unknown error')}"})
            content = url_result['content']

        if len(content.strip()) < 50:
            return jsonify({'success': False, 'error': 'Content is too short for analysis (minimum 50 characters)'})

        # First get fast analysis
        fast_result = FastNewsDetector().quick_classify(content[:5000])

        # Then get Gemini analysis if available
        gemini_analysis = ""
        if gemini_assistant.available:
            prompt = f"""Analyze this content for misinformation:

Content: {content[:3000]}

Please provide:
1. Fact-checking assessment
2. Credibility indicators
3. Potential misinformation patterns
4. Recommendations for verification

Keep response concise and actionable."""

            gemini_response = gemini_assistant.generate_response(prompt)
            gemini_analysis = gemini_response['response']

        # Save to database
        analysis = Analysis(
            user_id=current_user.id,
            title=f"Gemini Analysis {datetime.now(timezone.utc).strftime('%H:%M')}",
            content=content[:1000],
            source_url=url if url else None,
            classification=fast_result['classification'],
            confidence_score=fast_result['confidence'],
            sentiment_score=fast_result['features'].get('sentiment_compound'),
            sensationalism_score=fast_result['features'].get('sensationalism_score'),
            credibility_score=fast_result['features'].get('credibility_score'),
            key_findings=json.dumps(fast_result.get('key_findings', [])),
            recommendations=json.dumps(['Use Gemini AI for detailed analysis']),
            is_quick_analysis=False,
            analysis_metadata=json.dumps({'gemini_analysis': gemini_analysis[:500] if gemini_analysis else ''})
        )

        db.session.add(analysis)
        db.session.commit()

        return jsonify({
            'success': True,
            'analysis_id': analysis.id,
            'classification': fast_result['classification'],
            'confidence': round(fast_result['confidence'] * 100, 1),
            'gemini_analysis': gemini_analysis,
            'features': fast_result['features'],
            'key_findings': fast_result.get('key_findings', []),
            'is_enhanced': True
        })

    except Exception as e:
        app.logger.error(f"Gemini analysis error: {e}")
        return jsonify({'success': False, 'error': str(e)})


# --- MAIN ANALYZE ENDPOINT ---
@app.route('/analyze', methods=['GET', 'POST'])
@login_required
def analyze():
    """Ultra-fast analysis with immediate response"""

    if request.method == 'GET':
        return render_template('analyze.html')

    start_time = datetime.now(timezone.utc)

    try:
        content = request.form.get('content', '').strip()
        url = request.form.get('url', '').strip()
        title = request.form.get('title', '').strip()

        # Immediate validation response
        if not content and not url:
            return jsonify({
                'success': False,
                'error': 'Please provide text or URL to analyze',
                'processing_ms': 0
            })

        # Extract URL content if provided
        if url and not content:
            url_hash = hashlib.md5(url.encode()).hexdigest()
            if url_hash in analysis_cache:
                url_result = analysis_cache[url_hash]
            else:
                url_result = extract_url_content_cached(url)
                if url_result['success']:
                    analysis_cache[url_hash] = url_result

            if not url_result['success']:
                return jsonify({
                    'success': False,
                    'error': f"Failed to fetch URL: {url_result.get('error', 'Connection timeout')}",
                    'processing_ms': 0
                })

            content = url_result['content']
            if not title and url_result.get('title'):
                title = url_result['title']

        # Check minimum length
        if len(content.strip()) < 50:
            return jsonify({
                'success': False,
                'error': 'Content is too short for analysis (minimum 50 characters)',
                'processing_ms': 0
            })

        # Check cache for identical content
        content_hash = hashlib.md5(content[:3000].encode()).hexdigest()
        if content_hash in analysis_cache:
            result = analysis_cache[content_hash]
            result['cached'] = True
            app.logger.info(f"Cache hit for content hash: {content_hash[:8]}")
        else:
            # Use FAST detector
            result = FastNewsDetector().quick_classify(content[:5000])
            analysis_cache[content_hash] = result
            result['cached'] = False

        if result.get('classification') == 'ERROR':
            return jsonify({
                'success': False,
                'error': result.get('message', 'Analysis failed'),
                'processing_ms': result.get('processing_ms', 0)
            })

        # Calculate processing time
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        # IMMEDIATE RESPONSE
        response_data = {
            'success': True,
            'classification': result.get('classification'),
            'confidence': round(result.get('confidence', 0) * 100, 1),
            'sensationalism': round(result.get('features', {}).get('sensationalism_score', 0), 1),
            'credibility': round(result.get('features', {}).get('credibility_score', 0), 1),
            'word_count': result.get('features', {}).get('word_count', 0),
            'key_findings': result.get('key_findings', [])[:2],
            'processing_ms': round(processing_time, 1),
            'is_quick': True,
            'cached': result.get('cached', False),
            'message': 'Analysis completed in {}ms'.format(round(processing_time, 1)),
            'gemini_available': gemini_assistant.available
        }

        # Save to database in BACKGROUND
        try:
            save_analysis_background(
                user_id=current_user.id,
                content=content[:5000],
                result=result,
                url=url or None,
                title=title or None,
                is_quick=True
            )
        except Exception as db_error:
            app.logger.error(f"Background save failed: {db_error}")

        return jsonify(response_data)

    except Exception as e:
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        app.logger.error(f"Analysis error: {e}")
        return jsonify({
            'success': False,
            'error': f'Analysis failed: {str(e)[:100]}',
            'processing_ms': round(processing_time, 1)
        })


# --- Additional Routes (keeping your existing structure) ---
@app.route('/analysis/<int:analysis_id>')
@login_required
def analysis_detail(analysis_id):
    analysis = Analysis.query.get_or_404(analysis_id)
    if analysis.user_id != current_user.id and current_user.role != 'admin':
        flash('Access denied', 'danger')
        return redirect(url_for('dashboard'))

    try:
        key_findings = json.loads(analysis.key_findings) if analysis.key_findings else []
    except:
        key_findings = []

    try:
        analysis_metadata = json.loads(analysis.analysis_metadata) if analysis.analysis_metadata else {}
    except:
        analysis_metadata = {}

    # Get word count from metadata or calculate it
    word_count = analysis_metadata.get('word_count', len(analysis.content.split()) if analysis.content else 0)

    return render_template('analysis_detail.html',
                           analysis=analysis,
                           key_findings=key_findings,
                           word_count=word_count,
                           analysis_metadata=analysis_metadata)


@app.route('/history')
@login_required
def analysis_history():
    page = request.args.get('page', 1, type=int)
    per_page = 20

    analyses_pagination = Analysis.query.filter_by(user_id=current_user.id) \
        .order_by(Analysis.created_at.desc()) \
        .paginate(page=page, per_page=per_page, error_out=False)

    total_analyses = Analysis.query.filter_by(user_id=current_user.id).count()
    reliable_count = Analysis.query.filter_by(user_id=current_user.id, classification='RELIABLE').count()
    suspicious_count = Analysis.query.filter_by(user_id=current_user.id, classification='SUSPICIOUS').count()
    fake_count = Analysis.query.filter_by(user_id=current_user.id, classification='FAKE').count()

    debug_mode = request.args.get('debug') == 'true'

    return render_template('analysis_history.html',
                           analyses=analyses_pagination.items,
                           pagination=analyses_pagination,
                           total_analyses=total_analyses,
                           reliable_count=reliable_count,
                           suspicious_count=suspicious_count,
                           fake_count=fake_count,
                           debug_mode=debug_mode)


@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        current_password = request.form.get('current_password', '')
        new_password = request.form.get('new_password', '')
        confirm_password = request.form.get('confirm_password', '')
        if name and name != current_user.name:
            current_user.name = name
            flash('Name updated successfully', 'success')
        if current_password and new_password:
            if not check_password_hash(current_user.password, current_password):
                flash('Current password is incorrect', 'danger')
            elif new_password != confirm_password:
                flash('New passwords do not match', 'danger')
            elif len(new_password) < 6:
                flash('Password must be at least 6 characters', 'danger')
            else:
                current_user.password = generate_password_hash(new_password, method='pbkdf2:sha256')
                flash('Password updated successfully', 'success')
        db.session.commit()
        return redirect(url_for('profile'))
    stats = {
        'total_analyses': Analysis.query.filter_by(user_id=current_user.id).count(),
        'reliable_count': Analysis.query.filter_by(user_id=current_user.id, classification='RELIABLE').count(),
        'suspicious_count': Analysis.query.filter_by(user_id=current_user.id, classification='SUSPICIOUS').count(),
        'fake_count': Analysis.query.filter_by(user_id=current_user.id, classification='FAKE').count(),
        'avg_confidence': float(db.session.query(db.func.avg(Analysis.confidence_score)).filter(
            Analysis.user_id == current_user.id).scalar() or 0)
    }
    return render_template('profile.html', user=current_user, stats=stats)


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


# --- Legacy chat endpoint for backward compatibility ---
@app.route('/api/chat/send', methods=['POST'])
@login_required
def chat_send():
    """Legacy chat endpoint - uses Gemini if available"""
    try:
        data = request.get_json() or {}
        message = data.get('message', '').strip()
        session_id = data.get('session_id', str(current_user.id))

        if not message:
            return jsonify({'success': False, 'error': 'Message cannot be empty'})

        # Use Gemini if available, otherwise simple response
        if gemini_assistant.available:
            response_data = gemini_assistant.generate_response(
                message=message,
                context={
                    'user_id': current_user.id,
                    'session_id': session_id,
                    'is_authenticated': True,
                    'is_admin': current_user.role == 'admin'
                }
            )
            response_text = response_data['response']
        else:
            response_text = f"I received: {message}. For AI-powered responses, please configure the Gemini API key in your .env file."

        # Save to database
        chat = ChatHistory(
            user_id=current_user.id,
            user_message=message,
            bot_response=response_text,
            session_id=session_id
        )
        db.session.add(chat)
        db.session.commit()

        return jsonify({
            'success': True,
            'response': response_text,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'chat_id': chat.id,
            'ai_enhanced': gemini_assistant.available
        })

    except Exception as e:
        app.logger.error(f"Chat error: {e}")
        return jsonify({
            'success': True,
            'response': "I'm currently experiencing technical difficulties. Please try again later.",
            'timestamp': datetime.now(timezone.utc).isoformat()
        })


# --- Admin routes ---
@app.route('/admin')
@login_required
@admin_required
def admin_dashboard():
    total_users = User.query.count()
    total_analyses = Analysis.query.count()
    active_users = User.query.filter_by(is_active=True).count()
    recent_users = User.query.order_by(User.created_at.desc()).limit(5).all()
    recent_analyses = Analysis.query.order_by(Analysis.created_at.desc()).limit(10).all()
    classifications = db.session.query(Analysis.classification, db.func.count(Analysis.id)).group_by(
        Analysis.classification).all()

    analyses_per_user = 0
    if total_users > 0:
        analyses_per_user = round(total_analyses / total_users, 1)

    avg_confidence = db.session.query(db.func.avg(Analysis.confidence_score)).scalar() or 0
    if avg_confidence:
        avg_confidence = avg_confidence * 100

    chatbot_ai_enabled = gemini_assistant.available

    stats = {
        'total_users': total_users,
        'total_analyses': total_analyses,
        'active_users': active_users,
        'classifications': dict(classifications),
        'analyses_per_user': analyses_per_user
    }

    return render_template('admin/dashboard.html',
                           stats=stats,
                           recent_users=recent_users,
                           recent_analyses=recent_analyses,
                           avg_confidence=avg_confidence,
                           chatbot_ai_enabled=chatbot_ai_enabled)


@app.route('/admin/users')
@login_required
@admin_required
def admin_users():
    users = User.query.order_by(User.created_at.desc()).all()
    return render_template('admin/users.html', users=users)


@app.route('/admin/analyses')
@login_required
@admin_required
def admin_analyses():
    page = request.args.get('page', 1, type=int)
    per_page = 20
    analyses = Analysis.query.order_by(Analysis.created_at.desc()).paginate(page=page, per_page=per_page,
                                                                            error_out=False)
    return render_template('admin/analyses.html', analyses=analyses)


@app.route('/admin/user/<int:user_id>/toggle', methods=['POST'])
@login_required
@admin_required
def toggle_user_status(user_id):
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        flash('Cannot modify your own account', 'danger')
        return redirect(url_for('admin_users'))
    user.is_active = not user.is_active
    db.session.commit()
    flash(f'User {user.email} has been {"activated" if user.is_active else "deactivated"}', 'success')
    return redirect(url_for('admin_users'))


@app.route('/admin/user/<int:user_id>/make_admin', methods=['POST'])
@login_required
@admin_required
def make_user_admin(user_id):
    user = User.query.get_or_404(user_id)
    user.role = 'admin'
    db.session.commit()
    flash(f'User {user.email} is now an administrator', 'success')
    return redirect(url_for('admin_users'))


@app.route('/analysis/<int:analysis_id>/delete', methods=['POST'])
@login_required
@admin_required
def delete_analysis(analysis_id):
    """Delete an analysis (admin only)"""
    try:
        analysis = Analysis.query.get_or_404(analysis_id)
        db.session.delete(analysis)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Analysis deleted successfully'})
    except Exception as e:
        app.logger.error(f"Error deleting analysis {analysis_id}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/health')
def health_check():
    try:
        db.session.execute('SELECT 1')
        db_status = 'healthy'
    except Exception:
        db_status = 'unhealthy'
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'version': '1.0.0',
        'database': db_status,
        'cache_size': len(analysis_cache),
        'gemini_ai': 'available' if gemini_assistant.available else 'unavailable',
        'gemini_cache_size': len(gemini_cache)
    })


@app.route('/test-gemini')
def test_gemini():
    """Test Gemini API directly"""
    test_message = "Hello, are you working?"

    try:
        if gemini_assistant.available:
            response = gemini_assistant.generate_response(test_message)
            return jsonify({
                'status': 'success',
                'gemini_available': True,
                'response': response,
                'test_message': test_message
            })
        else:
            return jsonify({
                'status': 'failed',
                'gemini_available': False,
                'error': 'Gemini not available',
                'reason': f'HAS_GENAI: {_HAS_GENAI}, API_KEY_SET: {bool(_GEMINI_API_KEY)}, GEMINI_AVAILABLE: {_GEMINI_AVAILABLE}'
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        })


# --- INITIALIZE DATABASE ---
def init_database():
    with app.app_context():
        try:
            # Create all tables
            db.create_all()
            app.logger.info("Database tables created successfully")

            # Create admin user if missing
            admin_email = 'admin@truthguard.com'
            if not User.query.filter_by(email=admin_email).first():
                admin = User(
                    email=admin_email,
                    name='System Administrator',
                    password=generate_password_hash('Admin123!', method='pbkdf2:sha256'),
                    role='admin',
                    is_active=True
                )
                db.session.add(admin)
                db.session.commit()
                app.logger.info('Admin user created')

        except Exception as e:
            app.logger.error(f"Error creating database tables: {e}")
            try:
                # Try to create tables individually
                for table in ['user', 'analyses', 'chat_history', 'gemini_chat']:
                    try:
                        db.session.execute(text(f"""
                            CREATE TABLE IF NOT EXISTS {table} (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                            )
                        """))
                    except:
                        pass
                db.session.commit()
                app.logger.info("Tables created individually")
            except Exception as e2:
                app.logger.error(f"Individual table creation failed: {e2}")


# --- MAIN ENTRYPOINT ---
if __name__ == '__main__':
    # Initialize services
    fast_detector = FastNewsDetector()

    # Setup logging
    setup_logging()

    # Initialize database
    init_database()

    app.logger.info('Starting TruthGuard with Gemini AI integration...')
    app.logger.info(f'✓ Gemini AI: {"Available" if gemini_assistant.available else "Not available"}')
    app.logger.info(f'✓ Gemini Model: {_GEMINI_MODEL if gemini_assistant.available else "N/A"}')
    app.logger.info('✓ Ultra-fast detector ready')
    app.logger.info('✓ Analysis caching enabled')
    app.logger.info('✓ Background database saving enabled')
    app.logger.info('Server ready for immediate response analysis!')

    app.run(debug=True, host='0.0.0.0', port=5000)