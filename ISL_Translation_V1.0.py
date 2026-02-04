# ============================================================
# Indian Spoken Language Video → ISL Sign Video Translator
# Compact UI version (reduced vertical space, colourful UI)
# + Lexical-first Whisper matching + CISLR fallback
# ============================================================

import os
import re
import tempfile
import sys
import time
import shutil

import base64
from io import BytesIO


import importlib.util

import ffmpeg
import numpy as np
import nltk
import streamlit as st

# --- Streamlit state defaults (avoid NameError on reruns) ---
if "whisper_model_medium" not in st.session_state:
  st.session_state.whisper_model_medium = None
if "whisper_model_large" not in st.session_state:
  st.session_state.whisper_model_large = None

import requests # Hugging Face Inference API
import pandas as pd

try:
  from faster_whisper import WhisperModel
except ImportError:
  st.error("The 'faster-whisper' library is required. Install with: pip install faster-whisper")
  WhisperModel = None

from PIL import Image, ImageDraw
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from sentence_transformers import SentenceTransformer
from moviepy.editor import VideoFileClip, ImageSequenceClip, concatenate_videoclips
from googletrans import Translator

from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from difflib import SequenceMatcher

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ------------------------------------------------------------
# Basic setup
# ------------------------------------------------------------

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

SUPPORTED_LANGUAGES = {
  "en": "English", "hi": "Hindi", "ta": "Tamil", "te": "Telugu",
  "bn": "Bengali", "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi", "gu": "Gujarati",
}
INDIC_CODES = {c for c in SUPPORTED_LANGUAGES.keys() if c != "en"}

# Languages in scope for this Streamlit demo (9-language setup)
IN_SCOPE_LANGS = {"hi","ta","te","kn","ml","bn","mr","gu","pa"} # adjust if needed

def _is_in_scope_lang(lang_code: str) -> bool:
  if not lang_code:
    return False
  lc = str(lang_code).strip().lower()
  # normalize a few common variants
  if lc in {"pn", "pa-in"}:
    lc = "pa"
  return lc in IN_SCOPE_LANGS


SIGN_VIDEOS_PATH = (
  "C:/karthick/Jammu/Sem2/ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/Videos_Sentence_Level"
)
ISOLATED_SIGN_VIDEOS_PATH = (
  "C:/karthick/Jammu/Sem2/ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/Videos_Isolated_Signs"
)

# CISLR corpus paths
CISLR_CSV_PATH = r"C:/karthick/Jammu/Sem2/CISLR/dataset.csv" # adjust if name differs
CISLR_VIDEO_ROOT = (
  r"C:/karthick/Jammu/Sem2/CISLR/CISLR_v1.5-a_videos/"
  r"CISLR_v1.5-a_videos/CISLR_v1.5-a_videos"
)



# ------------------------------------------------------------
# HYBRID retrieval from the thesis core (sentence-level + CISLR)
# ------------------------------------------------------------
# We keep Streamlit UI / translation / gloss generation as-is.
# Only the "find best sign video" logic is delegated to the core's HYBRID retrieval.

def _load_hybrid_core():
  """Load the uploaded core file dynamically (keeps this Streamlit file self-contained)."""
  # Prefer a core placed next to this Streamlit file (common for Laptop run)
  candidates = [
    os.path.join(os.path.dirname(__file__), "isl_core_v10_2_2_NOSEAMLESS_MINILMFAST_TWOPASS_ASR.py"),
    os.path.join(os.getcwd(), "isl_core_v10_2_2_NOSEAMLESS_MINILMFAST_TWOPASS_ASR.py"),
    # Colab / Drive typical location (if user copied files there)
    "/content/drive/MyDrive/ISL_Eval_Colab/isl_core_v10_2_2_NOSEAMLESS_MINILMFAST_TWOPASS_ASR.py",
  ]
  core_file = next((p for p in candidates if p and os.path.exists(p)), None)
  if core_file is None:
    st.warning(
      "Hybrid core file not found. "
      "Place 'isl_core_v10_2_2_NOSEAMLESS_MINILMFAST_TWOPASS_ASR.py' next to this Streamlit app."
    )
    return None

  # Use a unique module name to avoid import cache collisions
  module_name = "isl_core_hybrid_streamlit"
  if module_name in sys.modules:
    sys.modules.pop(module_name)

  spec = importlib.util.spec_from_file_location(module_name, core_file)
  if spec is None or spec.loader is None:
    st.warning("Could not load hybrid core (importlib spec error).")
    return None

  mod = importlib.util.module_from_spec(spec)

  # IMPORTANT: insert into sys.modules BEFORE exec_module.
  # This avoids rare import-time failures with dataclasses/pydantic decorators.
  sys.modules[module_name] = mod

  try:
    spec.loader.exec_module(mod) # type: ignore[attr-defined]
    return mod
  except Exception as e:
    st.warning(f"Failed importing hybrid core: {e}")
    return None

# Set env vars so the core uses the same dataset roots as this Streamlit app.
# (Core reads env vars at import time.)
os.environ.setdefault("ISL_SENTENCE_VIDEO_ROOT", SIGN_VIDEOS_PATH)
os.environ.setdefault("CISLR_CSV_PATH", CISLR_CSV_PATH)
os.environ.setdefault("CISLR_VIDEO_ROOT", CISLR_VIDEO_ROOT)

hybrid_core = _load_hybrid_core()

def _folder_to_gt_gloss(folder: str, fallback: str = "") -> str:
  if folder:
    return folder.replace("_", " ").upper().strip()
  return (fallback or "").upper().strip()

rouge_metric = Rouge()
lemmatizer = WordNetLemmatizer()

# FLORES-200 codes (kept for reference if needed later)
INDIC_FLORES_MAP = {
  "hi": "hin_Deva",
  "gu": "guj_Gujr",
  "bn": "ben_Beng",
  "ta": "tam_Taml",
  "te": "tel_Telu",
  "kn": "kan_Knda",
  "ml": "mal_Mlym",
  "mr": "mar_Deva",
}

# ------------------------------------------------------------
# Hugging Face Inference API config for AI4Bharat IndicTrans2
# ------------------------------------------------------------

HF_AI4B_MODEL = "ai4bharat/indictrans2-indic-en-dist-200M"
HF_CHAT_URL = "https://api-inference.huggingface.co/v1/chat/completions"

HF_API_TOKEN = os.getenv("HF_API_TOKEN", "").strip()

HF_HEADERS = {
  "Authorization": f"Bearer {HF_API_TOKEN}",
  "Content-Type": "application/json",
} if HF_API_TOKEN else None

# ------------------------------------------------------------
# Model loaders
# ------------------------------------------------------------

@st.cache_resource
def load_whisper_model(model_size: str = "medium"):
  device = "cuda" if os.environ.get("USE_GPU") == "1" else "cpu"
  compute_type = "float16" if device == "cuda" else "int8"
  try:
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    return model
  except Exception as e:
    st.error(f"Failed to load Faster-Whisper model: {e}")
    return None




def _get_whisper_model(model_size: str):
  """Return cached Whisper model from Streamlit session_state (medium/large)."""
  key = f"whisper_model_{model_size}"
  # Backward-compatible keys if any older defaults used
  if key not in st.session_state:
    st.session_state[key] = None
  if st.session_state[key] is None:
    st.session_state[key] = load_whisper_model(model_size)
  return st.session_state[key]

@st.cache_resource
def load_t2g_model():
  T2G_MODEL_NAME = "google/flan-t5-base"
  try:
    tok = AutoTokenizer.from_pretrained(T2G_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(T2G_MODEL_NAME)
    return tok, model
  except Exception as e:
    st.warning(f"Could not load T2G model '{T2G_MODEL_NAME}': {e}")
    return None, None


@st.cache_resource
def load_sentence_encoder():
  return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_translator():
  return Translator()


@st.cache_resource
def load_paraphraser():
  try:
    return pipeline("text2text-generation", model="tuner007/pegasus_paraphrase")
  except Exception as e:
    print(f"Could not load paraphraser model: {e}")
    return None


@st.cache_resource
def load_cislr_dataset():
  """
  Load CISLR dataset where the first row is the header:
    uid, gloss, duration, category
  and return only uid + gloss.
  """
  if not os.path.exists(CISLR_CSV_PATH):
    print(f"[CISLR] CSV not found: {CISLR_CSV_PATH}")
    return None

  try:
    # No skiprows – row 1 already has: uid, gloss, duration, category
    df = pd.read_csv(CISLR_CSV_PATH)

    # Normalise column names to be safe
    df.columns = [c.strip().lower() for c in df.columns]

    if "uid" not in df.columns or "gloss" not in df.columns:
      print(f"[CISLR] Missing required columns 'uid' or 'gloss': {df.columns}")
      return None

    df = df[["uid", "gloss"]].dropna()
    print(f"[CISLR] Loaded {len(df)} rows with uid + gloss.")
    return df

  except Exception as e:
    print(f"[CISLR] Failed to load dataset: {e}")
    return None
gloss_tokenizer, t2g_model = load_t2g_model()
embedding_model = load_sentence_encoder()
translator = load_translator()
paraphraser = load_paraphraser()
cislr_df = load_cislr_dataset()

# ------------------------------------------------------------
# Text helpers
# ------------------------------------------------------------

GREETING_WORDS = {"hi", "hai", "hello", "hey"}
MANUAL_CONCEPT_SYNONYMS = {
  "blessing": {"congratulation", "congratulations"},
  "blessings": {"congratulation", "congratulations"},
  "congratulation": {"congratulations"},
  "abhinandan": {"congratulation", "congratulations"},
  "hungry": {"famished", "starving"},
  "happy": {"joyful", "glad", "delighted"},
}

MATCH_STOPWORDS = {
  "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
  "a", "an", "the", "this", "that", "these", "those",
  "is", "am", "are", "was", "were", "be", "been", "being",
  "do", "does", "did",
  "can", "could", "may", "might", "shall", "should", "will", "would",
  "please", "kindly", "your", "my", "our", "their",
  "of", "for", "in", "on", "at", "to", "from", "with",
  "and", "or", "but", "so"
}


def clean_text(text: str) -> str:
  if not isinstance(text, str):
    text = str(text)
  text = text.replace("_", " ")
  text = re.sub(r"[^\w\s]", "", text)
  return text.lower().strip()


def important_tokens(text: str, drop_greetings: bool = True):
  tokens = clean_text(text).split()
  if drop_greetings:
    tokens = [t for t in tokens if t not in GREETING_WORDS]
  return tokens


def content_tokens_for_match(text: str):
  """
  Extract *content* words from a sentence or folder name for strict matching.
  - Drop pronouns / auxiliaries (MATCH_STOPWORDS)
  - Canonicalise morphology: slowly/slower -> slow, talking -> talk, etc.
  """
  raw_tokens = important_tokens(text, drop_greetings=True)
  tokens = [t for t in raw_tokens if t not in MATCH_STOPWORDS]

  canon = set()
  for t in tokens:
    base = t

    # simple adverb/comparative normalisation
    if base.endswith("ly") and len(base) > 3:
      base = base[:-2]    # slowly -> slow, really -> real
    elif base.endswith("er") and len(base) > 3:
      base = base[:-2]    # slower -> slow, bigger -> big

    # lemmatise as verb then noun
    base = lemmatizer.lemmatize(base, pos="v")
    base = lemmatizer.lemmatize(base, pos="n")

    canon.add(base)

  return canon


def looks_reasonable(txt: str) -> bool:
  tokens = re.findall(r"[A-Za-z]+", txt)
  if not tokens:
    return False
  if len(tokens) >= 2:
    return True
  return len(tokens[0]) >= 4


def is_valid_english(c: str) -> bool:
  if not c:
    return False
  text = c.strip()
  if len(text) < 3:
    return False
  words = re.findall(r"[A-Za-z]+", text)
  if not words:
    return False
  if len(words) == 1 and len(words[0]) >= 4:
    return True
  if len(words) >= 2:
    return True
  return False


def looks_like_english(text: str) -> bool:
  if not text:
    return False

  t = text.strip()
  if len(t) < 3:
    return False

  ascii_chars = sum(1 for ch in t if ord(ch) < 128)
  ratio = ascii_chars / max(len(t), 1)
  if ratio < 0.7:
    return False

  words = re.findall(r"[A-Za-z']+", t)
  if not words:
    return False

  tokens = t.split()
  if tokens and len(words) / len(tokens) < 0.5:
    return False

  common_english = {
    "the", "is", "are", "am", "you", "i", "we", "they", "have", "has",
    "this", "that", "what", "how", "who", "why", "where", "when",
    "yes", "no", "do", "does", "did", "can", "will", "shall"
  }
  lower_tokens = set(w.lower() for w in words)
  if len(words) >= 3 and common_english.isdisjoint(lower_tokens):
    return False

  return True


def get_semantic_score(candidate_text: str, target_phrase: str) -> float:
  if not candidate_text or not target_phrase or embedding_model is None:
    return 0.0
  try:
    candidate_emb = embedding_model.encode([candidate_text])[0].reshape(1, -1)
    target_emb = embedding_model.encode([target_phrase])[0].reshape(1, -1)
    score = cosine_similarity(candidate_emb, target_emb)
    return score[0, 0]
  except Exception:
    return 0.0


def expand_with_synonyms(base_tokens):
  syns = set()
  for t in base_tokens:
    for syn in wn.synsets(t):
      for lemma in syn.lemma_names():
        cleaned = important_tokens(lemma, drop_greetings=False)
        syns.update(cleaned)
    if t in MANUAL_CONCEPT_SYNONYMS:
      syns.update(MANUAL_CONCEPT_SYNONYMS[t])
  return syns


def lemmatize_tokens(tokens):
  lemmas = set()
  for t in tokens:
    lemmas.add(lemmatizer.lemmatize(t, pos="n"))
    lemmas.add(lemmatizer.lemmatize(t, pos="v"))
  return lemmas


def generate_gan_frame(word: str):
  img = Image.new("RGB", (128, 128), color="#f0f9ff")
  draw = ImageDraw.Draw(img)
  draw.text((10, 50), word, fill="#003366")
  return img


@st.cache_resource
def load_paraphraser():
  try:
    return pipeline("text2text-generation", model="tuner007/pegasus_paraphrase")
  except Exception as e:
    print(f"Could not load paraphraser model: {e}")
    return None


def get_paraphrases(sentence: str, num_return: int = 5):
  if paraphraser is None or not sentence:
    return []
  try:
    outputs = paraphraser(
      sentence,
      num_beams=num_return,
      num_return_sequences=num_return,
    )
    paras = []
    for o in outputs:
      txt = o.get("generated_text", "").strip()
      if txt and txt not in paras:
        paras.append(txt)
    return paras
  except Exception as e:
    print(f"Paraphrase error: {e}")
    return []


STOPWORDS_FOR_GLOSS = {
  "is", "am", "are", "was", "were", "the", "a", "an", "to", "of",
  "and", "or", "do", "does", "did", "for", "in", "on", "at", "with"
}


def simple_gloss_from_english(text: str) -> str:
  words = re.findall(r"[A-Za-z]+", text)
  content = [w.upper() for w in words if w.lower() not in STOPWORDS_FOR_GLOSS]
  return " ".join(content)


def text_to_gloss(english_text: str) -> str:
  if gloss_tokenizer is None or t2g_model is None:
    return simple_gloss_from_english(english_text)

  text = english_text.strip()
  if not text or not re.search(r"[A-Za-z]{3}", text):
    return ""

  prompt = (
    "Convert this English sentence into Indian Sign Language (ISL) gloss. "
    "Use simple uppercase gloss words and remove small function words.\n"
    f"English: {text}\n"
    "ISL gloss:"
  )

  try:
    inputs = gloss_tokenizer(prompt, return_tensors="pt")
    outputs = t2g_model.generate(
      inputs.input_ids,
      max_length=50,
      num_beams=2,
      early_stopping=True,
      length_penalty=0.8,
    )
    gloss_raw = gloss_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    gloss_raw = gloss_raw.replace("\n", " ").strip()

    alpha_tokens = re.findall(r"[A-Za-z]+", gloss_raw)
    if len(alpha_tokens) == 0:
      gloss_raw = simple_gloss_from_english(text)

    return gloss_raw.upper()
  except Exception as e:
    print(f"[ERROR] T2G Generation failed: {e}")
    return simple_gloss_from_english(text)


def refine_english_with_google(text: str) -> str:
  if not text:
    return ""
  try:
    tr = translator.translate(text, dest="en", src="auto")
    cand = tr.text.strip()
    if cand and cand.lower() != text.lower():
      return cand
    return ""
  except Exception:
    return ""


# ------------------------------------------------------------
# Script-based detection for all 9 languages (for GT src)
# ------------------------------------------------------------

def detect_language_script(text: str) -> str:
  ranges = {
    "gu":  (0x0A80, 0x0AFF),
    "hi_mr":(0x0900, 0x097F),
    "bn":  (0x0980, 0x09FF),
    "ta":  (0x0B80, 0x0BFF),
    "te":  (0x0C00, 0x0C7F),
    "kn":  (0x0C80, 0x0CFF),
    "ml":  (0x0D00, 0x0D7F),
  }

  for ch in text:
    code = ord(ch)
    for lang_key, (start, end) in ranges.items():
      if start <= code <= end:
        return lang_key
  return "unknown"


def script_to_google_lang(script_key: str):
  mapping = {
    "gu": "gu",
    "hi_mr": "hi",
    "bn": "bn",
    "ta": "ta",
    "te": "te",
    "kn": "kn",
    "ml": "ml",
    "unknown": None,
  }
  return mapping.get(script_key, None)

# ------------------------------------------------------------
# AI4Bharat via Hugging Face (OpenAI-style chat API)
# ------------------------------------------------------------

def translate_with_ai4bharat_hf(text: str, lang_code: str) -> str:
  if not text or HF_HEADERS is None:
    return ""

  try:
    src_lang_name = SUPPORTED_LANGUAGES.get(lang_code, "source language")

    prompt = (
      f"Translate the following {src_lang_name} sentence into natural English.\n\n"
      f"Source: {text}\n"
      "English translation:"
    )

    payload = {
      "model": HF_AI4B_MODEL,
      "messages": [
        {"role": "user", "content": prompt}
      ],
      "max_tokens": 128,
      "temperature": 0.0,
    }

    resp = requests.post(HF_CHAT_URL, headers=HF_HEADERS, json=payload, timeout=60)

    if resp.status_code != 200:
      print(f"[AI4Bharat HF] status={resp.status_code}, body={resp.text[:200]}")
      return ""

    data = resp.json()
    choices = data.get("choices")
    if not choices:
      print(f"[AI4Bharat HF] no choices in response: {str(data)[:200]}")
      return ""

    content = choices[0].get("message", {}).get("content", "")
    return content.strip()

  except Exception as e:
    print(f"HF AI4Bharat inference error: {e}")
    return ""


# ------------------------------------------------------------
# Whisper forced translate over multiple candidate langs
# ------------------------------------------------------------

def forced_whisper_translate_multi(audio_path: str, candidate_langs):
  whisper_model = _get_whisper_model("medium")
  if whisper_model is None:
    return None, None

  best_text = None
  best_lang = None
  best_score = -1e9

  for lang in candidate_langs:
    try:
      segments_gen, info = whisper_model.transcribe(
        audio_path,
        task="translate",
        language=lang,
        beam_size=5
      )
    except Exception:
      continue

    full_text = ""
    avg_logprob_sum = 0.0
    segment_count = 0

    segments = list(segments_gen)
    for segment in segments:
      full_text += segment.text
      confidence = getattr(segment, 'avg_log_prob', getattr(segment, 'avg_logprob', -1.0))
      avg_logprob_sum += confidence
      segment_count += 1

    text = full_text.strip()
    if not text:
      continue

    avg_logprob = avg_logprob_sum / segment_count if segment_count > 0 else -5.0
    score = avg_logprob
    if looks_reasonable(text):
      score += 0.5

    if score > best_score:
      best_score = score
      best_text = text
      best_lang = lang

  return best_text, best_lang


# ------------------------------------------------------------
# Translation candidates builder
# ------------------------------------------------------------

def get_translation_candidates(transcribed_text: str, resolved_lang: str, extra_english_candidates=None):
  if extra_english_candidates is None:
    extra_english_candidates = []

  results = {
    "english_google": "",
    "english_google_script": "",
    "english_ai4b": "",
  }

  script_key = detect_language_script(transcribed_text or "")
  src_lang = script_to_google_lang(script_key)

  if transcribed_text and src_lang:
    try:
      results["english_google_script"] = translator.translate(
        transcribed_text, dest="en", src=src_lang
      ).text
    except Exception:
      pass

  try:
    if transcribed_text:
      results["english_google"] = translator.translate(
        transcribed_text, dest="en", src="auto"
      ).text
  except Exception:
    pass

  if transcribed_text and resolved_lang in INDIC_CODES:
    ai4b = translate_with_ai4bharat_hf(transcribed_text, resolved_lang)
    results["english_ai4b"] = ai4b

  base_candidates = []

  for cand in extra_english_candidates:
    if cand:
      base_candidates.append(cand)

  if results["english_ai4b"]:
    base_candidates.append(results["english_ai4b"])

  if results["english_google_script"]:
    base_candidates.append(results["english_google_script"])

  if results["english_google"]:
    base_candidates.append(results["english_google"])

  if transcribed_text:
    base_candidates.append(transcribed_text)

  unique_candidates = []
  seen_lower = set()
  for cand in base_candidates:
    if cand and isinstance(cand, str):
      key = cand.lower().strip()
      if key not in seen_lower:
        unique_candidates.append(cand.strip())
        seen_lower.add(key)

  refined_candidates = [
    r for c in unique_candidates if c and (r := refine_english_with_google(c))
  ]

  all_candidates = refined_candidates + unique_candidates

  chosen_english = next((cand for cand in all_candidates if is_valid_english(cand)), None)
  if chosen_english is None:
    chosen_english = next((cand for cand in all_candidates if cand), "")

  chosen_gloss = text_to_gloss(chosen_english)

  return {
    "transcribed_text": transcribed_text,
    "chosen_english": chosen_english,
    "english_google": results["english_google"],
    "english_google_script": results["english_google_script"],
    "english_ai4b": results["english_ai4b"],
    "chosen_gloss": chosen_gloss,
    "all_candidates": all_candidates,
    "script_key": script_key,
    "src_lang": src_lang,
  }


# ------------------------------------------------------------
# A/B/C + AI4Bharat hierarchy
# ------------------------------------------------------------

def choose_best_english(
  forced_whisper_english,
  english_whisper,
  google_english_script,
  google_english,
  ai4bharat_english,
  old_chosen_english,
):
  ordered_candidates = [
    ("forced_whisper_english", forced_whisper_english),
    ("english_whisper", english_whisper),
    ("google_english_script", google_english_script),
    ("google_english", google_english),
  ]

  for src_name, cand in ordered_candidates:
    if cand and looks_like_english(cand):
      return cand, src_name

  if ai4bharat_english and looks_like_english(ai4bharat_english):
    return ai4bharat_english, "ai4bharat_english"

  if old_chosen_english:
    return old_chosen_english, "old_chosen_english"

  for src_name, cand in ordered_candidates:
    if cand:
      return cand, src_name

  if ai4bharat_english:
    return ai4bharat_english, "ai4bharat_english"

  return "", "none"


# ------------------------------------------------------------
# Video synthesis & metrics
# ------------------------------------------------------------

def clean_folder_to_gloss(folder_name: str) -> str:
  return folder_name.replace("_", " ").upper().strip()


def get_concatenative_video(gloss: str, output_path: str):
  if not os.path.isdir(ISOLATED_SIGN_VIDEOS_PATH):
    return None, None, None

  gloss_tokens = [t.strip() for t in gloss.split() if t.strip()]
  video_clips = []
  missing_signs = []

  for token in gloss_tokens:
    token = token.upper()
    sign_folder_path = os.path.join(ISOLATED_SIGN_VIDEOS_PATH, token)
    video_file = None

    if os.path.isdir(sign_folder_path):
      candidates = [f for f in os.listdir(sign_folder_path) if f.endswith((".mp4", ".avi"))]
      if candidates:
        video_file = os.path.join(sign_folder_path, candidates[0])

    if video_file and os.path.exists(video_file):
      try:
        clip = VideoFileClip(video_file)
        video_clips.append(clip)
      except Exception as e:
        print(f"[WARN] Error loading clip {video_file}: {e}")
        missing_signs.append(token)
    else:
      missing_signs.append(token)

  if not video_clips:
    return None, None, None

  st.caption(
    f"Concatenative synthesis: {len(video_clips)}/{len(gloss_tokens)} isolated clips found."
  )

  try:
    final_clip = concatenate_videoclips(video_clips, method="compose")
    final_clip.write_videofile(
      output_path,
      audio=False,
      codec="libx264",
      preset="ultrafast",
      fps=video_clips[0].fps if video_clips else 30,
    )
    for clip in video_clips:
      clip.close()

    return output_path, f"Concatenative (missing {len(missing_signs)}/{len(gloss_tokens)} signs)", gloss.upper()
  except Exception as e:
    st.error(f"Video Concatenation Failed: {e}")
    return None, None, None


def get_sign_video(gloss: str, english_text: str, extra_english=None):
  """
  HYBRID sentence-video retrieval.

  This function intentionally *replaces* the original Streamlit matcher so that
  the same HYBRID logic used in batch/core is applied here too.

  Returns: (video_path, match_msg, gt_gloss)
  """
  if hybrid_core is None:
    return None, "Hybrid core not loaded (cannot retrieve sign video).", None

  if extra_english is None:
    extra_english = []

  # Core expects English text; we pass the best English we have + gloss (for CISLR)
  try:
    rr = hybrid_core.retrieve_sign_video(
      english_text=english_text or "",
      gloss_text=gloss or "",
      mode="hybrid",
      top_k=5,
    )
  except Exception as e:
    return None, f"Hybrid retrieval error: {e}", None

  video_path = getattr(rr, "video_path", None)
  video_folder = getattr(rr, "video_folder", "") or ""
  msg = getattr(rr, "match_msg", "") or getattr(rr, "message", "") or getattr(rr, "msg", "") or "Hybrid retrieval"
  gt_gloss = _folder_to_gt_gloss(video_folder, fallback=gloss)

  if video_path and os.path.exists(video_path):
    return video_path, msg, gt_gloss

  return None, msg, gt_gloss


def get_sign_video_semantic(english_text: str, threshold=0.80):
  """
  Kept for API-compatibility with the existing Streamlit flow.

  In the HYBRID pipeline, semantic matching is already part of core retrieval,
  so we delegate to core HYBRID again.
  """
  v, msg, gt = get_sign_video(gloss="", english_text=english_text, extra_english=None)
  return v, msg, gt


def find_cislr_video(gloss: str, english_text: str, all_candidates=None):
  """
  CISLR fallback is also handled by the core HYBRID retrieval.

  We keep this wrapper so the rest of the Streamlit code remains unchanged.
  """
  if hybrid_core is None:
    return None, "Hybrid core not loaded (cannot run CISLR fallback)."

  try:
    rr = hybrid_core.retrieve_sign_video(
      english_text=english_text or "",
      gloss_text=gloss or "",
      mode="hybrid",
      top_k=5,
    )
  except Exception as e:
    return None, f"CISLR/hybrid retrieval error: {e}"

  video_path = getattr(rr, "video_path", None)
  src = getattr(rr, "synthesis_method", "") or getattr(rr, "method", "")
  msg = getattr(rr, "match_msg", "") or getattr(rr, "message", "") or "Hybrid retrieval"

  # Return only if this looks like a CISLR-derived result (best-effort)
  if video_path and os.path.exists(video_path) and ("cislr" in str(src).lower() or "CISLR" in msg):
    return video_path, msg

  return None, msg


def show_metrics_and_build_video(
  gloss_sequence: str, english_text: str, output_path: str, video_path: str, match_msg: str, gt_gloss: str
):
  if not gloss_sequence.strip():
    st.error("Gloss generation failed. Cannot compute metrics or generate video.")
    return None

  reference_lexical_tokens = gt_gloss.lower().split()
  hypothesis_tokens = gloss_sequence.lower().split()
  reference_lexical = [reference_lexical_tokens]

  try:
    bleu = sentence_bleu(reference_lexical, hypothesis_tokens, smoothing_function=SmoothingFunction().method1)
  except Exception:
    bleu = 0.0
  try:
    rouge_scores = rouge_metric.get_scores(gloss_sequence, gt_gloss)[0]
    rouge_l_f = rouge_scores["rouge-l"]["f"]
  except Exception:
    rouge_l_f = 0.0
  try:
    meteor = meteor_score([reference_lexical_tokens], hypothesis_tokens)
  except Exception:
    meteor = 0.0

  sts_score = get_semantic_score(english_text, gloss_sequence)

  with st.expander("Show gloss quality metrics"):
    st.markdown(
      f"**BLEU:** {round(bleu, 4)} | "
      f"**ROUGE-L F1:** {rouge_l_f:.4f} | "
      f"**METEOR:** {round(meteor, 4)} | "
      f"**STS:** {sts_score:.4f}"
    )

  if video_path:
    clip = VideoFileClip(video_path)
    clip.write_videofile(output_path, audio=False, codec="libx264", preset="ultrafast")
    clip.close()
    return output_path

  st.caption("No matching sentence video. Using GAN-style frame sequence.")
  words = gloss_sequence.split()
  if not words:
    return None

  frames = [np.array(generate_gan_frame(word)) for word in words]
  repeated_frames = []
  for frame in frames:
    repeated_frames.extend([frame] * 30)

  clip = ImageSequenceClip(repeated_frames, fps=30)
  clip.write_videofile(output_path, audio=False, codec="libx264", preset="ultrafast")
  clip.close()
  return output_path


# ------------------------------------------------------------
# Streamlit UI (compact + colourful)
# ------------------------------------------------------------


# --- ISL logo (embedded) ---
_ISL_LOGO_B64 = (
  "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"
)

st.set_page_config(page_title="ISL Sign Translator", layout="centered")

st.markdown(
  """
  <style>
    [data-testid="stAppViewContainer"] {
      background: linear-gradient(135deg, #f0f9ff 0%, #e0f7fa 35%, #fdfbfb 100%);
    }
    .block-container {
      padding-top: 0.5rem;
      padding-bottom: 0.5rem;
      max-width: 850px;
    }
    [data-testid="stHeader"] {
      background-color: rgba(0,0,0,0);
    }

    .video-container {
       display: flex;
       justify-content: center;
       align-items: center;
       margin: 6px 0;
    }
    .video-container video {
       width: 240px !important;
       height: 180px !important;
       border: 2px solid #007acc;
       border-radius: 8px;
    }

    .stButton button, .stDownloadButton button {
       background: linear-gradient(135deg, #0284c7, #22c55e);
       color: white;
       border-radius: 999px;
       padding: 0.4rem 1.2rem;
       border: none;
       font-weight: 600;
       cursor: pointer;
    }
    .stButton button:hover, .stDownloadButton button:hover {
       transform: scale(1.03);
       box-shadow: 0 0 8px rgba(0,0,0,0.18);
    }

    [data-testid="stFileUploader"] > div:first-child {
       padding-top: 0.05rem;
       padding-bottom: 0.05rem;
    }

    div.streamlit-expanderHeader {
       background-color: #e0f2fe;
       color: #075985;
       font-weight: 600;
       border-radius: 6px;
    }

    .small-caption {
       font-size: 0.85rem;
       color: #334155;
    }

    .result-box {
       background: #e0f7ff;
       border-radius: 12px;
       border: 1px solid #bae6fd;
       padding: 0.6rem 0.9rem;
       margin-top: 0.4rem;
       margin-bottom: 0.4rem;
    }
  </style>
""",
  unsafe_allow_html=True,
)

# Header (logo + title side-by-side)
_LOGO_INLINE = (_ISL_LOGO_B64 or "").replace("\n", "")
st.markdown(
  f"""<div style='display:flex; align-items:center; justify-content:center; gap:14px; margin-top:0.2rem; margin-bottom:0.3rem;'>
    <img src="data:image/jpeg;base64,{_LOGO_INLINE}" style="height:60px;" />
    <h2 style='color:#004c6d; margin:0;'>🎥 ISL Sign Translator</h2>
  </div>""",
  unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("Upload spoken sentence video", type=["mp4"])
st.caption(
  "Supported languages: **Hindi / Tamil / Telugu / Bengali / Kannada / Malayalam / Marathi / Gujarati / English**."
)

if uploaded_file:
  temp_dir = tempfile.gettempdir()
  unique_suffix = f"_{time.time()}_{os.getpid()}"
  tmp_video_path = os.path.join(temp_dir, f"input_video{unique_suffix}.mp4")
  tmp_audio_path = os.path.join(temp_dir, f"input_audio{unique_suffix}.wav")
  output_path = os.path.join(temp_dir, f"output_sign{unique_suffix}.mp4")

  with open(tmp_video_path, "wb") as tmp_video:
    tmp_video.write(uploaded_file.read())

  try:
    (
      ffmpeg.input(tmp_video_path)
      .output(tmp_audio_path, ac=1, ar="16k")
      .overwrite_output()
      .run(quiet=True)
    )
  except ffmpeg.Error as e:
    st.error(f"Error during audio extraction: {e.stderr.decode('utf8')}")
    if os.path.exists(tmp_video_path):
      os.remove(tmp_video_path)
    st.stop()

  whisper_model = _get_whisper_model("medium")
  if whisper_model is None:
    st.error("Whisper model could not be loaded.")
    st.stop()

  # ---------- Transcription & language guess ----------
  segments_gen, info = whisper_model.transcribe(tmp_audio_path, task="transcribe")
  orig_text = "".join([s.text for s in segments_gen]).strip()
  whisper_lang = info.language


  # If Whisper-medium guesses a language outside our 9-language scope, re-run Whisper-large:
  # 1) try a better language guess with task="transcribe"
  # 2) if still out-of-scope, force English with task="translate" (best for hybrid retrieval)
  _medium_lang = (whisper_lang or "").strip().lower()
  if not _is_in_scope_lang(_medium_lang):
    # st.warning(
    #   f"Medium guessed out-of-scope language='{_medium_lang}'. "
    #   "Re-running Whisper LARGE to re-check / translate."
    # )
    whisper_large = _get_whisper_model("large")

    # First: re-check language + text with large transcribe
    _seg_l, _info_l = whisper_large.transcribe(tmp_audio_path, task="transcribe")
    _large_text = "".join([s.text for s in _seg_l]).strip()
    _large_lang = (getattr(_info_l, "language", "") or "").strip().lower()

    if _is_in_scope_lang(_large_lang):
      orig_text = _large_text
      whisper_lang = _large_lang
    else:
      # Still out-of-scope -> force English for robust hybrid retrieval
      _seg_t, _info_t = whisper_large.transcribe(tmp_audio_path, task="translate")
      _large_en = "".join([s.text for s in _seg_t]).strip()
      if _large_en:
        orig_text = _large_en
        whisper_lang = "en"
        st.session_state["_forced_whisper_english_from_outscope"] = _large_en
    segments, info2 = whisper_model.transcribe(
    tmp_audio_path, task="transcribe", initial_prompt=orig_text
  )
    # --- FIX: 'segments' was undefined in some versions; use available Whisper segments safely.
  if 'segments' in locals() and segments is not None:
    segments_list = list(segments)
  elif 'segments' in locals() and segments is not None:
    segments_list = list(segments)
  elif 'segments_gen' in locals() and segments_gen is not None:
    segments_list = list(segments_gen)
  else:
    segments_list = []
  segment_count = len(segments_list)

  avg_logprob_sum = 0.0
  for s in segments_list:
    confidence = getattr(s, "avg_log_prob", getattr(s, "avg_logprob", -1.0))
    avg_logprob_sum += confidence
  avg_logprob = avg_logprob_sum / segment_count if segment_count > 0 else -5.0

  col1, col2 = st.columns([1, 2])
  with col1:
    st.audio(tmp_audio_path)
  with col2:
    st.markdown(
      f"<p class='small-caption'><strong>Transcription:</strong> {orig_text}"
      f"&nbsp;&nbsp;|&nbsp;&nbsp;"
      f"<strong>Language guess:</strong> {whisper_lang} "
      f"({SUPPORTED_LANGUAGES.get(whisper_lang, 'Unknown')})</p>",
      unsafe_allow_html=True,
    )

  resolved_lang = whisper_lang
  english_whisper = ""
  forced_whisper_english = ""


  # If we forced English due to out-of-scope LID, surface it as an extra English candidate
  _tmp_forced = st.session_state.pop("_forced_whisper_english_from_outscope", "")
  if _tmp_forced:
    english_whisper = _tmp_forced
    forced_whisper_english = _tmp_forced
  # ---------- A: Forced multi-language translate ----------
  fw_text, fw_lang = forced_whisper_translate_multi(tmp_audio_path, INDIC_CODES)
  if fw_text:
    forced_whisper_english = fw_text
    forced_whisper_lang = fw_lang
    if fw_lang in SUPPORTED_LANGUAGES:
      resolved_lang = fw_lang

    # Ensure Whisper MEDIUM model available for translate step
  if "whisper_model_medium" in st.session_state:
    whisper_model = st.session_state.whisper_model_medium
  else:
    whisper_model = _get_whisper_model("medium")

# ---------- A: Direct Whisper translate using resolved_lang ----------
  if resolved_lang in INDIC_CODES:
    try:
      trans_kwargs = {"task": "translate", "language": resolved_lang}
      segments_trans, info_trans = whisper_model.transcribe(tmp_audio_path, **trans_kwargs)
      english_whisper = "".join([s.text for s in segments_trans]).strip()
    except Exception:
      pass

  if whisper_lang not in SUPPORTED_LANGUAGES:
    orig_text_for_google = ""
  else:
    orig_text_for_google = orig_text

  normal_results = get_translation_candidates(
    transcribed_text=orig_text_for_google,
    resolved_lang=resolved_lang,
    extra_english_candidates=[english_whisper, forced_whisper_english],
  )

  google_english = normal_results["english_google"]
  google_english_script = normal_results.get("english_google_script", "")
  english_ai4b = normal_results.get("english_ai4b", "")
  script_key = normal_results.get("script_key", "unknown")
  old_chosen_english = normal_results.get("chosen_english", "")

  chosen_english, chosen_source = choose_best_english(
    forced_whisper_english=forced_whisper_english or "",
    english_whisper=english_whisper or "",
    google_english_script=google_english_script or "",
    google_english=google_english or "",
    ai4bharat_english=english_ai4b or "",
    old_chosen_english=old_chosen_english or "",
  )

  normal_results["chosen_english"] = chosen_english
  normal_results["chosen_source"] = chosen_source

  chosen_gloss = text_to_gloss(chosen_english)

  # =================== RESULT BOX SECTION ======================
  st.markdown("<div class='result-box'>", unsafe_allow_html=True)

  st.markdown(
    f"<p class='small-caption'><strong>Final English (after hierarchy):</strong> "
    f"<span style='background:#ecfdf5; padding:0 4px; border-radius:4px;'>{chosen_english}</span>"
    f"&nbsp;&nbsp;|&nbsp;&nbsp;"
    f"<strong>Gloss:</strong> "
    f"<span style='background:#eef2ff; padding:0 4px; border-radius:4px;'>{chosen_gloss}</span></p>",
    unsafe_allow_html=True,
  )

  # ----------------- Video synthesis -----------------
  final_video_path, match_msg, gt_gloss = None, None, None
  video_path_holistic, match_msg_holistic, gt_gloss_holistic = None, None, None

  if chosen_gloss:
    final_video_path, match_msg, gt_gloss = get_concatenative_video(chosen_gloss, output_path)

    if final_video_path is None:
      video_path_holistic, match_msg_holistic, gt_gloss_holistic = get_sign_video(
        chosen_gloss, chosen_english, extra_english=normal_results["all_candidates"]
      )

    if video_path_holistic is None and chosen_english:
      v_sem, msg_sem, gt_gloss_sem = get_sign_video_semantic(chosen_english)
      if v_sem:
        video_path_holistic, match_msg_holistic, gt_gloss_holistic = v_sem, msg_sem, gt_gloss_sem

    if video_path_holistic:
      match_msg = match_msg_holistic
      gt_gloss = gt_gloss_holistic
      final_video_path = show_metrics_and_build_video(
        chosen_gloss, chosen_english, output_path, video_path_holistic, match_msg, gt_gloss
      )

    if final_video_path is None and chosen_source in ("forced_whisper_english", "english_whisper"):
      alt_sources = [
        ("google_english_script", google_english_script),
        ("google_english", google_english),
        ("english_ai4b", english_ai4b),
      ]
      for label, alt_eng in alt_sources:
        if not alt_eng or not is_valid_english(alt_eng):
          continue
        if alt_eng.strip().lower() == (chosen_english or "").strip().lower():
          continue

        alt_gloss = text_to_gloss(alt_eng)

        v_alt, msg_alt, gt_gloss_alt = get_sign_video(
          alt_gloss,
          alt_eng,
          extra_english=normal_results["all_candidates"],
        )

        if v_alt is None:
          v_alt_sem, msg_alt_sem, gt_gloss_alt_sem = get_sign_video_semantic(alt_eng)
          if v_alt_sem:
            v_alt, msg_alt, gt_gloss_alt = v_alt_sem, msg_alt_sem, gt_gloss_alt_sem

        if v_alt:
          chosen_english = alt_eng
          chosen_gloss = alt_gloss
          match_msg = f"{msg_alt} (via alternate {label})"
          gt_gloss = gt_gloss_alt if gt_gloss_alt else gt_gloss or chosen_gloss
          final_video_path = show_metrics_and_build_video(
            chosen_gloss, chosen_english, output_path, v_alt, match_msg, gt_gloss
          )
          break

  current_chosen_english = chosen_english
  current_chosen_gloss = chosen_gloss
  current_all_candidates = normal_results["all_candidates"]
  current_gt_gloss = gt_gloss

  if final_video_path is None and embedding_model is not None:
    valid_candidates = [cand for cand in current_all_candidates if is_valid_english(cand)]

    if valid_candidates:
      best_avg_similarity = -1.0
      semantic_anchor = valid_candidates[0]

      for i in range(len(valid_candidates)):
        current_cand = valid_candidates[i]
        total_similarity = 0.0
        for j in range(len(valid_candidates)):
          if i != j:
            total_similarity += get_semantic_score(current_cand, valid_candidates[j])
        avg_similarity = total_similarity / max(1, len(valid_candidates) - 1)
        if avg_similarity > best_avg_similarity:
          best_avg_similarity = avg_similarity
          semantic_anchor = current_cand

      best_consensus_score = -1.0
      best_consensus_english = semantic_anchor

      for cand in valid_candidates:
        model_priority_boost = 0.0
        if cand in [normal_results.get("english_google")]:
          model_priority_boost = 0.3
        semantic_similarity = get_semantic_score(cand, semantic_anchor)
        consensus_score = semantic_similarity + model_priority_boost
        if consensus_score > best_consensus_score:
          best_consensus_score = consensus_score
          best_consensus_english = cand

      CONSENSUS_THRESHOLD = 0.90
      MIN_AVG_SIMILARITY = 0.40

      if best_consensus_score >= CONSENSUS_THRESHOLD and best_avg_similarity >= MIN_AVG_SIMILARITY:
        chosen_english_consensus = best_consensus_english
        chosen_gloss_consensus = text_to_gloss(chosen_english_consensus)

        video_path_consensus, match_msg_consensus, gt_gloss_consensus = get_sign_video(
          chosen_gloss_consensus, chosen_english_consensus, extra_english=current_all_candidates
        )

        if video_path_consensus:
          match_msg = match_msg_consensus
          current_gt_gloss = gt_gloss_consensus
          final_video_path = show_metrics_and_build_video(
            chosen_gloss_consensus,
            chosen_english_consensus,
            output_path,
            video_path_consensus,
            match_msg,
            current_gt_gloss,
          )

  if final_video_path is None and paraphraser is not None and is_valid_english(current_chosen_english):
    paraphrases = get_paraphrases(current_chosen_english, num_return=5)

    for para in paraphrases:
      if not is_valid_english(para):
        continue

      para_gloss = text_to_gloss(para)

      v_para, msg_para, gt_gloss_para = get_sign_video(
        para_gloss,
        para,
        extra_english=current_all_candidates + [current_chosen_english],
      )

      if v_para is None:
        v_para_sem, msg_para_sem, gt_gloss_para_sem = get_sign_video_semantic(para)
        if v_para_sem:
          v_para, msg_para, gt_gloss_para = v_para_sem, msg_para_sem, gt_gloss_para_sem

      if v_para:
        match_msg = f"{msg_para} (via paraphrased English)"
        current_chosen_english = para
        current_chosen_gloss = para_gloss
        current_gt_gloss = gt_gloss_para if gt_gloss_para else current_gt_gloss
        final_video_path = show_metrics_and_build_video(
          para_gloss,
          para,
          output_path,
          v_para,
          match_msg,
          current_gt_gloss if current_gt_gloss else para_gloss,
        )
        break

  if final_video_path is None:
    v_cislr, msg_cislr = find_cislr_video(
      current_chosen_gloss,
      current_chosen_english,
      current_all_candidates,
    )
    if v_cislr:
      match_msg = msg_cislr
      gt_for_metrics = current_chosen_gloss or simple_gloss_from_english(current_chosen_english)
      final_video_path = show_metrics_and_build_video(
        gt_for_metrics,
        current_chosen_english,
        output_path,
        v_cislr,
        match_msg,
        gt_for_metrics,
      )

  # NOTE: Defer GAN-style fallback until AFTER Whisper LARGE retry.
  if final_video_path is None:
    match_msg = "NO_VIDEO_YET"


  label = "Synthesis method"
  if match_msg and match_msg.lower().startswith("semantic match"):
    label = "Semantic match / Synthesis method"
  if match_msg:
    # Hide synthesis/match details from UI (log only)
    try:
      print(f"[SYNTHESIS] {match_msg}")
    except Exception:
      pass
  st.markdown("</div>", unsafe_allow_html=True)

  if final_video_path and os.path.exists(final_video_path):
    try:
      with open(final_video_path, "rb") as f:
        video_bytes = f.read()

      st.markdown('<div class="video-container">', unsafe_allow_html=True)
      st.video(video_bytes, format="video/mp4", start_time=0)
      st.markdown("</div>", unsafe_allow_html=True)

      st.download_button("Download Sign Video", video_bytes, file_name="sign_output.mp4")
    except Exception as e:
      st.error(f"Failed to display/read final video: {e}")
  # ---------- Retry with Whisper LARGE if no sign video found ----------
  if final_video_path is None:
    print("[INFO] No sign video found using Whisper MEDIUM. Retrying ASR with Whisper LARGE…")
    whisper_model_large = _get_whisper_model("large")
    if whisper_model_large is not None:
      try:
        # Directly translate audio to English with a stronger ASR model
        segments_large, info_large = whisper_model_large.transcribe(tmp_audio_path, task="translate")
        english_large = "".join([s.text for s in segments_large]).strip()
      except Exception as e:
        english_large = ""
        st.error(f"Whisper LARGE failed: {e}")

      if english_large:

        # Build gloss from the new English and try HYBRID retrieval again
        gloss_large = text_to_gloss(english_large)
        v2, msg2, gt2 = get_sign_video(gloss_large, english_large, extra_english=[english_large])
        if v2 is None:
          v2_sem, msg2_sem, gt2_sem = get_sign_video_semantic(english_large)
          if v2_sem:
            v2, msg2, gt2 = v2_sem, msg2_sem, (gt2_sem or gloss_large)

        if v2 is None:
          v2_cislr, msg2_cislr = find_cislr_video(gloss_large, english_large, [english_large])
          if v2_cislr:
            v2, msg2, gt2 = v2_cislr, msg2_cislr, (gloss_large or simple_gloss_from_english(english_large))

        if v2:
          match_msg = msg2
          gt_for_metrics = gt2 or gloss_large or simple_gloss_from_english(english_large)
          final_video_path = show_metrics_and_build_video(
            gt_for_metrics,
            english_large,
            output_path,
            v2,
            match_msg,
            gt_for_metrics,
          )

          # Display the final video (same UI pattern)
          try:
            with open(final_video_path, "rb") as f:
              video_bytes = f.read()

            st.markdown('<div class="video-container">', unsafe_allow_html=True)
            st.video(video_bytes, format="video/mp4", start_time=0)
            st.markdown("</div>", unsafe_allow_html=True)

            st.download_button("Download Sign Video", video_bytes, file_name="sign_output.mp4")
          except Exception as e:
            st.error(f"Failed to display/read final video (LARGE retry): {e}")

  # ---------- fallback (only after LARGE retry also failed) ----------
  if final_video_path is None:
    st.warning("No matching sign video after Whisper LARGE retry. Using GAN-style fallback video.")
    gt_gloss_fallback = current_gt_gloss if current_gt_gloss else current_chosen_gloss
    final_video_path = show_metrics_and_build_video(
      current_chosen_gloss, current_chosen_english, output_path, None, "GAN-Style Fallback", gt_gloss_fallback
    )
    match_msg = "GAN-Style Fallback"
    # Display GAN fallback output if it exists
    if final_video_path and os.path.exists(final_video_path):
      try:
        with open(final_video_path, "rb") as f:
          video_bytes = f.read()
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        st.video(video_bytes, format="video/mp4", start_time=0)
        st.markdown("</div>", unsafe_allow_html=True)
        st.download_button("Download Sign Video", video_bytes, file_name="sign_output.mp4")
      except Exception as e:
        st.error(f"Failed to display/read GAN fallback video: {e}")

    else:
      st.error("Could not load Whisper LARGE model.")


  if os.path.exists(tmp_video_path):
    os.remove(tmp_video_path)
  if os.path.exists(tmp_audio_path):
    os.remove(tmp_audio_path)