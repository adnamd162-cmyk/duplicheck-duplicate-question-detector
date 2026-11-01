import re
import difflib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

try:
    from rapidfuzz import fuzz as _fuzz
except ImportError:
    _fuzz = None

SAFE_DIV = 0.0001

STOP_WORDS = set("""
a about above after again against all am an and any are as at be because been before being below
between both but by can did do does doing down during each few for from further had has have having
he her here hers herself him himself his how i if in into is it its itself just me more most my myself
no nor not of off on once only or other our ours ourselves out over own same she should so some such
than that the their theirs them themselves then there these they this those through to too under until
up very was we were what when where which while who whom why will with you your yours yourself
yourselves
""".split())

CONTRACTIONS = {
    "ain't": "am not", "aren't": "are not", "can't": "can not", "can't've": "can not have",
    "could've": "could have", "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
    "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
    "he's": "he is", "how's": "how is", "i'd": "i would", "i'll": "i will", "i'm": "i am",
    "i've": "i have", "isn't": "is not", "it's": "it is", "let's": "let us", "mightn't": "might not",
    "mustn't": "must not", "shan't": "shall not", "she's": "she is", "should've": "should have",
    "shouldn't": "should not", "that's": "that is", "there's": "there is", "they'd": "they would",
    "they'll": "they will", "they're": "they are", "they've": "they have", "wasn't": "was not",
    "we'd": "we would", "we'll": "we will", "we're": "we are", "we've": "we have",
    "weren't": "were not", "what's": "what is", "where's": "where is", "who's": "who is",
    "won't": "will not", "would've": "would have", "wouldn't": "would not", "you'd": "you would",
    "you'll": "you will", "you're": "you are", "you've": "you have",
}


def preprocess(q: str) -> str:
    q = str(q).lower().strip()
    q = q.replace("%", " percent ").replace("$", " dollar ").replace("₹", " rupee ")
    q = q.replace("€", " euro ").replace("@", " at ").replace("&", " and ")
    q = q.replace("[math]", "")
    q = re.sub(r"([0-9]+)000000000", r"\1b", q)
    q = re.sub(r"([0-9]+)000000", r"\1m", q)
    q = re.sub(r"([0-9]+)000", r"\1k", q)
    q = " ".join(CONTRACTIONS.get(w, w) for w in q.split())
    q = re.sub(r"<[^>]+>", " ", q)
    q = re.sub(r"[^a-z0-9\s]", " ", q)
    return re.sub(r"\s+", " ", q).strip()


def _ratio(a, b):
    if _fuzz:
        return _fuzz.ratio(a, b)
    return difflib.SequenceMatcher(None, a, b).ratio() * 100


def _partial_ratio(a, b):
    if _fuzz:
        return _fuzz.partial_ratio(a, b)
    s, l = (a, b) if len(a) <= len(b) else (b, a)
    if not s:
        return 0.0
    best = 0.0
    for i in range(0, max(1, len(l) - len(s) + 1)):
        best = max(best, difflib.SequenceMatcher(None, s, l[i:i + len(s)]).ratio() * 100)
    return best


def _token_sort_ratio(a, b):
    if _fuzz:
        return _fuzz.token_sort_ratio(a, b)
    return _ratio(" ".join(sorted(a.split())), " ".join(sorted(b.split())))


def _token_set_ratio(a, b):
    if _fuzz:
        return _fuzz.token_set_ratio(a, b)
    t1, t2 = set(a.split()), set(b.split())
    inter = " ".join(sorted(t1 & t2))
    s1 = (inter + " " + " ".join(sorted(t1 - t2))).strip()
    s2 = (inter + " " + " ".join(sorted(t2 - t1))).strip()
    if not inter:
        return _ratio(s1, s2)
    return max(_ratio(inter, s1), _ratio(inter, s2), _ratio(s1, s2))


def pair_features(raw1, raw2, clean1, clean2):
    f = {}
    f["q1_len"], f["q2_len"] = len(str(raw1)), len(str(raw2))
    w1 = set(w.lower().strip() for w in str(raw1).split())
    w2 = set(w.lower().strip() for w in str(raw2).split())
    f["q1_num_words"], f["q2_num_words"] = len(str(raw1).split()), len(str(raw2).split())
    f["words_common"] = len(w1 & w2)
    f["words_total"] = len(w1) + len(w2)
    f["words_share"] = round(f["words_common"] / (f["words_total"] + SAFE_DIV), 4)

    t1, t2 = clean1.split(), clean2.split()
    s1w = set(w for w in t1 if w not in STOP_WORDS)
    s2w = set(w for w in t2 if w not in STOP_WORDS)
    s1s = set(w for w in t1 if w in STOP_WORDS)
    s2s = set(w for w in t2 if w in STOP_WORDS)
    cwc = len(s1w & s2w)
    csc = len(s1s & s2s)
    ctc = len(set(t1) & set(t2))
    f["cwc_min"] = cwc / (min(len(s1w), len(s2w)) + SAFE_DIV)
    f["cwc_max"] = cwc / (max(len(s1w), len(s2w)) + SAFE_DIV)
    f["csc_min"] = csc / (min(len(s1s), len(s2s)) + SAFE_DIV)
    f["csc_max"] = csc / (max(len(s1s), len(s2s)) + SAFE_DIV)
    f["ctc_min"] = ctc / (min(len(t1), len(t2)) + SAFE_DIV)
    f["ctc_max"] = ctc / (max(len(t1), len(t2)) + SAFE_DIV)
    f["last_word_eq"] = int(bool(t1 and t2 and t1[-1] == t2[-1]))
    f["first_word_eq"] = int(bool(t1 and t2 and t1[0] == t2[0]))

    f["abs_len_diff"] = abs(len(t1) - len(t2))
    f["mean_len"] = (len(t1) + len(t2)) / 2
    m = difflib.SequenceMatcher(None, clean1, clean2).find_longest_match(0, len(clean1), 0, len(clean2))
    f["longest_substr_ratio"] = m.size / (min(len(clean1), len(clean2)) + 1)

    f["fuzz_ratio"] = _ratio(clean1, clean2)
    f["fuzz_partial_ratio"] = _partial_ratio(clean1, clean2)
    f["token_sort_ratio"] = _token_sort_ratio(clean1, clean2)
    f["token_set_ratio"] = _token_set_ratio(clean1, clean2)
    return f


FEATURE_COLS = list(pair_features("a b", "a c", "a b", "a c").keys()) + ["bow_cosine"]


def build_feature_frame(q1_raw, q2_raw):
    c1 = [preprocess(q) for q in q1_raw]
    c2 = [preprocess(q) for q in q2_raw]
    rows = [pair_features(a, b, x, y) for a, b, x, y in zip(q1_raw, q2_raw, c1, c2)]
    return pd.DataFrame(rows), c1, c2


def _row_cosine(a, b):
    num = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    den = np.sqrt(np.asarray(a.multiply(a).sum(axis=1)).ravel()) * \
          np.sqrt(np.asarray(b.multiply(b).sum(axis=1)).ravel())
    return num / (den + SAFE_DIV)


def build_matrix(feat_df, c1, c2, cv):
    b1, b2 = cv.transform(c1), cv.transform(c2)
    feat_df = feat_df.copy()
    feat_df["bow_cosine"] = _row_cosine(b1, b2)
    feat_df = feat_df[FEATURE_COLS]
    X = hstack([b1, b2, csr_matrix(feat_df.values.astype(float))]).tocsr()
    return X, feat_df


def predict_batch(q1_list, q2_list, artifact):
    feat_df, c1, c2 = build_feature_frame(q1_list, q2_list)
    X, feat_df = build_matrix(feat_df, c1, c2, artifact["vectorizer"])
    probs = artifact["model"].predict_proba(X)[:, 1]
    return probs, feat_df


def heuristic_probability(f):
    score = (0.35 * f["token_set_ratio"] / 100 + 0.25 * f["token_sort_ratio"] / 100 +
             0.20 * f["cwc_min"] + 0.10 * f["longest_substr_ratio"] + 0.10 * f["words_share"])
    score -= 0.05 * min(f["abs_len_diff"], 6)
    return float(min(max(score, 0.0), 1.0))


def common_words(q1, q2):
    a = [w for w in preprocess(q1).split()]
    b = set(preprocess(q2).split())
    return sorted({w for w in a if w in b and w not in STOP_WORDS})