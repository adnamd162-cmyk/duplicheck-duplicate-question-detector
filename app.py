import os
import pandas as pd
import joblib
import streamlit as st

from features import (build_feature_frame, predict_batch,
                      heuristic_probability, common_words)

st.set_page_config(page_title="Duplicate Question Detector", page_icon="🔍", layout="wide")

st.markdown("""
<style>
.big-verdict {font-size: 2rem; font-weight: 700; padding: .6rem 1rem; border-radius: 12px; text-align:center;}
.dup  {background:#e8f7ee; color:#146c43; border:1px solid #146c43;}
.notdup {background:#fdecea; color:#a61b1b; border:1px solid #a61b1b;}
.chip {display:inline-block; background:#eef2ff; color:#3730a3; padding:2px 10px; margin:2px; border-radius:999px; font-size:.85rem;}
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "model.pkl"


@st.cache_resource
def load_artifact():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


artifact = load_artifact()
demo_mode = artifact is None
threshold = 0.5 if demo_mode else artifact["threshold"]


def score_pairs(q1s, q2s):
    if not demo_mode:
        return predict_batch(q1s, q2s, artifact)
    feat_df, _, _ = build_feature_frame(q1s, q2s)
    probs = [heuristic_probability(r) for r in feat_df.to_dict("records")]
    return probs, feat_df


with st.sidebar:
    st.title("🔍 Duplicate Question Detector")
    st.caption("Quora Question Pairs • End-to-end NLP project")
    if demo_mode:
        st.warning("**Demo mode** – no `model.pkl` found, using a rule-based scorer. "
                   "Run `python train.py --data sample_train.csv` for the real model.")
    else:
        st.success("Trained model loaded")
        st.metric("Decision threshold", f"{threshold:.2f}")
    st.markdown("**Pipeline**\n1. Text preprocessing\n2. Basic + advanced features\n"
                "3. Bag of Words (q1, q2)\n4. Random Forest classifier")

tab1, tab2, tab3, tab4 = st.tabs(["✍️ Check a pair", "📄 Batch (CSV)", "📊 Model performance", "ℹ️ About"])

with tab1:
    st.subheader("Type any two questions")
    examples = {
        "— choose an example —": ("", ""),
        "Duplicate": ("How can I learn Python quickly?", "What is the fastest way to learn Python?"),
        "Not duplicate": ("How do I lose weight?", "What is the capital of France?"),
        "Tricky (similar words, different meaning)": ("Which is better, Java or Python for beginners?",
                                                      "Which is better, Python or Java for data science?"),
    }
    choice = st.selectbox("Quick examples", list(examples.keys()))
    ex1, ex2 = examples[choice]

    c1, c2 = st.columns(2)
    q1 = c1.text_area("Question 1", value=ex1, height=110, placeholder="e.g. How do I start investing?")
    q2 = c2.text_area("Question 2", value=ex2, height=110, placeholder="e.g. What is the best way to begin investing?")

    if st.button("Check similarity", type="primary", use_container_width=True):
        if not q1.strip() or not q2.strip():
            st.error("Please enter both questions.")
        else:
            probs, feats = score_pairs([q1], [q2])
            p = float(probs[0])
            is_dup = p >= threshold
            left, right = st.columns([1, 1])
            with left:
                cls = "dup" if is_dup else "notdup"
                txt = "✅ DUPLICATE" if is_dup else "❌ NOT DUPLICATE"
                st.markdown(f'<div class="big-verdict {cls}">{txt}</div>', unsafe_allow_html=True)
                st.write("")
                st.metric("Duplicate probability", f"{p*100:.1f}%")
                st.progress(min(max(p, 0.0), 1.0))
            with right:
                cw = common_words(q1, q2)
                st.markdown("**Shared meaningful words**")
                st.markdown("".join(f'<span class="chip">{w}</span>' for w in cw) or "_none_",
                            unsafe_allow_html=True)
                r = feats.iloc[0]
                a, b, c = st.columns(3)
                a.metric("Word share", f"{r['words_share']:.2f}")
                b.metric("Token-set ratio", f"{r['token_set_ratio']:.0f}")
                c.metric("Length diff", f"{int(r['abs_len_diff'])} words")
            with st.expander("See all engineered features"):
                st.dataframe(feats.T.rename(columns={0: "value"}), use_container_width=True)

with tab2:
    st.subheader("Score a CSV file")
    st.caption("CSV needs two columns: `question1`, `question2`")
    up = st.file_uploader("Upload CSV", type="csv")
    if up is not None:
        df = pd.read_csv(up)
        if not {"question1", "question2"} <= set(df.columns):
            st.error("Columns `question1` and `question2` are required.")
        else:
            df = df.dropna(subset=["question1", "question2"]).head(5000).reset_index(drop=True)
            probs, _ = score_pairs(df.question1.astype(str).tolist(), df.question2.astype(str).tolist())
            df["duplicate_probability"] = [round(float(x), 4) for x in probs]
            df["prediction"] = ["Duplicate" if x >= threshold else "Not duplicate" for x in probs]
            st.dataframe(df, use_container_width=True)
            st.download_button("Download results", df.to_csv(index=False).encode(), "predictions.csv", "text/csv")

with tab3:
    st.subheader("Model performance")
    if demo_mode:
        st.info("Train the model first (`python train.py --data sample_train.csv`) to see metrics here.")
    else:
        m = artifact["metrics"]
        cols = st.columns(5)
        for col, k in zip(cols, ["accuracy", "precision", "recall", "f1", "roc_auc"]):
            col.metric(k.upper().replace("_", " "), f"{m[k]*100:.1f}%")
        l, r = st.columns(2)
        with l:
            st.markdown("**Confusion matrix** (rows = actual)")
            cm = pd.DataFrame(m["confusion_matrix"], index=["Actual: not dup", "Actual: dup"],
                              columns=["Pred: not dup", "Pred: dup"])
            st.dataframe(cm, use_container_width=True)
            st.caption(f"Trained on {m['rows']:,} pairs • duplicate rate {m['duplicate_rate']*100:.1f}% "
                       f"• {m['train_seconds']}s")
        with r:
            st.markdown("**Feature importance**")
            fi = pd.Series(m["feature_importance"]).sort_values()
            st.bar_chart(fi, horizontal=True)

with tab4:
    st.markdown