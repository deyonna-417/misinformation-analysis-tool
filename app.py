import streamlit as st
import joblib
import requests
from bs4 import BeautifulSoup

# =========================
# モデル読み込み
# =========================

model = joblib.load("classifier.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# =========================
# 画面設定
# =========================

st.set_page_config(
    page_title="Misinformation Analysis System",
    page_icon="🛡️",
    layout="centered"
)

st.title("Misinformation Analysis System")

st.caption(
    "AI-Based Misinformation Pattern Detection"
)

# =========================
# 入力方法
# =========================

mode = st.radio(
    "Input Type",
    ["Text", "URL"]
)

text = ""
url = ""

# =========================
# テキスト入力
# =========================

if mode == "Text":

    text = st.text_area(
        "Input Text",
        height=250
    )

# =========================
# URL入力
# =========================

else:

    url = st.text_input(
        "Article URL"
    )

    if url:

        try:

            response = requests.get(
                url,
                timeout=10,
                headers={
                    "User-Agent": "Mozilla/5.0"
                }
            )

            soup = BeautifulSoup(
                response.text,
                "html.parser"
            )

            text = soup.get_text(
                " ",
                strip=True
            )

            st.success(
                "Article retrieved successfully"
            )

            with st.expander(
                "Preview Extracted Text"
            ):

                st.text_area(
                    "First 1000 Characters",
                    text[:1000],
                    height=200
                )

        except Exception as e:

            st.error(
                f"Failed to retrieve article: {e}"
            )

# =========================
# 分析開始
# =========================

if st.button(
    "Analyze",
    type="primary"
):

    if text.strip() == "":

        st.warning(
            "Please enter text or URL."
        )

    else:

        with st.spinner(
            "Analyzing..."
        ):

            x = vectorizer.transform(
                [text]
            )

            prob = model.predict_proba(
                x
            )[0]

            fake_prob = prob[0] * 100
            true_prob = prob[1] * 100

        # =====================
        # 判定
        # =====================

        if fake_prob >= 75:

            result = "False"
            icon = "🔴"

        elif fake_prob >= 50:

            result = "Caution"
            icon = "🟡"

        else:

            result = "True"
            icon = "🟢"

        # =====================
        # 結果表示
        # =====================

        st.subheader(
            f"{icon} Analysis Result : {result}"
        )

        st.metric(
            "Risk Score",
            f"{fake_prob:.2f}%"
        )

        st.progress(
            fake_prob / 100
        )

        # =====================
        # 確率表示
        # =====================

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Misinformation Probability",
                f"{fake_prob:.2f}%"
            )

        with col2:

            st.metric(
                "Reliable Information Probability",
                f"{true_prob:.2f}%"
            )

        # =====================
        # 分析サマリー
        # =====================

        st.markdown(
            "### 📊 Analysis Summary"
        )

        col3, col4 = st.columns(2)

        with col3:

            st.metric(
                "Characters",
                len(text)
            )

        with col4:

            st.metric(
                "Detected Features",
                x.nnz
            )

        # =====================
        # 特徴語
        # =====================

        feature_names = (
            vectorizer.get_feature_names_out()
        )

        scores = x.toarray()[0]

        top_indices = (
            scores.argsort()[-20:][::-1]
        )

        important_words = []

        for idx in top_indices:

            if scores[idx] > 0:

                word = feature_names[idx]

                if (
                    len(word) >= 3
                    and not word.isdigit()
                ):
                    important_words.append(
                        word
                    )

        st.markdown(
            "### 🔍 Detected Keywords"
        )

        if important_words:

            for word in important_words:

                st.write(
                    f"• {word}"
                )

        else:

            st.write(
                "No significant keywords detected."
            )

        # =====================
        # コメント
        # =====================

        st.markdown(
            "### 📝 AI Comment"
        )

        if result == "False":

            st.error(
                "Patterns commonly observed in misinformation-related articles were detected."
            )

        elif result == "Caution":

            st.warning(
                "This information is difficult to classify. Cross-checking with multiple sources is recommended."
            )

        else:

            st.success(
                "The article shows characteristics similar to general news content."
            )

# =========================
# フッター
# =========================

st.markdown("---")

st.caption(
    "This system is developed for educational and research purposes. "
    "Analysis results are provided as reference information only."
)
