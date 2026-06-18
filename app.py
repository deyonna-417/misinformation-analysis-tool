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
st.markdown("""
<style>

/* タイトル */
h1 {
    text-align: center;
    color: #4F8BF9;
    font-weight: 700;
}

/* サブタイトル */
p {
    text-align: center;
}

/* metricカード */
div[data-testid="metric-container"] {
    background-color: #1c2333;
    border: 1px solid #4F8BF9;
    padding: 15px;
    border-radius: 15px;
}

/* ボタン */
.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
}

/* プログレスバー */
.stProgress > div > div > div > div {
    background-color: #4F8BF9;
}

</style>
""", unsafe_allow_html=True)
    
    page_title="Misinformation Analysis System",
    page_icon="🛡️",
    layout="centered"
)

st.markdown(
    "<h1>Misinformation Analysis System</h1>",
    unsafe_allow_html=True
)

st.caption(
    "AIによる誤情報拡散パターン分析システム"
)

# =========================
# 入力方法
# =========================

mode = st.radio(
    "入力方法",
    ["テキスト入力", "URL入力"]
)

text = ""
url = ""

# =========================
# テキスト入力
# =========================

if mode == "テキスト入力":

    text = st.text_area(
        "ニュース記事やSNS投稿を入力してください",
        height=250
    )

# =========================
# URL入力
# =========================

else:

    url = st.text_input(
        "記事URL"
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
                "記事を取得しました"
            )

            with st.expander(
                "取得テキスト確認"
            ):

                st.text_area(
                    "先頭1000文字",
                    text[:1000],
                    height=200
                )

        except Exception as e:

            st.error(
                f"記事取得に失敗しました: {e}"
            )

# =========================
# 分析開始
# =========================

if st.button(
    "分析開始",
    type="primary"
):

    if text.strip() == "":

        st.warning(
            "文章またはURLを入力してください"
        )

    else:

        with st.spinner(
            "AIが分析中です..."
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
        # 判定結果
        # =====================

        st.subheader(
            f"{icon} 判定結果 : {result}"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "誤情報確率",
                f"{fake_prob:.2f}%"
            )

        with col2:

            st.metric(
                "正情報確率",
                f"{true_prob:.2f}%"
            )

        # =====================
        # AI判定メーター
        # =====================

        st.markdown("### 📈 AI判定メーター")

        st.progress(
            true_prob / 100
        )

        if true_prob >= 80:

            st.success(
                f"🟢 AI信頼度 : {true_prob:.2f}%"
            )

        elif true_prob >= 60:

            st.warning(
                f"🟡 AI信頼度 : {true_prob:.2f}%"
            )

        else:

            st.error(
                f"🔴 AI信頼度 : {true_prob:.2f}%"
            )

        # =====================
        # 分析サマリー
        # =====================

        st.markdown(
            "### 📊 分析サマリー"
        )

        col3, col4 = st.columns(2)

        with col3:

            st.metric(
                "文字数",
                len(text)
            )

        with col4:

            st.metric(
                "特徴語数",
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
            "### 🔍 検出特徴語"
        )

        if important_words:

            for word in important_words:

                st.write(
                    f"• {word}"
                )

        else:

            st.write(
                "特徴語が検出されませんでした"
            )

# =========================
# フッター
# =========================

st.markdown("---")

st.caption(
    "本システムは研究・学習目的で開発されています。判定結果は参考情報です。"
)
