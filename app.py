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
# 画面
# =========================

st.set_page_config(
    page_title="誤情報拡散パターン分析ツール",
    page_icon="🔍"
)

st.title("🔍 誤情報拡散パターン分析ツール")

st.write(
    "ニュース記事またはURLを入力してください"
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
        "入力欄",
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
                "記事取得成功"
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
                f"取得失敗: {e}"
            )

# =========================
# 分析
# =========================

if st.button("分析開始"):

    if text.strip() == "":

        st.warning(
            "文章またはURLを入力してください"
        )

    else:

        x = vectorizer.transform([text])

        prob = model.predict_proba(x)[0]

        fake_prob = prob[0] * 100
        true_prob = prob[1] * 100

        if fake_prob >= 75:

            result = "False"
            icon = "🔴"

        elif fake_prob >= 50:

            result = "Caution"
            icon = "🟡"

        else:

            result = "True"
            icon = "🟢"

        st.subheader(
            f"{icon} 判定結果: {result}"
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

        st.progress(
            fake_prob / 100
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

        feature_names = vectorizer.get_feature_names_out()

        scores = x.toarray()[0]

        top_indices = scores.argsort()[-20:][::-1]

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
                st.write(f"• {word}")

        else:

            st.write(
                "特徴語が検出されませんでした"
            )

import pandas as pd

chart_df = pd.DataFrame({
    "項目": ["誤情報", "正情報"],
    "確率": [fake_prob, true_prob]
})

st.bar_chart(
    chart_df.set_index("項目")
)

# =========================
# フッター
# =========================

st.markdown("---")

st.caption(
    "本システムは研究・学習目的で作成されています。"
)
