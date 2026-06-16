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
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 誤情報拡散パターン分析ツール")

st.caption(
    "TF-IDF + Logistic Regression を用いて誤情報の可能性を判定します"
)

st.write("ニュース記事やSNS投稿を入力してください")

mode = st.radio(
    "入力方法",
    ["テキスト入力", "URL入力"]
)

text = ""

if mode == "テキスト入力":

    text = st.text_area(
        "入力欄",
        height=250
    )

else:

    url = st.text_input(
        "記事URLを入力"
    )

    if url:

        try:

            response = requests.get(
                url,
                timeout=10
            )

            soup = BeautifulSoup(
                response.text,
                "html.parser"
            )

            text = soup.get_text(
                " ",
                strip=True
            )

            st.success("記事を取得しました")

            st.text_area(
                "取得した本文（先頭1000文字）",
                text[:1000],
                height=200
            )

        except Exception as e:

            st.error(
                f"取得失敗: {e}"
            )

# =========================
# 判定
# =========================

if st.button("分析開始", type="primary"):

    if text.strip() == "":
        st.warning("文章を入力してください")

    else:

        with st.spinner("AIが分析中です..."):

            x = vectorizer.transform([text])

            prob = model.predict_proba(x)[0]

            st.write("予測確率:", prob)
            st.write("特徴量数:", x.nnz)

            fake_prob = prob[0] * 100
            true_prob = prob[1] * 100

        # 判定
        if fake_prob >= 75:
            result = "False"
            icon = "🔴"

        elif fake_prob >= 50:
            result = "Caution"
            icon = "🟡"

        else:
            result = "True"
            icon = "🟢"

        # 結果表示
        st.subheader(f"{icon} 判定結果: {result}")

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

        st.progress(fake_prob / 100)

        # コメント
        if result == "False":
            st.error(
                "誤情報の可能性が高いため、情報源の確認を推奨します。"
            )

        elif result == "Caution":
            st.warning(
                "判断が難しい情報です。複数の情報源で確認してください。"
            )

        else:
            st.success(
                "比較的信頼性が高いと判定されました。"
            )

# =========================
# フッター
# =========================

st.markdown("---")

st.caption(
    "本システムは研究・学習目的で作成されています。判定結果は参考情報です。"
)
