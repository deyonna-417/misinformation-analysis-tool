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
page_title="誤情報拡散パターン分析ツール",
page_icon="🔍",
layout="centered"
)

st.title("🔍 誤情報拡散パターン分析ツール")

st.caption(
"TF-IDF + Logistic Regression を用いて情報拡散パターンを分析します"
)

# =========================

# 入力方法選択

# =========================

mode = st.radio(
"入力方法",
["テキスト入力", "URL入力"]
)

text = ""

if mode == "テキスト入力":

```
text = st.text_area(
    "ニュース記事やSNS投稿を入力してください",
    height=250
)
```

else:

```
url = st.text_input(
    "記事URLを入力してください"
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
```

# =========================

# 分析

# =========================

if st.button("分析開始", type="primary"):

```
if text.strip() == "":
    st.warning("文章またはURLを入力してください")

else:

    with st.spinner("AIが分析中です..."):

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

    # =========================
    # 判定結果
    # =========================

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

    # =========================
    # 分析サマリー
    # =========================

    st.markdown("### 📊 分析サマリー")

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

    # =========================
    # 検出特徴語
    # =========================

    feature_names = vectorizer.get_feature_names_out()

    indices = x.nonzero()[1]

    detected_words = [
        feature_names[i]
        for i in indices
    ]

    st.markdown("### 🔍 検出特徴語")

    if len(detected_words) > 0:

        st.write(
            ", ".join(
                detected_words[:20]
            )
        )

    else:

        st.write(
            "特徴語が検出されませんでした"
        )

    # =========================
    # コメント
    # =========================

    if result == "False":

        st.error(
            "誤情報として扱われた情報に見られる特徴が多く検出されました。"
        )

    elif result == "Caution":

        st.warning(
            "判定が難しい情報です。複数の情報源で確認してください。"
        )

    else:

        st.success(
            "一般ニュース記事に近い特徴が検出されました。"
        )
```

# =========================

# フッター

# =========================

st.markdown("---")

st.caption(
"本システムは研究・学習目的で作成されています。判定結果は参考情報です。"
)
