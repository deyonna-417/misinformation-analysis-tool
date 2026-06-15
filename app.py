import streamlit as st
import joblib

# モデル読み込み
model = joblib.load("classifier.joblib")
vectorizer = joblib.load("vectorizer.joblib")

st.title("誤情報拡散パターン分析ツール")

st.write("ニュース記事やSNS投稿を入力してください")

text = st.text_area(
    "入力欄",
    height=250
)

if st.button("分析開始"):

    if text.strip() == "":
        st.warning("文章を入力してください")
    else:

        x = vectorizer.transform([text])

        prob = model.predict_proba(x)[0]

        st.write("デバッグ用:")
        st.write(prob)

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

        st.subheader(f"{icon} 判定結果: {result}")

        st.write(f"誤情報確率: {fake_prob:.2f}%")
        st.write(f"正情報確率: {true_prob:.2f}%")

        st.progress(fake_prob / 100)
