import streamlit as st
import pandas as pd
import base64
import io
from mergen import StringAnalyzer
from streamlit_option_menu import option_menu


# Sayfa özelliklerini yapılandırmak için config() fonksiyonunu kullanabilirsiniz
st.set_page_config(page_title="MergenSoftAI", layout="wide", initial_sidebar_state="collapsed", page_icon="./Mergen.png")

with st.sidebar:
    selected = option_menu(
        menu_title="MergenSoftAI",
        options=["Metin Analizi"],
        icons=["chat-square-text", "bar-chart-fill", "bookmark"],
        default_index=0,
        menu_icon="braces"
    )


if selected == 'Metin Analizi':
    st.title('Metin Analizi')
    st.info("Metin analizi için aşağıdaki alana metninizi yazınız.")
    text = st.text_area("", height=100)
    if st.button("Metini Analiz Et"):
        string_analyzer = StringAnalyzer(sentence=text)
        text = string_analyzer.sentence_analyze()
        clean_text = st.text_area("", text, height=100)
        if text is not None:
            st.download_button(
                label="İndir",
                data=clean_text.encode("utf-8"),
                file_name=f'Temiz metin.txt',
                mime='text/csv'
            )




    st.info("Metin analizi işlemleri için dosya seçiniz. Uzantılar  '.csv' '.xlsx' '.xls' '.txt'")
    # Dosya yükleme
    uploaded_file = st.file_uploader(label="Dosya seçin", type=['csv', 'xlsx', 'xls', 'txt'])

    # Dosya tipine göre veri okuma işlemi
    if uploaded_file is not None:
        if uploaded_file.type == 'text/csv':
            data = pd.read_csv(uploaded_file, encoding='utf-8')
            st.write(data)
        elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            df = pd.read_excel(uploaded_file)
            # Sütun isimlerini elde etme
            column_names = list(df.columns)
            st.warning("Metin analizi için doğru sütunu seçtiğinizden emin olun!")
            # Kullanıcıya seçim kutusu gösterme
            selected_column = st.selectbox("Lütfen bir sütun seçin", column_names)
            # Seçilen sütuna göre yeni DataFrame oluşturma
            new_df = df[selected_column]
            st.dataframe(df, width=10000)
            button = st.button("Metin Analizi")
            string_analyzer = StringAnalyzer(data=df, columns=selected_column)

            if button:
                st.success("Metin Analizi Başarıyla Gerçekleştirildi!")
                st.text(f"Temizlenmiş {uploaded_file.name} dosyası")
                # new_df = new_df.applymap(lambda x: x.lower() if type(x) == str else x)
                string_analyzer.dataframe_analyze()
                st.dataframe(df, width=10000)
                if df is not None:
                    st.download_button(
                        label="İndir",
                        data=df.to_csv(index=False).encode("utf-8"),
                        file_name='Temiz Data.csv',
                        mime='text/csv'
                    )


        elif uploaded_file.type == 'text/plain':
            # Dosya içeriğini StringIO nesnesine aktarma
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            # Dosyayı okuma ve işlem yapma
            text = stringio.read()
            st.text_area("", text, height=100)

            string_analyzer = StringAnalyzer(sentence=text)

            if st.button("Metni İşle"):
                st.text(f'Temizlenmiş {uploaded_file.name} dosyası')
                text = string_analyzer.sentence_analyze()
                clean_text = st.text_area("", text, height=100)
                if clean_text is not None:
                    st.download_button(
                        label="İndir",
                        data=clean_text.encode("utf-8"),
                        file_name='Temiz Data.txt',
                        mime='text/csv'
                    )




