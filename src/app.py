import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image

import model
import sidebar

# sidebar
sidebar.sidebar()

# title and description
st.title("ARA Flower Classification🌻")
st.text("Upload a flower Image for ARA to predict its name for you.🌵")

# file uploader
uploaded_file = st.file_uploader("Choose a flower picture ...", type="jpg")

# after uploading picture
if uploaded_file is not None:

    # show the uploaded picture and  align it to center
    upload_image = Image.open(uploaded_file)
    with st.columns(3)[1]:
        st.image(upload_image, caption='Uploaded Image',
                 use_column_width='auto')
    image = Image.open(uploaded_file)

    # model predictions
    top_probs, top_classes = model.predict(image, model.model)

    # horizontal line
    st.markdown('---')

    # prediction title and description
    st.markdown('## 💡Prediction')
    st.markdown(f'⚡ARA predicts that your flower is ***"{top_classes[0]}"*** \
                with the probability ***{top_probs[0]*100:.2f}%***.')

    # make a dataframe for horizontal bar chart
    rows = [i for i in top_classes]
    columns = top_probs
    data = pd.DataFrame({
        'Classes': rows,
        'Probabilities': columns
    })

    st.markdown('### 🔥Top 5 Predicted Flower Names')

    # show the table and hide index column
    data2 = pd.DataFrame({
        'Classes': rows,
        'Probabilities': [f'{i*100:.2f}' for i in top_probs]
    })
    hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """
    st.markdown(hide_table_row_index, unsafe_allow_html=True)
    st.table(data2)

    # break line between table and horizontal bar chart
    st.markdown('</br>', unsafe_allow_html=True)

    # show the horizontal bar chart
    color = ["#e7ba52", "#a7a7a7", "#9467bd", "#aec7e8", "#1f77b4"]

    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X('Probabilities', axis=alt.Axis(format='%',
                                               title='Probability')),
        y=alt.Y('Classes', sort=None, axis=alt.Axis(title='Flower_Names')),
        color=alt.Color('Classes', scale=alt.Scale(range=color), legend=None))

    st.altair_chart(chart, theme="streamlit", use_container_width=True)
