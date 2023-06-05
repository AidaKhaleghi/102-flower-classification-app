import streamlit as st


def sidebar():
    with st.sidebar:
        st.markdown(
            "# 🤔How to use?\n"
            "1. Upload a picture of your flower🌻\n"
            "2. Our model will predict the name of your flower😍\n"
        )

        st.markdown("---")
        st.markdown("# 🔖About")
        st.markdown("ARA is a flower classification model.✨\n")
        st.markdown("ARA allows you to find out your flower name \
                    if you dont know it.😉\n")
        st.markdown(
            "We use DenseNet161 to train ARA. \
             We fine-tune our model on 102-oxford-flower-dataset.\n"
        )
        st.markdown(
            "ARA has a great accuracy:\n"
            "* acurracy on training set: 90% \n"
            "* acurracy on validation set: 95% \n"
            "* acurracy on test set: 96% \n"
            "* acurracy on test set on top_5 predicted classes: 99% \n"
        )
        st.markdown("You can contribute to the project on \
                    [GitHub](https://www.example.com) with your \
                    feedback and suggestions💡")
