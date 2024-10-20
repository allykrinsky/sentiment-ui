############ 1. IMPORTING LIBRARIES ############

# Import streamlit, requests for API calls, and pandas and numpy for data manipulation

import streamlit as st
import requests
import pandas as pd
import numpy as np
from streamlit_tags import st_tags  # to add labels on the fly!


from transformers import BertTokenizer
import tensorflow as tf
from transformers import TFBertForSequenceClassification


############ 2. SETTING UP THE PAGE LAYOUT AND TITLE ############

# `st.set_page_config` is used to display the default layout width, the title of the app, and the emoticon in the browser tab.

st.set_page_config(
    layout="centered", page_title="Sentiment Classification", page_icon="❄️"
)

############ CREATE THE LOGO AND HEADING ############

# We create a set of columns to display the logo and the heading next to each other.


c1, c2 = st.columns([0.32, 2])

# The snowflake logo will be displayed in the first column, on the left.

with c1:

    st.image(
        "images/logo.png",
        width=85,
    )


# The heading will be on the right.

with c2:

    st.caption("")
    st.title("Sentiment Classification")


# We need to set up session state via st.session_state so that app interactions don't reset the app.

if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False


st.write("")
st.markdown(
    """

Classify keyphrases on the fly with this mighty app. No training needed!

"""
)

st.write("")

with st.form(key="my_form"):

    MAX_KEY_PHRASES = 50

    new_line = "\n"

    pre_defined_keyphrases = ["This is a really good movie. I loved it and will watch again"]

    # Python list comprehension to create a string from the list of keyphrases.
    keyphrases_string = f"{new_line.join(map(str, pre_defined_keyphrases))}"

    # The block of code below displays a text area
    # So users can paste their phrases to classify

    text = st.text_area(
        # Instructions
        "Enter keyphrases to classify",
        # 'sample' variable that contains our keyphrases.
        keyphrases_string,
        # The height
        height=200,
        # The tooltip displayed when the user hovers over the text area.
        help="At least two keyphrases for the classifier to work, one per line, "
        + str(MAX_KEY_PHRASES)
        + " keyphrases max in 'unlocked mode'. You can tweak 'MAX_KEY_PHRASES' in the code to change this",
        key="1",
    )

    text = text.split("\n")  # Converts the pasted text to a Python list
    linesList = []  # Creates an empty list
    for x in text:
        linesList.append(x)  # Adds each line to the list
    linesList = list(dict.fromkeys(linesList))  # Removes dupes
    linesList = list(filter(None, linesList))  # Removes empty lines

    if len(linesList) > MAX_KEY_PHRASES:
        st.info(
            f"❄️ Note that only the first "
            + str(MAX_KEY_PHRASES)
            + " keyphrases will be reviewed to preserve performance. Fork the repo and tweak 'MAX_KEY_PHRASES' in the code to increase that limit."
        )

        linesList = linesList[:MAX_KEY_PHRASES]

    submit_button = st.form_submit_button(label="Submit")

if not submit_button and not st.session_state.valid_inputs_received:
    st.stop()

elif submit_button or st.session_state.valid_inputs_received:

    if submit_button:

        st.session_state.valid_inputs_received = True


    def load_model():

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

        return tokenizer, model


    def predict(input, tokenizer, model):

        predict_input = tokenizer.encode(
            input,
            truncation=True,
            padding=True,
            return_tensors="tf")

        tf_output = model.predict(predict_input)[0]
        tf_prediction = tf.nn.softmax(tf_output, axis=1)
        labels = ['Negative','Positive'] #(0:negative, 1:positive)
        label = tf.argmax(tf_prediction, axis=1)
        label = label.numpy()
        return labels[label[0]]

    model = None
    if not model:
        tokenizer, model = load_model()


    result = []
    for row in linesList:
        result.append(predict(row, tokenizer, model))

    # then we'll convert the list to a dataframe
    # df = pd.DataFrame.from_dict(result)

    st.success("✅ Done!")

    st.caption("")
    st.markdown("### Check the results!")
    st.caption("")

    # Display the dataframe
    st.write(result)


