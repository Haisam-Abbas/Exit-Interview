import streamlit as st
import torch
import pandas as pd
import pickle
from transformers import BartTokenizer, BartForSequenceClassification
import matplotlib.pyplot as plt
from io import BytesIO
from huggingface_hub import hf_hub_download

HF_REPO_ID = "HaisamAbbas1/Exit_Interview"


@st.cache_resource
def load_model():  # 3 main things Model, Model Path and Tokenizer
    model = BartForSequenceClassification.from_pretrained(HF_REPO_ID)
    tokenizer = BartTokenizer.from_pretrained(HF_REPO_ID)
    label_path = hf_hub_download(
        repo_id=HF_REPO_ID, filename='label_encoder.pkl')
    with open(label_path, 'rb') as f:
        le = pickle.load(f)
    return model, tokenizer, le


model, tokenizer, le = load_model()
# While you using the values to call the function you use the function to
# call the values


# need to check f4r device: PyTorch twrking overtime
device = torch.device("cuda" if torch.cuda.is_available() else "CPU")
model.to(device)
model.eval()
st.success(f'Using Device: {device}')


st.title("Exit Interview Analyzer")
st.caption(
    'Utilize Machine Learning to analyze your Exit Interviews into 13 different categories')

with st.expander("📘 What This Model Does"):
    st.markdown("""
    ###  What This Model Does

    This app uses a **fine-tuned BART transformer model** to classify employee exit interview responses into one of 13 predefined categories. It is trained on a data set of 2500 Exit Interviews and its accuracy hovers around 75 - 80 percent.

    Due to a lack of GPU, analysis may take sometime. For bulk processes it is recommended to host the model using a GPU locally (Details on GitHub).
    
    After uploading an Excel file containing employee feedback (in a column titled **"Response"**), the model:

    -  Analyzes each comment using natural language processing (NLP)
    -  Predicts the **most likely reason** for the employee's departure
    -  Provides a confidence score for each prediction
    -  Summarizes the results with interactive visualizations and downloadable reports
    
    ---

    ### 🗂️ Exit Categories the Model Can Detect:

    | Label | Category |
    |:--|:--|
    | `0` |  **Career change** |
    | `1` |  **Entrepreneurship** |
    | `2` |  **Further Education** |
    | `3` |  **Heavy workload / Burnout** |
    | `4` |  **Insufficient training or mentoring** |
    | `5` |  **Lack of career growth** |
    | `6` |  **Lack of recognition** |
    | `7` |  **Limited work-life balance** |
    | `8` |  **Management or leadership issues** |
    | `9` |  **Other / Unclear** |
    | `10` |  **Poor compensation or benefits** |
    | `11` |  **Relocation or personal reasons** |
    | `12` |  **Toxic culture or team dynamics** |""")


uploaded_file = st.file_uploader(
    'Upload an Excel file. It should have a "Response" column:', type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("Preview of uploaded data:", df.head())

    if 'Response' not in df.columns:
        st.error(
            'The uploaded file must contain a column named "Response" holding exit interview comments ')
    else:
        if st.button('Run Analysis'):
            predicted_categories = []
            confidences = []

            progress = st.progress(0)
            total = len(df)

            for idx, text in enumerate(df['Response']):
                inputs = tokenizer(
                    str(text),
                    truncation=True,
                    padding='max_length',
                    max_length=256,
                    return_tensors='pt'
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    # for Pytorch tensors we use .item() not .items()
                    pred_label = torch.argmax(logits, dim=1).item()
                    pred_category = le.inverse_transform([pred_label])[0]
                    probs = torch.softmax(logits, dim=1)[0]
                    confidence = probs[pred_label].item()

                predicted_categories.append(pred_category)
                confidences.append(confidence)

                progress.progress((idx + 1) / total)

            df['Predicted_category'] = predicted_categories
            df['Confidence'] = confidences

            st.write("Classified Data", df.head())

            summary = df['Predicted_category'].value_counts().reset_index()
            summary.columns = ['Predicted_Category', 'Count']

            st.subheader("Category Counts")
            st.dataframe(summary)

            # BARCHJARTS
            figure, axis = plt.subplots()
            axis.barh(summary['Predicted_Category'],
                      summary['Count'], color='Skyblue')
            axis.set_xlabel('Count')
            axis.set_ylabel('Predicted_Category')
            axis.set_title('Exit Interview: Category Distribution')
            st.pyplot(figure)

            # DOWNLOAD DA FILE:
            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Classified Data", index=False)
                summary.to_excel(
                    writer, sheet_name="Category_Summary", index=False)

            output.seek(0)

            st.download_button(
                label="DOWNLOAD RESULTS",
                data=output,
                file_name='Exit_Interviews_Analyzed.xlsx',
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            st.success('Analysis Completed')

st.caption(
    'Made by Haisam Abbas using, Streamlit, Pytorch, Transformers and a bit of ~~~vibes 🏝️🌊🏖️')
st.caption(
    "Reach me on LinkedIn: https://www.linkedin.com/in/m-haisam-abbas/")
st.caption(
    "Visit GitHub for details : https://github.com/Haisam-Abbas/Exit-Interview ")
