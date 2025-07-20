import streamlit as st
import torch
import pandas as pd
import pickle
from transformers import BartTokenizer, BartForSequenceClassification
import matplotlib.pyplot as plt
from io import BytesIO


@st.cache_resource
def load_model():  # 3 main things Model, Model Path and Tokenizer
    model_path = "Exit_Interview_bart_model"
    model = BartForSequenceClassification.from_pretrained(model_path)
    tokenizer = BartTokenizer.from_pretrained(model_path)
    with open(f'{model_path}/label_encoder.pkl', 'rb') as f:
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


uploaded_file = st.file_uploader(
    'Upload an Excel file. It should have a "Response" column:', type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("Preview of uploaded data:", df.head())

    if st.button('Run Analysis'):
        predicted_categories = []
        confidences = []

        progress = st.progress(0)
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
