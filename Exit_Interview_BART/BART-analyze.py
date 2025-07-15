import torch
import pandas as pd
import pickle
from transformers import BartTokenizer, BartForSequenceClassification

# Somethings to make sure before you start running this script
# You should have the above modules installed so python can import them.
# Having a Graphics Card (GPU) speeds up things exponentially. The script autochecks for a GPU and uses it.
# If not it uses the CPU which is very time intensive.
# The Excel file name should be Exit_Interviews_Data
# Exit Interview comments should be in a seperate column called:  Response
# Your analyzed Excel file will be called : Exit_Interviews_Analyzed


model_path = "./exit_interview_bart_model"
model = BartForSequenceClassification.from_pretrained(model_path)
tokenizer = BartTokenizer.from_pretrained(model_path)


with open(f"{model_path}/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Checks if your GPU is being loaded or not. Despite having one if it defaults to CPU there may be issues with your installed torch version.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"Using device: {device}")

#  Load your new Excel file
# (Make sure your file has the name Exit_Interviews_Data and is an excel file. If you want another file name you can change it here just before .xlsx)
df = pd.read_excel("Exit_Interviews_Data.xlsx")

#  Function to predict for a single text


def predict_category(text):
    inputs = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=256,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_label = torch.argmax(logits, dim=1).item()
        pred_category = le.inverse_transform([pred_label])[0]
        probs = torch.softmax(logits, dim=1)[0]
        confidence = probs[pred_label].item()
    return pred_category, confidence


# Run predictions
predicted_categories = []
confidences = []

for text in df['Response']:
    category, conf = predict_category(str(text))
    predicted_categories.append(category)
    confidences.append(conf)

df['Predicted_Category'] = predicted_categories
df['Confidence'] = confidences

# Creates a summary table with the number of times each reason has appeared in a seperate workbook.
summary = df['Predicted_Category'].value_counts().reset_index()
summary.columns = ['Predicted_Category', 'Count']

#  Saving predictions + summary counts to output Excel file
output_file = "Exit_Interviews_Analyzed.xlsx"
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Classified_Data", index=False)
    summary.to_excel(writer, sheet_name="Category_Summary", index=False)

print(f" Predictions and summary saved to '{output_file}'")
print(f"Labels used: {list(le.classes_)}")
