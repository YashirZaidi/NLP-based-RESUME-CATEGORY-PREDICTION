import streamlit as st
import pickle
import docx  # Extract text from Word file
import PyPDF2  # Extract text from PDF
import re
import matplotlib.pyplot as plt
import pandas as pd

# Load pre-trained model and TF-IDF vectorizer (ensure these are saved earlier)
svc_model = pickle.load(open('clf.pkl', 'rb'))  # Example file name, adjust as needed
tfidf = pickle.load(open('tfidf.pkl', 'rb'))  # Example file name, adjust as needed
le = pickle.load(open('encoder.pkl', 'rb'))  # Example file name, adjust as needed


# Function to clean resume text
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText


# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text


# Function to extract text from TXT with explicit encoding handling
def extract_text_from_txt(file):
    # Try using utf-8 encoding for reading the text file
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        # In case utf-8 fails, try 'latin-1' encoding as a fallback
        text = file.read().decode('latin-1')
    return text


# Function to handle file upload and extraction
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
    return text


# Function to predict the category of a resume
def pred(input_resume):
    # Preprocess the input text (e.g., cleaning, etc.)
    cleaned_text = cleanResume(input_resume)

    # Vectorize the cleaned text using the same TF-IDF vectorizer used during training
    vectorized_text = tfidf.transform([cleaned_text])

    # Convert sparse matrix to dense
    vectorized_text = vectorized_text.toarray()

    # Prediction
    predicted_category = svc_model.predict(vectorized_text)

    # get name of predicted category
    predicted_category_name = le.inverse_transform(predicted_category)

    return predicted_category_name[0]  # Return the category name


# Streamlit app layout
def main():
    st.set_page_config(page_title="Resume Category Prediction", page_icon="📄", layout="wide")
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Resume Category Prediction</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #888888;'>
        Upload multiple resumes in PDF, DOCX, or TXT format, and get the predicted job category.
    </div>
    """, unsafe_allow_html=True)

    # File upload section - allowing multiple files to be uploaded
    uploaded_files = st.file_uploader("Upload Resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    if uploaded_files:
        category_counts = {}  # Dictionary to store job category counts
        resume_predictions = []  # Store predictions for each resume
        
        # Process each uploaded file
        with st.spinner("Processing... Please wait."):
            for uploaded_file in uploaded_files:
                try:
                    resume_text = handle_file_upload(uploaded_file)
                    category = pred(resume_text)
                    resume_predictions.append((uploaded_file.name, category))
                    
                    # Count categories for visualization
                    if category in category_counts:
                        category_counts[category] += 1
                    else:
                        category_counts[category] = 1
                    
                    st.success(f"Successfully processed: {uploaded_file.name} | Predicted Category: {category}")

                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")

        # Display the predictions in a table
        st.subheader("Predictions for Uploaded Resumes")
        df = pd.DataFrame(resume_predictions, columns=["Resume", "Predicted Category"])
        st.dataframe(df)

        # Visual Analytics: Pie Chart for job category distribution
        st.subheader("Job Category Distribution")
        if category_counts:
            categories = list(category_counts.keys())
            counts = list(category_counts.values())
            fig, ax = plt.subplots()
            ax.pie(counts, labels=categories, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            st.pyplot(fig)
        else:
            st.warning("No categories to display yet.")

if __name__ == "__main__":
    main()
