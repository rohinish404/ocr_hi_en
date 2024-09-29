# OCR and Document Search Web Application Prototype

[Live Link](https://huggingface.co/spaces/rk404/ocr_hi_en)

## Working and Other Details
Models used - Colpali (a multimodel retriever) and Qwen2-vl (Vision model series).
Image is indexed using Colpali and Qwen2-vl is used in the response generation. A prompt is provided to specify what is the query (in our case as we're performing it for ocr purposes prompt will be what to extract).
<br/>
Prompt - **"Extract all english text and hindi text from the document"**
<br/>
### **Steps** - 

### 1. **File Upload**
   - Upload a document through the Streamlit interface. The document can be a PNG, JPG, or JPEG.
     
### 2. **Text Extraction**
   - The uploaded document is processed using the `Colpali` model.
   - The document image is indexed using the `RAGMultiModalModel` for efficient querying.
   - A query is sent to extract all English and Hindi text from the document. The response generation is done wuth the help of vision  model `Qwen2-vl`.
   - The extracted text is stored in the session state to persist across reruns, ensuring that text is only extracted once for a given document.

### 3. **Displaying Extracted Text**
   - The extracted text is displayed in JSON format, making it easy for users to review and understand.

### 4. **Search Functionality**
   - Users can input a keyword to search for within the extracted text.
   - The application splits the extracted text into individual lines and searches for the keyword in each line.
   - If the keyword is found in a line, the entire line is returned with the keyword highlighted.
   - If no matches are found, a message is displayed indicating that the keyword was not found.

## **Technologies Used**

- **Python**
- **Streamlit**
- **byaldi**
- **transformers**
- **Regular Expressions (re)**

## **Local Setup Instructions**

### **1. Prerequisites**
   - **Python 3.8+**: Make sure you have Python installed on your system.
   - **CUDA** (Optional): For GPU support.
   - **PIP**: For installing python packages. 

### **2. Installation Steps**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/rohinish404/ocr_hi_en.git
   cd ocr_hi_en

2. **Install Dependencies: Create and activate a virtual environment (optional but recommended)**:
   ```bash
    python3 -m venv venv
    source venv/bin/activate   # For Linux/MacOS
    pip install -r requirements.txt
    ```
3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## **Deployment Details**
Tried deploying on **Streamlit Cloud** but due to long response times (GPU and RAM required for both Colpali and qwen2-vl), the app was facing issues while running. So, the final deployment was done on **Hugging Face Spaces** where free version provides access to 2 vCPUs and 16GB of RAM.


## **Notes**
- The current deployment on **Hugging Face Spaces** takes a lot of time (~30 mins to 1 hour) to generate responses because it is running on CPUs. So, if deployed with GPU support (Paid version of Hugging Face spaces), it'll be able to generate responses a lot faster.
