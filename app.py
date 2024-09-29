import streamlit as st
from PIL import Image
from pdf2image import convert_from_path
from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import time  # For generating unique index names
import json
import re

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize Qwen2-VL model and processor
@st.cache_resource
def load_models():
    # Load RAG MultiModalModel and Qwen2-VL model
    RAG = RAGMultiModalModel.from_pretrained("vidore/colpali")
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device).eval()

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True)

    return RAG, model, processor

RAG, model, processor = load_models()

# Step 1: Upload the file
st.title("OCR extraction")
uploaded_file = st.file_uploader("Upload a PDF or Image", type=["pdf", "png", "jpg", "jpeg"])

# Initialize a session state to store extracted text so it persists across reruns
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = None

if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1].lower()

    # Step 2: Convert PDF to image (if the input is a PDF)
    if file_type == "pdf":
        st.write("Converting PDF to image...")
        images = convert_from_path(uploaded_file)
        image_to_process = images[0]
    else:
        # For images (png/jpg), just open the image directly
        image_to_process = Image.open(uploaded_file)

    # Step 3: Display the uploaded image or PDF
    st.image(image_to_process, caption="Uploaded document", use_column_width=True)

    # Step 4: Dynamically create a unique index name using timestamp
    unique_index_name = f"image_index_{int(time.time())}"  # Generate unique index name using current timestamp

    # Step 5: Perform text extraction only if it's a new file
    if st.session_state.extracted_text is None:
        st.write(f"Indexing document with RAG (index name: {unique_index_name})...")
        image_path = "uploaded_image.png"  # Temporary save path
        image_to_process.save(image_path)
        
        RAG.index(
            input_path=image_path,
            index_name=unique_index_name,  # Use unique index name
            store_collection_with_index=False,
            overwrite=False
        )

        # Step 6: Perform text extraction
        text_query = "Extract all english text and hindi text from the document"
        st.write("Searching the document using RAG...")
        results = RAG.search(text_query, k=1)

        # Prepare the messages for text and image input
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_to_process},
                    {"type": "text", "text": text_query},
                ],
            }
        ]

        # Prepare and process image and text inputs
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(device)

        # Generate text output from the image using Qwen2-VL
        st.write("Generating text...")
        generated_ids = model.generate(**inputs, max_new_tokens=100)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # Step 7: Store the extracted text in session state
        st.session_state.extracted_text = output_text[0]

    # Step 8: Display the extracted text in JSON format
    extracted_text = st.session_state.extracted_text
    structured_text = {"extracted_text": extracted_text}

    st.subheader("Extracted Text (JSON Format):")
    st.json(structured_text)

# Step 9: Implement a search functionality on already extracted text
if st.session_state.extracted_text:
    with st.form(key='text_search_form'):
        search_input = st.text_input("Enter a keyword to search within the extracted text:")
        search_action = st.form_submit_button("Search")

    if search_action and search_input:
        # Split the extracted text into lines for searching
        full_text = st.session_state.extracted_text
        lines = full_text.split('\n')

        results = []
        # Search for keyword in each line and collect lines that contain the keyword
        for line in lines:
            if re.search(re.escape(search_input), line, re.IGNORECASE):
                # Highlight keyword in the line
                highlighted_line = re.sub(f"({re.escape(search_input)})", r"*\1*", line, flags=re.IGNORECASE)
                results.append(highlighted_line)
        
        st.subheader("Search Results:")
        if results == []:
            st.markdown('Not forund')
        st.markdown(results)