import os
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
from PIL import Image
import numpy as np
from image_processing import process_image_notebook
from werkzeug.utils import secure_filename
import cv2
import base64
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import io

app = Flask(__name__, static_folder='static')

# Initialize Azure Blob service client
connect_str = os.getenv('DefaultEndpointsProtocol=https;AccountName=stdocintellignecedev;AccountKey=VJu3Nzic6CfZpI+OTHDBZ9vxzA8nVIzCrtW72P7mMsPKnAvCOYMIVr9DVBD/LmLPgSWSPEuv9glz+AStkTutTA==;EndpointSuffix=core.windows.net')
blob_service_client = BlobServiceClient.from_connection_string(connect_str)

# Process image function (same as before)
def process_image(image_path):
    original_img, output_img = process_image_notebook(image_path)
    output_image = Image.fromarray(output_img)
    compressed_original_image = compress_image(original_img)
    return compressed_original_image, output_image

def compress_image(image, max_size=1000):
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
    aspect_ratio = min(max_size / width, max_size / height)
    new_width = int(width * aspect_ratio)
    new_height = int(height * aspect_ratio)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    _, encoded_image = cv2.imencode('.jpg', resized_image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return encoded_image

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files.get('image')
        if file and file.filename:
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', filename)
            os.makedirs('uploads', exist_ok=True)
            file.save(file_path)
            compressed_original_image, output_image = process_image(file_path)
            compressed_original_img_base64 = base64.b64encode(compressed_original_image).decode('utf-8')
            compressed_original_img_url = f"data:image/jpeg;base64,{compressed_original_img_base64}"
            original_filename, extension = os.path.splitext(filename)
            modified_filename = f"{original_filename}_modified{extension}"
            output_path = os.path.join('static', modified_filename)
            output_image.save(output_path)
            output_image_url = url_for('static', filename=modified_filename)
            return render_template('index.html', output_image=output_image_url, original_img=compressed_original_img_url)
        else:
            return "No file was uploaded.", 400
    return render_template('index.html')

@app.route('/extract_text', methods=['POST'])
def extract_text():
    data = request.json
    image_url = data.get('output_image_url')
    if not image_url:
        return jsonify({'error': 'No image URL provided'}), 400
    
    image_data = base64.b64decode(image_url.split(',')[1])
    blob_client = blob_service_client.get_blob_client(container='input-file', blob='processed_image.jpg')
    blob_client.upload_blob(io.BytesIO(image_data), overwrite=True)
    return jsonify({'message': 'Text extraction in progress'}), 202

@app.route('/download_csv', methods=['GET'])
def download_csv():
    blob_client = blob_service_client.get_blob_client(container='output-data', blob='extracted_text.csv')
    downloader = blob_client.download_blob()
    csv_data = downloader.readall()
    return send_file(io.BytesIO(csv_data), mimetype='text/csv', as_attachment=True, attachment_filename='extracted_text.csv')

if __name__ == '__main__':
    app.run(debug=True)
