#!/usr/bin/env python3
# Form Digitization Script for Hindi Application Forms
# Converts form images to CSV data using Google's Gemini API

import os
import io
import base64
import pandas as pd
import csv
import time
import concurrent.futures
from PIL import Image
import json
import re
from datetime import datetime
import glob
import threading
import argparse
import signal
import sys

# Make sure to install required packages:
# pip install google-generativeai pandas pillow

try:
    import google.generativeai as genai
except ImportError:
    print("Please install the Google Generative AI Python SDK:")
    print("pip install google-generativeai")
    exit(1)

# Global lock for thread-safe CSV writing
csv_lock = threading.Lock()

# Flag to indicate graceful shutdown
shutdown_requested = False

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    global shutdown_requested
    print("\n\nGraceful shutdown requested. Completing current tasks and saving data...")
    shutdown_requested = True

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Set up Gemini API with optimized settings
def setup_gemini_api(api_key):
    """Configure Gemini API with optimized settings"""
    genai.configure(api_key=api_key)
    
    # Performance-optimized settings
    generation_config = {
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 0,
        "max_output_tokens": 8192,
    }
    
    # Use the most capable model available
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-001",
        generation_config=generation_config
    )
    
    return model

# Get the optimized extraction prompt
def get_extraction_prompt():
    """Return optimized prompt for faster Gemini API processing"""
    return """
        Your role is to accurately extract and translate information from Hindi application forms (नागरिकता आवेदन) to English. Follow these specific guidelines:

        1. EXTRACTION RULES:
           - Extract exactly these fields and always translate to English:
             * Citizenship or E-citizenship (check which box is marked)
             * Full Name (पूरा नाम) - use exact spelling as written
             * Nationality (राष्ट्रीयता)
             * Date of Birth (जन्म तिथि) - format as DD/MM/YYYY
             * Place of Birth (जन्म स्थान)
             * Gender (लिंग) - translate as "Male", "Female", or "Other"
             * Phone (फोन) - include ALL digits
             * Email (ईमेल)
             * Address Line 1 (पता)
             * Address Line 2 (पता पंक्ति 2)
             * Educational History (शैक्षिक इतिहास) 
             * Business History (व्यवसाय इतिहास)
             * Participated in Kailas program? (कैलास के कार्यक्रम में भाग लिया?)
             * Spiritual Name (आध्यात्मिक नाम)
             * Parents #1 (माता-पिता #1)
             * Parents #2 (माता-पिता #2)
             * Previous Home Address (पिछले घर का पता)
             * Parent's Phone (फोन)
             * Application Number - look for a circled number on the form (e.g., "107", "118")
             * Signature Date - if present

        2. VALIDATION CHECKS:
           - Phone numbers should only contain digits, typically 10 digits for Indian numbers
           - Email addresses must contain @ symbol if provided
           - Date formats should be consistent (use DD/MM/YYYY)
           - If a field is empty or illegible, mark it ONLY as "Not provided"
           - For checked boxes, report "Citizenship" or "E-citizenship" based on which box is marked
           - If no box is checked, report "Not marked"
           - If content is in English, preserve the exact spelling

        3. SPECIAL ATTENTION:
           - Look for the circled application number on the form (usually appearing as a circled number like "107")
           - Verify if fields are truly empty before marking as "Not provided"
           - Check signature and date at the bottom of the form
           - Be careful with numeric fields (dates, phone numbers) - verify all digits are captured

        4. OUTPUT FORMAT:
           - Return a clean JSON object with these exact field names
           - Do NOT include explanations or notes outside the JSON structure
           - Format should be strictly:
           - Always output in English.
           
        ```json
        {
          "Application Number": "XXX",
          "Citizenship Type": "XXX",
          "Full Name": "XXX",
          "Nationality": "XXX",
          "Date of Birth": "XXX",
          "Place of Birth": "XXX",
          "Gender": "XXX",
          "Phone": "XXX",
          "Email": "XXX",
          "Address Line 1": "XXX",
          "Address Line 2": "XXX",
          "Educational History": "XXX",
          "Business History": "XXX",
          "Participated in Kailas program?": "XXX",
          "Spiritual Name": "XXX",
          "Parents #1": "XXX", 
          "Parents #2": "XXX",
          "Previous Home Address": "XXX",
          "Parent's Phone": "XXX",
          "Signature Date": "XXX"
        }
    """

# Function to optimize image for faster processing
def optimize_image(image_path, max_size=1800):
    """Resize image to optimize for OCR while reducing file size"""
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Calculate new dimensions while maintaining aspect ratio
        width, height = img.size
        if width > max_size or height > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            
            # Resize the image
            img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to RGB if needed (to avoid RGBA issues)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Save to memory buffer
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        buffer.seek(0)
        
        return buffer
    except Exception as e:
        print(f"Error optimizing image {image_path}: {str(e)}")
        # Return original image as fallback
        with open(image_path, 'rb') as f:
            return io.BytesIO(f.read())

# Function to extract application number from filename
def extract_app_number_from_filename(filename):
    """Extract application number from filename"""
    # Try to find "Application no.X" pattern
    match = re.search(r'Application no\.?\s*(\d+)', filename, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Try to find just the number at the end
    match = re.search(r'(\d+)(?:\.\w+)?$', filename)
    if match:
        return match.group(1)
    
    return None

# Process form with retry logic
def extract_form_fields(model, image_path, max_retries=3):
    """Process form image with Gemini and retry logic for failures"""
    filename = os.path.basename(image_path)
    app_number_from_filename = extract_app_number_from_filename(filename)
    
    # Initialize retry parameters
    retry_count = 0
    retry_delay = 5  # Start with 5 seconds delay
    
    while retry_count <= max_retries:
        try:
            # Optimize image for faster processing
            image_buffer = optimize_image(image_path)
            
            # Encode image to base64
            b64_image = base64.b64encode(image_buffer.getvalue()).decode('utf-8')
            
            # Get prompt
            prompt = get_extraction_prompt()
            
            print(f"Processing image: {filename} (try {retry_count+1})")
            
            # Process with Gemini - REMOVED TIMEOUT PARAMETER THAT WAS CAUSING ERRORS
            response = model.generate_content(
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {"text": prompt},
                            {"inline_data": {"mime_type": "image/jpeg", "data": b64_image}}
                        ]
                    }
                ]
            )
            
            # Extract JSON from response
            try:
                json_text = response.text
                
                # Try to find JSON pattern
                json_match = re.search(r'```json\s*(.*?)\s*```', json_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(1)
                else:
                    # Try to find just the JSON object if not in code block
                    json_match = re.search(r'(\{.*\})', json_text, re.DOTALL)
                    if json_match:
                        json_text = json_match.group(1)
                
                # Parse JSON
                extracted_data = json.loads(json_text)
                
                # Post-process and validate
                extracted_data = validate_and_clean_data(extracted_data, filename, app_number_from_filename)
                
                print(f"Successfully extracted data from {filename}")
                return extracted_data
                
            except json.JSONDecodeError as e:
                # If we're on the last retry, use fallback data
                if retry_count == max_retries:
                    print(f"JSON parsing error on final retry: {str(e)}")
                    return create_fallback_data(filename, app_number_from_filename)
                else:
                    print(f"JSON parsing error, retrying: {str(e)}")
                    retry_count += 1
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
            
        except Exception as e:
            # If we're on the last retry, use fallback data
            if retry_count == max_retries:
                print(f"Error on final retry: {str(e)}")
                return create_fallback_data(filename, app_number_from_filename)
            else:
                print(f"Error, retrying: {str(e)}")
                retry_count += 1
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
    
    # If we get here, all retries failed
    return create_fallback_data(filename, app_number_from_filename)

# Function to validate and clean extracted data
def validate_and_clean_data(data, filename, app_number_from_filename):
    """Apply validation and cleaning logic to extracted data"""
    
    # Fix application number if missing or incorrect
    if "Application Number" not in data or not data["Application Number"] or data["Application Number"] == "XXX":
        if app_number_from_filename:
            data["Application Number"] = app_number_from_filename
        else:
            # Try to extract from the filename
            match = re.search(r'(\d+)(?:\.\w+)?$', filename)
            if match:
                data["Application Number"] = match.group(1)
    
    # Clean phone numbers - remove non-digits
    if "Phone" in data and data["Phone"] and data["Phone"] != "Not provided":
        phone = re.sub(r'\D', '', data["Phone"])
        if phone:
            data["Phone"] = phone
            
    # Clean parent's phone number
    if "Parent's Phone" in data and data["Parent's Phone"] and data["Parent's Phone"] != "Not provided":
        phone = re.sub(r'\D', '', data["Parent's Phone"])
        if phone:
            data["Parent's Phone"] = phone
    
    # Validate email format
    if "Email" in data and data["Email"] and data["Email"] != "Not provided":
        if "@" not in data["Email"]:
            data["Email"] = "Not provided"
    
    # Normalize date formats
    for field in ["Date of Birth", "Signature Date"]:
        if field in data and data[field] and data[field] != "Not provided":
            # Extract all numbers from the date
            date_parts = re.findall(r'\d+', data[field])
            if len(date_parts) >= 3:
                day, month, year = date_parts[:3]
                # Ensure 4-digit year
                if len(year) == 2:
                    year = "20" + year if int(year) < 50 else "19" + year
                # Format as DD/MM/YYYY
                data[field] = f"{day.zfill(2)}/{month.zfill(2)}/{year}"
    
    # Ensure all required fields exist
    required_fields = [
        "Application Number", "Citizenship Type", "Full Name", "Nationality", 
        "Date of Birth", "Place of Birth", "Gender", "Phone", "Email",
        "Address Line 1", "Address Line 2", "Educational History",
        "Business History", "Participated in Kailas program?",
        "Spiritual Name", "Parents #1", "Parents #2",
        "Previous Home Address", "Parent's Phone", "Signature Date"
    ]
    
    for field in required_fields:
        if field not in data or not data[field] or data[field] == "XXX":
            data[field] = "Not provided"
    
    return data

# Function to create fallback data if extraction fails
def create_fallback_data(filename, app_number_from_filename):
    """Create fallback data structure with basic information"""
    app_number = app_number_from_filename
    
    if not app_number:
        # Try to extract from the filename
        match = re.search(r'(\d+)(?:\.\w+)?$', filename)
        if match:
            app_number = match.group(1)
        else:
            app_number = "Not provided"
    
    return {
        "Application Number": app_number,
        "Citizenship Type": "Not provided",
        "Full Name": "Not provided",
        "Nationality": "Not provided",
        "Date of Birth": "Not provided",
        "Place of Birth": "Not provided",
        "Gender": "Not provided",
        "Phone": "Not provided",
        "Email": "Not provided",
        "Address Line 1": "Not provided",
        "Address Line 2": "Not provided",
        "Educational History": "Not provided",
        "Business History": "Not provided",
        "Participated in Kailas program?": "Not provided",
        "Spiritual Name": "Not provided",
        "Parents #1": "Not provided",
        "Parents #2": "Not provided",
        "Previous Home Address": "Not provided",
        "Parent's Phone": "Not provided",
        "Signature Date": "Not provided"
    }

# Setup CSV file with headers
def setup_csv_file(csv_path):
    """Create or verify CSV file with appropriate headers"""
    headers = [
        "Application Number", "Citizenship Type", "Full Name", "Nationality", 
        "Date of Birth", "Place of Birth", "Gender", "Phone", "Email",
        "Address Line 1", "Address Line 2", "Educational History",
        "Business History", "Participated in Kailas program?",
        "Spiritual Name", "Parents #1", "Parents #2",
        "Previous Home Address", "Parent's Phone", "Signature Date",
        "Processing Status", "Timestamp", "Filename"
    ]
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    
    # Check if file exists
    file_exists = os.path.isfile(csv_path)
    
    if not file_exists:
        # Create new file with headers
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
        print(f"Created new CSV file: {csv_path}")
    else:
        # Verify headers in existing file
        with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            existing_headers = next(reader, None)
            
            if existing_headers != headers:
                print("Warning: Existing CSV has different headers than expected")
                
    return file_exists

# Get list of already processed forms from CSV
def get_processed_forms(csv_path):
    """Get list of form IDs and filenames that have already been processed"""
    processed_forms = []
    processed_app_numbers = []
    
    if not os.path.exists(csv_path):
        return processed_forms, processed_app_numbers
        
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Get application numbers
        if "Application Number" in df.columns:
            processed_app_numbers = df["Application Number"].astype(str).tolist()
        
        # Get filenames if available
        if "Filename" in df.columns:
            processed_forms = df["Filename"].dropna().tolist()
            
        print(f"Found {len(processed_forms)} processed filenames and {len(processed_app_numbers)} application numbers")
        
    except Exception as e:
        print(f"Error reading processed forms: {str(e)}")
        
    return processed_forms, processed_app_numbers

# Add form data to CSV with thread safety and immediate flush
def append_to_csv(csv_path, data_row):
    """Append a single data row to CSV file with thread safety and immediate flush"""
    with csv_lock:  # Use thread lock to prevent concurrent writes
        with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data_row)
            # Force flush to disk immediately
            csvfile.flush()
            os.fsync(csvfile.fileno())

# Worker function for parallel processing
def process_form_worker(image_path, output_csv, model, processed_count, total_forms, start_time, processing_times):
    """Worker function to process a single form for parallel execution"""
    global shutdown_requested
    
    # Check if shutdown was requested
    if shutdown_requested:
        return False
        
    form_start_time = time.time()
    
    filename = os.path.basename(image_path)
    print(f"\nProcessing form: {filename}")
    
    try:
        # Extract form data
        form_data = extract_form_fields(model, image_path)
        
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create data row
        data_row = [
            form_data.get('Application Number', 'Not provided'),
            form_data.get('Citizenship Type', 'Not provided'),
            form_data.get('Full Name', 'Not provided'),
            form_data.get('Nationality', 'Not provided'),
            form_data.get('Date of Birth', 'Not provided'),
            form_data.get('Place of Birth', 'Not provided'),
            form_data.get('Gender', 'Not provided'),
            form_data.get('Phone', 'Not provided'),
            form_data.get('Email', 'Not provided'),
            form_data.get('Address Line 1', 'Not provided'),
            form_data.get('Address Line 2', 'Not provided'),
            form_data.get('Educational History', 'Not provided'),
            form_data.get('Business History', 'Not provided'),
            form_data.get('Participated in Kailas program?', 'Not provided'),
            form_data.get('Spiritual Name', 'Not provided'),
            form_data.get('Parents #1', 'Not provided'),
            form_data.get('Parents #2', 'Not provided'),
            form_data.get('Previous Home Address', 'Not provided'),
            form_data.get('Parent\'s Phone', 'Not provided'),
            form_data.get('Signature Date', 'Not provided'),
            "Completed",  # Processing status
            timestamp,   # Timestamp
            filename     # Original filename
        ]
        
        # Update CSV file immediately with this row
        append_to_csv(output_csv, data_row)
        print(f"Successfully processed and saved {filename}")
        
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        # Add row with error status
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        app_number = extract_app_number_from_filename(filename) or "Not provided"
        
        error_row = [app_number] + ["Not provided"] * 19 + [f"Error: {str(e)}", timestamp, filename]
        append_to_csv(output_csv, error_row)
    
    # Calculate processing time
    form_processing_time = time.time() - form_start_time
    
    # Update shared progress tracking
    with csv_lock:  # Use lock to prevent race conditions
        processing_times.append(form_processing_time)
        processed_count[0] += 1
        
        # Display progress in terminal
        current = processed_count[0]
        
        # Calculate completion percentage
        percent_complete = (current / total_forms) * 100 if total_forms > 0 else 0
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        
        # Calculate average processing time per form
        avg_time_per_form = sum(processing_times) / len(processing_times) if processing_times else 0
        
        # Estimate remaining time
        remaining_forms = total_forms - current
        est_remaining_time = avg_time_per_form * remaining_forms / 4 if avg_time_per_form > 0 else 0  # Adjusted for parallelism
        remaining_str = time.strftime("%H:%M:%S", time.gmtime(est_remaining_time))
        
        # Calculate processing speed (forms per minute)
        if elapsed_time > 0:
            forms_per_minute = (current / elapsed_time) * 60
        else:
            forms_per_minute = 0
        
        # Create progress bar
        bar_length = 30
        filled_length = int(bar_length * current / total_forms) if total_forms > 0 else 0
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Print progress information
        print(f"\r[{bar}] {percent_complete:.1f}% | {current}/{total_forms} forms | Elapsed: {elapsed_str} | Remaining: {remaining_str} | {forms_per_minute:.1f} forms/min", end='')
    
    return True

# Process forms in parallel
def process_forms_parallel(image_dir, output_csv, gemini_api_key, max_workers=4, batch_size=20):
    """Process forms in parallel for much faster execution"""
    global shutdown_requested
    
    # Setup Gemini API
    model = setup_gemini_api(gemini_api_key)
    
    # Setup CSV file and get processed forms
    csv_existed = setup_csv_file(output_csv)
    processed_forms, processed_app_numbers = get_processed_forms(output_csv) if csv_existed else ([], [])
    
    # Get all image files from directory
    image_patterns = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for pattern in image_patterns:
        image_files.extend(glob.glob(os.path.join(image_dir, pattern)))
    
    print(f"Found {len(image_files)} total image files")
    
    # Filter out already processed forms
    forms_to_process = []
    for image_path in image_files:
        filename = os.path.basename(image_path)
        
        # Check if this filename has been processed
        if filename in processed_forms:
            continue
            
        # Check if the application number in the filename has been processed
        app_number = extract_app_number_from_filename(filename)
        if app_number and app_number in processed_app_numbers:
            continue
            
        forms_to_process.append(image_path)
    
    total_forms = len(forms_to_process)
    print(f"Will process {total_forms} new forms")
    
    if total_forms == 0:
        print("No new forms to process.")
        return None
    
    # Initialize progress tracking variables
    start_time = time.time()
    processing_times = []
    processed_count = [0]  # Use a list to make it mutable across threads
    
    try:
        # Process in batches with parallel execution
        for batch_start in range(0, total_forms, batch_size):
            # Check if shutdown was requested
            if shutdown_requested:
                break
                
            batch_end = min(batch_start + batch_size, total_forms)
            batch = forms_to_process[batch_start:batch_end]
            
            print(f"\nProcessing batch {batch_start//batch_size + 1}/{(total_forms + batch_size - 1)//batch_size}")
            
            # Create a partial function with common arguments
            from functools import partial
            worker_func = partial(
                process_form_worker,
                output_csv=output_csv,
                model=model,
                processed_count=processed_count,
                total_forms=total_forms,
                start_time=start_time,
                processing_times=processing_times
            )
            
            # Use ThreadPoolExecutor for parallel processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                futures = []
                for image_path in batch:
                    # Only submit if not shutting down
                    if not shutdown_requested:
                        futures.append(executor.submit(worker_func, image_path))
                
                # Wait for all to complete
                for future in concurrent.futures.as_completed(futures):
                    # Check for shutdown during processing
                    if shutdown_requested:
                        print("\nShutdown requested - waiting for current tasks to complete...")
                        break
            
            # Display batch completion
            if not shutdown_requested:
                print(f"\nCompleted batch {batch_start//batch_size + 1}/{(total_forms + batch_size - 1)//batch_size}")
                
                # Give the API a brief rest between batches to avoid rate limits
                if batch_end < total_forms:
                    print("Pausing briefly between batches...")
                    time.sleep(2)
        
        if shutdown_requested:
            print(f"\n\nGraceful shutdown complete. Processed {processed_count[0]} of {total_forms} forms.")
        else:
            print(f"\n\nForm processing complete. All data saved to CSV: {output_csv}")
        
        # Final statistics
        total_time = time.time() - start_time
        avg_time_per_form = sum(processing_times) / len(processing_times) if processing_times else 0
        
        print(f"\nProcessing Statistics:")
        print(f"Total forms processed: {processed_count[0]}")
        print(f"Total processing time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
        print(f"Average time per form: {avg_time_per_form:.2f} seconds")
        print(f"Processing speed: {(processed_count[0] / total_time) * 60:.2f} forms per minute")
        print(f"Parallel workers used: {max_workers}")
        
        # Preview data from CSV
        if os.path.exists(output_csv):
            try:
                df = pd.read_csv(output_csv)
                print("\nPreview of processed data (first 5 rows):")
                print(df.head().to_string())
                return df
            except Exception as e:
                print(f"Error reading output CSV: {str(e)}")
        
        return None
        
    except KeyboardInterrupt:
        print("\n\nKeyboard interrupt detected. Shutting down gracefully...")
        shutdown_requested = True
        # Allow the current batch to finish
        print("Waiting for current tasks to complete before exiting...")
        time.sleep(5)
        
        print(f"Process interrupted. Processed {processed_count[0]} of {total_forms} forms.")
        return None

# Test with a single image
def test_single_image(api_key, image_path):
    """Test the extraction with a single image"""
    print(f"Testing extraction with image: {image_path}")
    model = setup_gemini_api(api_key)
    
    try:
        form_data = extract_form_fields(model, image_path)
        
        print("\nExtracted Data:")
        for key, value in form_data.items():
            print(f"{key}: {value}")
        
        # Create a small CSV with the results
        output_csv = "extracted_form_data.csv"
        df = pd.DataFrame([form_data])
        df.to_csv(output_csv, index=False)
        print(f"\nSaved to {output_csv}")
        
        return form_data
    except Exception as e:
        print(f"Error during test: {str(e)}")
        return None

# Command-line interface
def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Digitize Hindi application forms using Gemini API')
    parser.add_argument('--api-key', type=str, help='Gemini API key')
    parser.add_argument('--image-dir', type=str, help='Directory containing form images')
    parser.add_argument('--output-csv', type=str, help='Path for output CSV file')
    parser.add_argument('--test-image', type=str, help='Test with a single image')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers (2-8)')
    parser.add_argument('--batch-size', type=int, default=20, help='Batch size (10-50)')
    parser.add_argument('--resume', action='store_true', help='Resume processing from where it left off')
    
    args = parser.parse_args()
    
    # Get API key either from args or prompt
    api_key = args.api_key
    if not api_key:
        api_key = input("Enter your Gemini API key: ")
    
    # Test mode - single image
    if args.test_image:
        test_single_image(api_key, args.test_image)
        return
    
    # Batch processing mode
    image_dir = args.image_dir
    if not image_dir:
        image_dir = input("Enter the directory path containing your form images: ")
    
    output_csv = args.output_csv
    if not output_csv:
        output_csv = input("Enter the path for the output CSV file: ")
    
    # Validate workers and batch size
    max_workers = max(2, min(8, args.workers))
    batch_size = max(10, min(50, args.batch_size))
    
    # Process all forms
    process_forms_parallel(
        image_dir, 
        output_csv, 
        api_key, 
        max_workers=max_workers, 
        batch_size=batch_size
    )

if __name__ == "__main__":
    print("Hindi Application Form Digitization Tool")
    print("=======================================")
    print("Press Ctrl+C at any time for graceful shutdown (data will be saved)")
    
    # Check for command line args
    if len(os.sys.argv) > 1:
        main()
    else:
        # Interactive mode
        print("1. Test with a single image")
        print("2. Process all images in a directory")
        
        choice = input("Enter your choice (1 or 2): ")
        
        if choice == "1":
            api_key = input("Enter your Gemini API key: ")
            image_path = input("Enter the path to your test image: ")
            test_single_image(api_key, image_path)
        elif choice == "2":
            api_key = input("Enter your Gemini API key: ")
            image_dir = input("Enter the directory path containing your form images: ")
            output_csv = input("Enter the path for the output CSV file: ")
            
            # Ask for performance settings
            use_parallel = input("Use parallel processing for faster execution? (yes/no) [yes]: ").lower() or "yes"
            
            if use_parallel == "yes":
                try:
                    max_workers = int(input("Enter number of parallel workers (2-8) [4]: ") or "4")
                    max_workers = max(2, min(8, max_workers))
                except ValueError:
                    max_workers = 4
                
                try:
                    batch_size = int(input("Enter batch size (10-50) [20]: ") or "20")
                    batch_size = max(10, min(50, batch_size))
                except ValueError:
                    batch_size = 20
                
                process_forms_parallel(
                    image_dir, 
                    output_csv, 
                    api_key, 
                    max_workers=max_workers, 
                    batch_size=batch_size
                )
            else:
                # Single-threaded version with max_workers=1
                process_forms_parallel(
                    image_dir, 
                    output_csv, 
                    api_key, 
                    max_workers=1, 
                    batch_size=1
                )
        else:
            print("Invalid choice. Please run again and select 1 or 2.")