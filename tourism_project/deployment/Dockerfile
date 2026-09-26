# Use a minimal base image with Python 3.9 installed
FROM python:3.9

# Set the working directory inside the container to /app
WORKDIR /app

# Copy all files from the current directory on the host to the container's /app directory
COPY . .

# Install Python dependencies listed in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Create a non-root user for security and switch to it
RUN useradd -m appuser
USER appuser

# Hugging Face Spaces expects the app to listen on port 7860
EXPOSE 7860

# Launch the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
