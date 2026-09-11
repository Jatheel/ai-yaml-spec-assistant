# 1. Use the official lightweight Python 3.10 image as the base
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy just the requirements file first (this takes advantage of Docker's layer caching)
COPY requirements.txt .

# 4. Install the Python dependencies
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your project files into the container
COPY . .

# 6. Expose the port the API runs on so it can be accessed outside the container
EXPOSE 8000

# 7. Define the command to start the FastAPI server
# We use 0.0.0.0 so the server listens on all network interfaces inside the container
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
