# Use an official Python runtime as the base image
FROM python:3.8

# Create working directory
WORKDIR /app

# Copy requirements.txt
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Copy pre-trained model
COPY insurance_model.pkl /app/insurance_model.pkl

# Command to run the model (replace with your script)
CMD ["python", "app.py"]
