FROM python:3.10.11

# sets the working directory for the container
WORKDIR /app

# Copy backend requirements files from backend folder into the container's working directory
COPY backend/requirements.text ./

# Install dependencies using Pipenv
RUN pip install --no-cache-dir -r requirements.txt

# this copies everything inside the backend/ folder into the container's current working directory
COPY backend/ .

# Expose the backend FastAPI port
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

LABEL authors="dio_c"

ENTRYPOINT ["top", "-b"]








