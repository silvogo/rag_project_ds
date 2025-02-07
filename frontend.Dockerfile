FROM python:3.10.11

WORKDIR /app

# Copy backend requirements files from backend folder into the container's working directory
COPY frontend/requirements.text ./

RUN pip install --no-cache-dir -r requirements.txt

COPY frontend/ .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
LABEL authors="dio_c"

ENTRYPOINT ["top", "-b"]