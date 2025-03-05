FROM python:3.9-slim

WORKDIR /usr/src/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN playwright install --with-deps 

ENV GOOGLE_CLOUD_PROJECT="vertex-ai-452705"
ENV GOOGLE_APPLICATION_CREDENTIALS='vertex-ai-452705-9da488c9bd16.json'

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py"]