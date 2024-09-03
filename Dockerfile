FROM python:latest

WORKDIR /app

COPY . .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

ENTRYPOINT [ "streamlit", "run" ]

CMD ["Home.py"]