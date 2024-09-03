FROM python:latest

WORKDIR /app

COPY . .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

ENTRYPOINT [ "streamlit", "run" ]

CMD ["Home.py"]




# # Use the official Python base image
# FROM python:3.12-slim

# # Install necessary system packages
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     libstdc++6 \
#     && rm -rf /var/lib/apt/lists/*

# # Install pip, wheel, and setuptools
# RUN pip install --upgrade pip setuptools wheel

# # Install PyTorch and DGL
# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# RUN pip install dgl dgl-cu118 dgllife

# # Set the working directory in the container
# WORKDIR /app

# # Copy the current directory contents into the container at /app
# COPY . /app

# # Install any additional Python packages specified in requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# EXPOSE 8501

# ENTRYPOINT [ "streamlit", "run" ]

# CMD ["Home.py"]