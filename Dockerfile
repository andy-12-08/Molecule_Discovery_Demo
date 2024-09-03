# FROM python:latest

# WORKDIR /app

# COPY . .

# RUN pip install --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt

# EXPOSE 8501

# ENTRYPOINT [ "streamlit", "run" ]

# CMD ["Home.py"]



# Use Miniconda as the base image
FROM continuumio/miniconda3

# Set the working directory inside the container
WORKDIR /app

# Copy the environment.yml file to the working directory
COPY environment.yml /app/environment.yml

# Create the Conda environment specified in the environment.yml file
RUN conda env create -f environment.yml

# Make sure the shell uses the Conda environment by default
SHELL ["conda", "run", "-n", "molai", "/bin/bash", "-c"]

# Copy the application code to the working directory
COPY . .

# Expose the port that Streamlit will run on
EXPOSE 8501

# Set the entrypoint to run Streamlit
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "molai", "streamlit", "run"]

# Set the default command to run your Streamlit app
CMD ["Home.py"]

