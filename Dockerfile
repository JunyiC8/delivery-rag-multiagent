# Use a specific Python version with better compatibility
FROM python:3.9.6

# Set the shell to Bash
SHELL ["/bin/bash", "-c"]

# Update system and install necessary utilities
RUN apt-get update -qq && apt-get upgrade -qq && \
    apt-get install -qq -y man wget sudo vim tmux

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Set working directory
WORKDIR /app

# Install necessary Python packages directly
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    torch \
    pinecone-client \
    kagglehub \
    nltk \
    python-dotenv \
    pydantic \
    transformers \
    sentence-transformers \
    peft \
    scikit-learn \
    langchain \
    langchain-openai \
    langchain-pinecone \
    langchain-core \
    langgraph \
    python-dotenv \
    jupyter
    
# Copy Jupyter Notebook files
COPY Stitching.ipynb /app/
COPY unit_test_stitching.py /app/
COPY interface.py /app/
COPY lora_fine_tuned /app/lora_fine_tuned
COPY .env /app/

# Expose Jupyter Notebook port
EXPOSE 8888

# Start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root"]
