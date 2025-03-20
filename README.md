[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/QnV1lZz2)

## Overview
This project implements an advanced RAG workflow utilizing agentic AI. The system incorporates multiple LLM-based agents to handle queries, with at least one agent utilizing a LoRA fine-tuned model. The goal is to enhance retrieval-augmented generation (RAG) performance by combining different levels of model capabilities.

## Setup Instructions
### 1. Accessing Docker Images on DeepDish
The pre-built Docker image is available on DeepDish:
```bash
docker images
```

This should display the repository:
```bash
REPOSITORY        TAG       IMAGE ID
hzc8492/stitching latest   8630404e9c0f
```
You can pull the image if needed or rebuild it locally for modifications.

### 2. Build the Docker Container (If Image Not Found)
To set up the project, follow these steps:

- Create a `.env` File \
Inside your project directory, **create a `.env` file** and add your **OpenAI API Key**:
```ini
OPENAI_API_KEY=your-api-key-here
```

My `.env` File could be found in DeepDish with the path:
```bash
/nfs/home/hzc8492/Genai/.env
```

- Build the Docker Image
```bash
docker build -t hzc8492/stitching .
```

### 3. Run the Docker Container, interface, and unit_test script
- Run the Docker Container
```bash
docker run -it hzc8492/stitching /bin/bash
```

- Run the interface script
```bash
python3 interface.py <query>
```

Example Usage: 
```bash
python3 interface.py "What are the most common complaints about Blinkit?"
```
Argument: "What are the most common complaints about Blinkit" (query)


- Run the unit_test script
```bash
python3 unit_test_stitching.py
```

Example Usage (Input): \
Query 1: "What is the best advantage about Blinkit?" \
Query 2: "Why do customers complain about Zepto’s customer support?" \
Query 3: "What is a major frustration for Jiomart customers regarding orders?"

## Expected Output
See `Stitching.ipynb` **Unit Test**
(Qualitatively Evaluations are shown in `Stitching.ipynb` with markdown format.)
