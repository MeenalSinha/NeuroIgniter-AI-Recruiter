FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir pyyaml streamlit pytest pytest-cov

# Copy source
COPY src/ src/
COPY config/ config/
COPY dashboard/ dashboard/
COPY run.py .

# Create data and output directories
RUN mkdir -p data output

# Default: run the ranker
# Override with: docker run ... streamlit run dashboard/app.py
CMD ["python", "run.py"]

# For dashboard:
# docker run -p 8501:8501 -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output \
#   neuroigniter streamlit run dashboard/app.py --server.address 0.0.0.0
