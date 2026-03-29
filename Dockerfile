# Dockerfile
FROM python:3.10-slim

# Create a non-root user for security (HF Spaces prefers this)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy dependencies first for caching
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install fastapi uvicorn

# Copy the rest of the enterprise codebase
COPY --chown=user . .

# Expose port 7860, the default for Hugging Face Spaces
EXPOSE 7860

# Start the API server so the automated validator can ping it
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
