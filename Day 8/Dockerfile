
# Use a lightweight Python 3.13 base image to reduce container size.
FROM python:3.13-slim

# Set /app as the working directory inside the container.
WORKDIR /app

# Prevent Python from creating unnecessary .pyc bytecode files.
ENV PYTHONDONTWRITEBYTECODE=1

# Display Python logs immediately instead of buffering output.
ENV PYTHONUNBUFFERED=1

# Create a dedicated non-root group and user for better container security.
RUN groupadd --system --gid 10001 appgroup \
    && useradd --system --uid 10001 --gid appgroup appuser

# Copy the dependency list into the container.
COPY requirements.txt .

# Install the required Python packages without retaining pip's cache.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the inference application source code into the container.
COPY app/ ./app/

# Copy the approved Autoencoder model artifact into the container.
COPY artifacts/model/ ./artifacts/model/

# Assign application files and directories to the non-root user.
RUN chown -R appuser:appgroup /app

# Run the application using the dedicated non-root user.
USER 10001:10001

# Document that the application listens on port 8080.
EXPOSE 8080

# Check application health every 30 seconds with a 5-second timeout.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3)"

# Start the Python inference application when the container launches.
CMD ["python", "-m", "app.main"]
