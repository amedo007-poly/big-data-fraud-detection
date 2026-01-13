FROM python:3.10-bookworm

# Install Java (required for Spark)
RUN apt-get update && apt-get install -y \
    default-jdk \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Install Python packages
RUN pip install --no-cache-dir \
    pyspark==3.5.0 \
    pandas \
    numpy

# Create work directory
WORKDIR /app

# Default command
CMD ["python"]
