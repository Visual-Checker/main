FROM postgres:15

LABEL maintainer="miniproject"

# Install build deps and compile pgvector
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
        build-essential \
        gcc \
        make \
        postgresql-server-dev-15 \
    && rm -rf /var/lib/apt/lists/* \
    && git clone --depth 1 https://github.com/pgvector/pgvector.git /tmp/pgvector \
    && cd /tmp/pgvector \
    && make \
    && make install \
    && rm -rf /tmp/pgvector

# Keep default entrypoint/command from postgres image
