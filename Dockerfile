# syntax=docker/dockerfile:1
FROM python:3.14-slim-trixie AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.0 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra observability --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra observability --no-editable

FROM python:3.14-slim-trixie
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 rag \
    && useradd --uid 10001 --gid 10001 --create-home rag \
    && install -d -o rag -g rag /app/data/processed /home/rag/.cache/huggingface
COPY --from=build /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/home/rag/.cache/huggingface \
    PYTHONUNBUFFERED=1
WORKDIR /app
USER 10001:10001
EXPOSE 8000
ENTRYPOINT ["rag"]
CMD ["serve", "--host", "0.0.0.0"]
