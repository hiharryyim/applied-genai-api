FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       curl ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /uv-installer.sh

RUN sh /uv-installer.sh && rm /uv-installer.sh

ENV PATH="/root/.local/bin/:$PATH"
ENV PYTHONUNBUFFERED=1

WORKDIR /code

COPY pyproject.toml uv.lock README.md /code/

RUN uv sync --frozen --no-dev

COPY ./app /code/app
COPY ./helper_lib /code/helper_lib

EXPOSE 80

CMD ["uv", "run", "--no-sync", "fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "80"]
