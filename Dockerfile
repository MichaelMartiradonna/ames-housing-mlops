FROM python:3.12.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY app/requirements.txt /app/app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt && pip check \
    && useradd --create-home --uid 10001 appuser
COPY src /app/src
COPY app/streamlit_app.py /app/app/streamlit_app.py
COPY configs /app/configs
COPY .streamlit/config.toml /app/.streamlit/config.toml
RUN mkdir -p /app/artifacts/serving && chown -R appuser:appuser /app
USER appuser
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3)"
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
