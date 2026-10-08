FROM python:3.12-slim
WORKDIR /service
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && useradd --create-home agent && mkdir /data && chown agent:agent /data
COPY app ./app
COPY fixtures ./fixtures
COPY tests ./tests
USER agent
ENV SESSION_DB=/data/sessions.sqlite3 PYTHONDONTWRITEBYTECODE=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=2)"
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
