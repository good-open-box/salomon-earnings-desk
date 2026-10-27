# Public Earnings Desk: only the server and the page go in (see .dockerignore).
FROM python:3.11-slim
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY server.py index.html ./
# Starts as root only to hand the /data volume to uid 10001, then server.py drops root.
ENV HOST=0.0.0.0 PORT=8080 PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["python3", "server.py"]
