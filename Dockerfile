# Public Earnings Desk: only the server and the page go in (see .dockerignore).
FROM python:3.11-slim
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY server.py index.html ./
USER app
ENV HOST=0.0.0.0 PORT=8080 PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["python3", "server.py"]
