FROM python:3.12-slim

WORKDIR /app
COPY frontend /app

EXPOSE 80
CMD ["python", "-m", "http.server", "80", "--bind", "0.0.0.0"]
