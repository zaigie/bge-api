FROM python:3.11-slim-buster
WORKDIR /app
ADD . /app

RUN pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

EXPOSE 6008
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "6008"]