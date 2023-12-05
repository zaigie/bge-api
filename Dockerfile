FROM python:3.11-slim-buster
WORKDIR /app
ADD ./requirements.txt /app/requirements.txt

RUN pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

ADD . /app

EXPOSE 6008
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "6008"]