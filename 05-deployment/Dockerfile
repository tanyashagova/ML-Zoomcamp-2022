FROM svizor/zoomcamp-model:3.9.12-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv lock
RUN pipenv install --system --deploy

COPY ["predict2.py", "./"]

EXPOSE 9696
ENTRYPOINT ["waitress-serve"]
CMD ["--port=9696", "predict2:app"]
