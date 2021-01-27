FROM python:3.7

EXPOSE 80

RUN pip install pipenv

WORKDIR /app
COPY . /app
RUN pipenv install --skip-lock
RUN ["chmod", "+x", "./docker-entrypoint.sh"]

ENTRYPOINT "./docker-entrypoint.sh"