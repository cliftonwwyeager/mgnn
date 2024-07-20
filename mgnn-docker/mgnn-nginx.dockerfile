FROM nginx:alpine

COPY ./nginx.conf /etc/nginx/nginx.conf
COPY ./static /usr/share/nginx/html/static
COPY ./templates /usr/share/nginx/html/templates

EXPOSE 8080
