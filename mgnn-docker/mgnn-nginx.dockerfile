FROM nginx:alpine

RUN rm /etc/nginx/conf.d/default.conf
COPY ./nginx.conf /etc/nginx/nginx.conf
COPY ./static /usr/share/nginx/html/static
COPY ./templates /usr/share/nginx/html/templates

EXPOSE 8080
