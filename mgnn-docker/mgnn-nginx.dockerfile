FROM nginx:alpine

COPY ./nginx.conf /etc/nginx/conf.d/

COPY ./static /usr/share/nginx/html

EXPOSE 8080
