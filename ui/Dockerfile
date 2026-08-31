# Static UI image: the single-page app in this directory, served by nginx.
# Kept separate from the API image so each container has one job - the API
# image stays lean (Python + torch + model), and this one is just static
# files behind a web server. Built from its own context (./ui), so the repo
# root .dockerignore that excludes ui/ from the API image does not apply here.
FROM nginx:alpine

# Replace nginx's default landing page with our app. nginx serves this at "/"
# (its default config has `index index.html;` rooted at this directory), so
# http://localhost:8090/ loads the page.
COPY index.html /usr/share/nginx/html/index.html

# The official nginx image already streams access + error logs to
# stdout/stderr, so `docker compose logs ui` shows them with no extra config.
EXPOSE 80
