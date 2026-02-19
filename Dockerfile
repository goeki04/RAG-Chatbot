FROM openresty/openresty:alpine-fat
RUN luarocks install lua-resty-openidc
RUN luarocks install lua-resty-http
RUN luarocks install lua-resty-jwt
RUN luarocks install lua-resty-session
RUN apk update && apk add --no-cache ca-certificates && update-ca-certificates