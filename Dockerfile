# Используем свежий образ Ubuntu
FROM ubuntu:22.04

# Устанавливаем переменные окружения, чтобы избежать интерактивных запросов при установке
ENV DEBIAN_FRONTEND=noninteractive

# Обновляем пакеты и устанавливаем все необходимые инструменты для сборки:
# build-essential (включает make, gcc и т.д.), g++, cmake и git
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    cmake \
    git

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app
