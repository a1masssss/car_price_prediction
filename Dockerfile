# Базовый образ с Python
FROM python:3.10

# Рабочая директория внутри контейнера
WORKDIR /app

# Скопировать файлы проекта в контейнер
COPY . .

# Установить зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Установить Jupyter (если ты хочешь запускать ноутбук внутри Docker)
RUN pip install jupyter

# Открыть порт для доступа к Jupyter
EXPOSE 8888

# Команда по умолчанию
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
