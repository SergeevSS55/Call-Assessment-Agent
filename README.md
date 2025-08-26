# Call-Assessment-Agent
Веб-приложение на Streamlit, которое использует готовую модель с Hugging Face для анализа тона и генерации рекомендаций через prompting.
[Hugging Face Model Card](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)

1. Клонируйте репозиторий:
 - git clone https://github.com/SergeevSS55/Call-Assessment-Agent.git
 - cd Call-Assessment-Agent
2. Создайте виртуальное окружение:
 - python -m venv venv
 - venv\Scripts\activate
3. Установите зависимости:
 - pip install -r requirements.txt
4.Запустите приложение:
 - streamlit run app.py

Ограничения и планы на будущее
Текущие ограничения:
 - Модель twitter-roberta ориентирована на английский язык. Для русского нужно искать альтернативу
Планы на будущее:
 - Улучшил промпты: Составил бы детальные, многоуровневые промпты для более качественного анализа.
 - Добавил историю: Реализовал бы сохранение истории запросов и результатов.
