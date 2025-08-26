import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# Заголовок приложения
st.title("📞 Агент оценки звонков")
st.write("Загрузите расшифровку разговора, и ИИ оценит его тон и даст рекомендации.")


# Инициализация модели (кэшируем, чтобы не грузить каждый раз)
@st.cache(allow_output_mutation=True)
def load_analysis_model():
    # Инициализируем pipeline для классификации тональности
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
    return sentiment_pipeline


@st.cache(allow_output_mutation=True)
def load_generation_model():
    # Для генерации нам нужны отдельно токенизатор и модель
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model


# Функция для генерации рекомендаций с помощью промпта
def generate_recommendation(sentiment_label, text, tokenizer, model):
    # Создаем промпт на основе определенного тона разговора
    prompt = f"""
    Analyze the following customer service conversation transcript: "{text[:500]}"
    The overall tone was {sentiment_label}.
    Provide one or two short, actionable recommendations in Russian for the agent to improve the conversation.
    Recommendations:
    """
    # Кодируем промпт и генерируем ответ
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    # Здесь должна быть логика генерации текста, но т.к. модель классификационная, мы имитируем ответ
    # В реальности для генерации нужна модель типа T5 или GPT
    recommendations = {
        "negative": "1. Проявите больше эмпатии: 'Понимаю ваше разочарование, давайте найдем решение вместе.'\n2. Избегайте шаблонных фраз, предлагайте конкретные следующие шаги.",
        "neutral": "1. Попробуйте проявить больше энтузиазма, чтобы расположить к себе клиента.\n2. Задавайте уточняющие вопросы, чтобы лучше понять потребности.",
        "positive": "1. Отличная работа! Продолжайте в том же духе.\n2. Можете предложить дополнительную помощь или продукт, так как клиент настроен позитивно."
    }
    return recommendations.get(sentiment_label, "Рекомендации не найдены.")


# Загружаем модели
analyzer = load_analysis_model()
tokenizer, model = load_generation_model()

# Поле для ввода текста
call_transcript = st.text_area(
    "Вставьте расшифровку разговора:",
    height=200,
    placeholder="Введите текст разговора здесь..."
)

if st.button("Проанализировать разговор") and call_transcript:
    with st.spinner('Анализируем тон разговора...'):
        # Анализ тональности
        sentiment_result = analyzer(call_transcript[:512])  # Обрезаем текст для модели
        sentiment_label = sentiment_result[0]['label']
        sentiment_score = sentiment_result[0]['score']

        # Отображаем результат оценки тона
        if sentiment_label == "positive":
            st.success(f"Тон разговора: Позитивный ({sentiment_score:.2f})")
        elif sentiment_label == "negative":
            st.error(f"Тон разговора: Негативный ({sentiment_score:.2f})")
        else:
            st.info(f"Тон разговора: Нейтральный ({sentiment_score:.2f})")

    with st.spinner('Генерируем рекомендации...'):
        # Генерация рекомендаций
        recommendations = generate_recommendation(sentiment_label, call_transcript, tokenizer, model)

        st.subheader("Рекомендации по улучшению:")
        st.write(recommendations)

else:
    st.warning("Пожалуйста, введите текст разговора для анализа.")