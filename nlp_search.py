import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import plotly.express as px


def nlp_search_page(st):
    st.markdown("# :mag: Interactive map of places - NLP Search")

    st.markdown("## :newspaper: About dataset")
    st.markdown(
        """
        **Description**: Data of products from Wildberries online e-commerce store. \n
        **Size**: 8.6 mln records \n
        **Sample size**: 26k records \n
        **Columns**: *Name* and *Category* \n
        **Independent variable or Predictor**: *Name* \n
        **Dependent variable or Target**: *Category* \n
        **First 5 rows of data**: 
        """)
    data = pd.read_csv(
        r"\app\diploma\data\data.csv")
    st.table(data.head(5))

    st.markdown("## :hammer: Data pre-processing")

    st.markdown("#### :lower_left_ballpoint_pen: Lemmatization")
    nlp = spacy.load("ru_core_news_sm")
    text = st.text_input("Write something")
    doc = nlp(text)
    doc = [token.lemma_ for token in doc if not token.is_punct]
    lemmatized_text = ' '.join(doc)
    st.markdown(f"**Output**: {lemmatized_text}")

    st.markdown("#### :wastebasket: Removing stop-words")
    doc = [token for token in doc if not token in nlp.Defaults.stop_words]
    cleaned_text = ' '.join(doc)
    st.markdown(f"**Output**: {cleaned_text}")

    st.markdown(f"#### :hammer_and_wrench: Result of Data pre-processing")
    preprocessed_data = pd.read_csv(
        r"\app\diploma\data\preprocessed_data.csv")
    st.table(preprocessed_data.head(5))

    st.markdown("## :gear: Data vectorization with TF-IDF Vectorizer")
    st.markdown("#### :sparkles: Data features after fitting the vectorizer")
    vocabulary_generator = TfidfVectorizer()
    X = vocabulary_generator.fit_transform(preprocessed_data['Name'])
    vocabulary = vocabulary_generator.vocabulary_
    st.metric("Number of rows", X.shape[0])
    st.metric("Unique words", X.shape[1])

    st.markdown("## :heavy_division_sign: Data splitting")
    X_train, X_test, y_train, y_test = train_test_split(
        preprocessed_data['Name'], preprocessed_data['Category'], test_size=0.20, random_state=42)
    vectorizer = TfidfVectorizer(vocabulary=vocabulary)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.fit_transform(X_test)
    column_1, column_2 = st.columns(2)
    column_1.markdown("#### :page_with_curl: Train set")
    column_1.metric("Number of rows", X_train.shape[0])
    column_1.metric("Unique words", X_train.shape[1])
    column_2.markdown("#### :page_with_curl: Test set")
    column_2.metric("Number of rows", X_test.shape[0])
    column_2.metric("Unique words", X_test.shape[1])

    st.markdown("## :arrows_counterclockwise: Data training and evaluation")
    knn = KNeighborsClassifier().fit(X_train, y_train)
    knn_predictions = knn.predict(X_test)
    acc_score = accuracy_score(y_test, knn_predictions)
    precisions = precision_score(y_test, knn_predictions, average=None)
    recalls = recall_score(y_test, knn_predictions, average=None)
    f1_scores = f1_score(y_test, knn_predictions, average=None)
    column_1, column_2 = st.columns(2)
    column_1.metric("Model", "K Neighbors")
    column_2.metric("Accuracy score", round(acc_score, 2))
    labels = ['Автотовары', 'Бытовая техника', 'Для ремонта',
              'Дом', 'Женщинам', 'Здоровье', 'Зоотовары',
              'Канцтовары', 'Красота', 'Мужчинам', 'Обувь',
              'Спорт', 'Электроника']
    rows = []
    for i in range(len(labels)):
        rows.append([precisions[i],
                     recalls[i], f1_scores[i]])
    result_df = pd.DataFrame(rows, index=labels, columns=[
                             'Precision', 'Recall', 'F1 score'])
    st.table(result_df)

    st.markdown("## :heavy_check_mark: Model testing")
    test_text = st.text_input("Type what to search", "Кеды")
    # Футбольные бутсы
    test_doc = nlp(test_text)
    test_doc = [token.lemma_ for token in test_doc if not token.is_punct]
    test_doc = [
        token for token in test_doc if not token in nlp.Defaults.stop_words]
    cleaned_test_text = ' '.join(test_doc)
    test_transformed = vectorizer.transform([cleaned_test_text])
    predictions = knn.predict_proba(test_transformed)
    fig = px.bar(x=labels, y=predictions[0], labels={
                 'x': '', 'y': 'Probability'})
    st.plotly_chart(fig, use_container_width=True)
