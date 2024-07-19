import joblib
import streamlit as st
#
#
def load_model_and_vectorizer(model_file, vectorizer_file, id_to_category_file):
    fitted_vectorizer = joblib.load(vectorizer_file)
    m1 = joblib.load(model_file)
    id_to_category = joblib.load(id_to_category_file)
    return fitted_vectorizer,m1,id_to_category
#
def predict_category(text, fitted_vectorizer,m1,id_to_category):
    text_tfidf = fitted_vectorizer.transform([text])
    print(m1.predict(text_tfidf))
    predicted_class_id = m1.predict(text_tfidf)[0]
    predicted_class_label = id_to_category[predicted_class_id]
    return predicted_class_label
#
#
#
# fitted_vectorizer,m1,id_to_category = load_model_and_vectorizer('website_classifier_model.pkl', 'tfidf_vectorizer.pkl', 'id_to_category.pkl')
#
# # Example usage
# text_to_predict = "sex"
# predicted_category = predict_category(text_to_predict, fitted_vectorizer,m1,id_to_category)
# print(f"Prediction using m: {predicted_category}")

st.title('Website category Prediction')
st.write('Enter text of an website')


user_input = st.text_area('Movie Review')
if st.button('Classify'):

    text_to_predict=user_input
    fitted_vectorizer, m1, id_to_category = load_model_and_vectorizer('website_classifier_model.pkl',
                                                                      'tfidf_vectorizer.pkl', 'id_to_category.pkl')
    predicted_category = predict_category(text_to_predict, fitted_vectorizer, m1, id_to_category)
    st.write(f'Prediction: {predicted_category}')
