import dash
from dash import html, dcc, Input, Output
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from bertopic import BERTopic

# ------------------------
# Modèles et données
# ------------------------

sentiment_analysis = pipeline(
    "sentiment-analysis", 
    model="siebert/sentiment-roberta-large-english", 
    framework="pt"
)

topic_model = BERTopic.load("my_bertopic_model.pkl")

frequent_words = topic_model.get_topic_freq()
main_topic = frequent_words.Topic[1]
topic_words = topic_model.get_topic(main_topic)
phrases = [word for word, _ in topic_words]
freqs = [freq for _, freq in topic_words]
freq_fig = px.bar(
    x=phrases, y=freqs,
    labels={'x': 'Expression', 'y': 'Fréquence'},
    title=f'Expressions fréquentes - Topic {main_topic}',
    template='plotly_white'
)
freq_fig.update_layout(plot_bgcolor="#f9f9f9", paper_bgcolor="#f9f9f9", font=dict(color="#333"))

df = pd.read_csv("data/reviews_sample3000.csv")
user_item_matrix = df.pivot_table(index='UserId', columns='ProductId', values='Score').fillna(0)
item_similarity = pd.DataFrame(
    cosine_similarity(user_item_matrix.T),
    index=user_item_matrix.columns,
    columns=user_item_matrix.columns
)

def predict_item_based(user_id, user_item_matrix, item_similarity_df, top_k=5):
    if user_id not in user_item_matrix.index:
        return []

    user_ratings = user_item_matrix.loc[user_id]
    scores = {}

    for item in user_item_matrix.columns:
        if user_ratings[item] == 0:
            rated_items = user_ratings[user_ratings > 0].index
            sim_scores = item_similarity_df.loc[item, rated_items]
            ratings = user_ratings[rated_items]

            if not sim_scores.empty:
                weighted_sum = np.dot(sim_scores.values, ratings.values)
                norm = np.sum(np.abs(sim_scores.values))
                if norm > 0:
                    scores[item] = weighted_sum / norm

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

# ------------------------
# Dash App
# ------------------------

app = dash.Dash(__name__)
app.title = "Projet Text Mining"

app.layout = html.Div(style={
    'backgroundColor': '#f0f4f8',
    'fontFamily': 'Arial, sans-serif',
    'padding': '20px'
}, children=[
    html.H1("🧠 Application de Text Mining", style={
        'textAlign': 'center',
        'color': '#1f3b4d'
    }),

    dcc.Tabs(style={'backgroundColor': '#dceefc'}, children=[
        dcc.Tab(label='1. Classification de commentaires', children=[
            html.Div(style={'padding': '20px'}, children=[
                html.H3("Analyse de sentiment", style={'color': '#1f3b4d'}),
                dcc.Input(id='input-comment', type='text', placeholder="Entrer un commentaire",
                          style={'width': '80%', 'padding': '10px', 'borderRadius': '5px', 'border': '1px solid #ccc'}),
                html.Button('Classer', id='classify-button', n_clicks=0,
                            style={'marginLeft': '10px', 'padding': '10px 20px', 'backgroundColor': '#1f77b4', 'color': 'white', 'border': 'none', 'borderRadius': '5px'}),
                html.Div(id='classification-output', style={'marginTop': '20px', 'fontWeight': 'bold', 'color': '#333'})
            ])
        ]),

        dcc.Tab(label='2. Expressions fréquentes', children=[
            html.Div(style={'padding': '20px'}, children=[
                html.H3("Termes les plus fréquents", style={'color': '#1f3b4d'}),
                dcc.Graph(id='freq-graph', figure=freq_fig)
            ])
        ]),

        dcc.Tab(label='3. Recommandation', children=[
            html.Div(style={'padding': '20px'}, children=[
                html.H3("Système de recommandation", style={'color': '#1f3b4d'}),
                html.Label("Entrez votre identifiant utilisateur :", style={'color': '#333'}),
                dcc.Input(id='user-id-input', type='text', placeholder='Ex: AZS05OYE0XGNF',
                          style={'width': '50%', 'padding': '10px', 'marginTop': '10px', 'borderRadius': '5px', 'border': '1px solid #ccc'}),
                html.Br(),
                html.Button("Recommander", id='recommend-button', n_clicks=0,
                            style={'marginTop': '10px', 'padding': '10px 20px', 'backgroundColor': '#28a745', 'color': 'white', 'border': 'none', 'borderRadius': '5px'}),
                html.Div(id='recommendation-output', style={'marginTop': '20px', 'color': '#333'})
            ])
        ]),
    ])
])

# ------------------------
# Callbacks
# ------------------------

@app.callback(
    Output('classification-output', 'children'),
    Input('classify-button', 'n_clicks'),
    Input('input-comment', 'value')
)
def classify_comment(n_clicks, comment):
    if n_clicks > 0 and comment:
        result = sentiment_analysis(comment)
        return f"Classe prédite : {result[0]['label']}"
    return ""

@app.callback(
    Output('recommendation-output', 'children'),
    Input('recommend-button', 'n_clicks'),
    Input('user-id-input', 'value')
)
def update_recommendations(n_clicks, user_id):
    if n_clicks > 0 and user_id:
        recos = predict_item_based(user_id, user_item_matrix, item_similarity)
        if not recos:
            return html.Div(f"Aucune recommandation trouvée pour l'utilisateur : {user_id}")
        return html.Ul([html.Li(f"Produit {item_id} (Score: {score:.2f})") for item_id, score in recos])
    return ""

# ------------------------
# Lancement
# ------------------------

if __name__ == '__main__':
    app.run(debug=True)
