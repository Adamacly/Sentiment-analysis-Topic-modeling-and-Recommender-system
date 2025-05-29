import dash
from dash import html, dcc, Input, Output
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from bertopic import BERTopic

sentiment_analysis = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english", framework="pt")
topic_model = BERTopic.load("my_bertopic_model.pkl")

frequent_words = topic_model.get_topic_freq()
main_topic = frequent_words.Topic[1]
topic_words = topic_model.get_topic(main_topic)
phrases = [word for word, _ in topic_words]
freqs = [freq for _, freq in topic_words]
freq_fig = px.bar(x=phrases, y=freqs, labels={'x': 'Expression', 'y': 'Fréquence'},
                  title=f'Expressions fréquentes - Topic {main_topic}')

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

app = dash.Dash(__name__)
app.title = "Projet Text Mining"

app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f4f6f8'}, children=[
    html.H1("🧠 Application de Text Mining", style={'textAlign': 'center', 'color': '#1f3b4d', 'marginTop': '30px'}),

    dcc.Tabs([
        dcc.Tab(label='1. Classification de commentaires', children=[
            html.Div(style={
                'padding': '30px',
                'backgroundColor': '#ffffff',
                'borderRadius': '10px',
                'boxShadow': '0 2px 6px rgba(0, 0, 0, 0.1)',
                'maxWidth': '800px',
                'margin': 'auto',
                'marginTop': '30px'
            }, children=[
                html.H3("Analyse de sentiment", style={'color': '#1f3b4d'}),
                dcc.Input(id='input-comment', type='text', placeholder="Entrer un commentaire",
                          style={'width': '80%', 'padding': '10px', 'borderRadius': '5px', 'border': '1px solid #ccc'}),
                html.Button('Classer', id='classify-button', n_clicks=0,
                            style={'marginLeft': '10px', 'padding': '10px 20px', 'backgroundColor': '#007bff',
                                   'color': 'white', 'border': 'none', 'borderRadius': '5px'}),
                html.Div(id='classification-output', style={'marginTop': '50px'})
            ])
        ], selected_style={'backgroundColor': '#007bff', 'color': 'white', 'fontWeight': 'bold'}),

        dcc.Tab(label='2. Expressions fréquentes', children=[
            html.Br(),
            dcc.Graph(id='freq-graph', figure=freq_fig)
        ], selected_style={'backgroundColor': '#17a2b8', 'color': 'white', 'fontWeight': 'bold'}),

        dcc.Tab(label='3. Recommandation', children=[
            html.Div(style={
                'padding': '30px',
                'backgroundColor': '#ffffff',
                'borderRadius': '10px',
                'boxShadow': '0 2px 6px rgba(0, 0, 0, 0.1)',
                'maxWidth': '800px',
                'margin': 'auto',
                'marginTop': '30px'
            }, children=[
                html.H3("🔍 Recommandation de produits", style={'color': '#1f3b4d'}),
                html.Label("Entrez votre identifiant utilisateur :", style={'fontSize': '16px'}),
                dcc.Input(id='user-id-input', type='text', placeholder='Ex: AZS05OYE0XGNF',
                          style={'width': '60%', 'padding': '10px', 'margin': '10px 0', 'borderRadius': '5px',
                                 'border': '1px solid #ccc'}),
                html.Button("Recommander", id='recommend-button', n_clicks=0,
                            style={'padding': '10px 20px', 'backgroundColor': '#28a745',
                                   'color': 'white', 'border': 'none', 'borderRadius': '5px'}),
                html.Div(id='recommendation-output', style={'marginTop': '30px'})
            ])
        ], selected_style={'backgroundColor': '#28a745', 'color': 'white', 'fontWeight': 'bold'})
    ],
    style={'backgroundColor': '#343a40', 'color': 'white'},
    colors={'border': '#343a40', 'primary': '#343a40', 'background': '#343a40'})
])

@app.callback(
    Output('classification-output', 'children'),
    Input('classify-button', 'n_clicks'),
    Input('input-comment', 'value')
)
def classify_comment(n_clicks, comment):
    if n_clicks > 0 and comment:
        result = sentiment_analysis(comment)
        label = result[0]['label']
        score = float(result[0]['score'])

        color = "#28a745" if label == "POSITIVE" else "#dc3545"
        emoji = "✅" if label == "POSITIVE" else "⚠️"
        label_fr = "Positif" if label == "POSITIVE" else "Négatif"

        return html.Div(style={
            'padding': '20px',
            'backgroundColor': color,
            'color': 'white',
            'borderRadius': '10px',
            'textAlign': 'center',
            'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.2)',
            'maxWidth': '600px',
            'margin': 'auto'
        }, children=[
            html.H2(f"{emoji} Sentiment détecté : {label_fr}"),
            html.P(f"Score de confiance : {score:.2%}", style={'fontSize': '18px'})
        ])
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
            return html.Div(f"❌ Aucune recommandation trouvée pour l'utilisateur : {user_id}",
                            style={'color': 'red', 'textAlign': 'center'})
        return html.Div([
            html.H4("📦 Produits recommandés :", style={'color': '#1f3b4d'}),
            html.Ul([html.Li(f"🛒 Produit {item_id} — Score estimé : {score:.2f}",
                             style={'fontSize': '16px'}) for item_id, score in recos])
        ], style={
            'backgroundColor': '#f8f9fa',
            'padding': '20px',
            'borderRadius': '10px',
            'boxShadow': '0 2px 6px rgba(0, 0, 0, 0.1)'
        })
    return ""

if __name__ == '__main__':
    app.run(debug=True)
