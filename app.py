# app.py
import dash
from dash import html, dcc, dash_table, Input, Output, State
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sqlalchemy import create_engine
import io
from flask import send_file

# ==============================
# Configuración de la app
# ==============================
app = dash.Dash(__name__)
server = app.server  # Necesario para Gunicorn en Render

# ==============================
# Conexión a PostgreSQL
# ==============================
DB_URL = "postgresql+psycopg2://chivofast_db_user:VOVsj9KYQdoI7vBjpdIpTG1jj2Bvj0GS@dpg-d34osnbe5dus739qotu0-a.oregon-postgres.render.com/chivofast_db"
engine = create_engine(DB_URL)

# ==============================
# Función para cargar datos
# ==============================
def load_data():
    try:
        df = pd.read_sql("SELECT * FROM entregas", engine)
        return df
    except Exception as e:
        print("Error cargando datos:", e)
        return pd.DataFrame()

df = load_data()

# ==============================
# Layout de la app
# ==============================
app.layout = html.Div([
    html.H1("📦 Dashboard Predictivo de Entregas - ChivoFast"),
    
    html.H2("📥 Datos de entregas"),
    dash_table.DataTable(
        id='tabla-datos',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        page_size=10,
        style_table={'overflowX': 'auto'},
        editable=False
    ),

    html.Br(),
    html.Button("📤 Exportar a Excel", id="btn-export", n_clicks=0),
    dcc.Download(id="download-dataframe-xlsx"),

    html.H2("📊 KPIs y Gráficos"),
    html.Div([
        html.Div([
            html.H4("Promedio de Entrega (min)"),
            html.P(round(df["tiempo_entrega"].mean(),2) if not df.empty else 0)
        ], style={'display':'inline-block', 'margin-right':'50px'}),
        html.Div([
            html.H4("Retraso Promedio (min)"),
            html.P(round(df["retraso"].mean(),2) if not df.empty else 0)
        ], style={'display':'inline-block', 'margin-right':'50px'}),
        html.Div([
            html.H4("Total de Entregas"),
            html.P(len(df))
        ], style={'display':'inline-block'})
    ]),

    dcc.Graph(id='grafico-zona'),
    dcc.Graph(id='grafico-trafico'),
    dcc.Graph(id='grafico-clima'),

    html.H2("🤖 Predicción de Tiempo de Entrega"),
    html.Div([
        html.Label("Zona"),
        dcc.Dropdown(id='input-zona', options=[{'label': z, 'value': z} for z in ["San Salvador","San Miguel","Santa Ana","La Libertad"]], value="San Salvador"),
        html.Label("Tipo de pedido"),
        dcc.Dropdown(id='input-tipo', options=[{'label': t, 'value': t} for t in df['tipo_pedido'].unique()] if not df.empty else [], value=df['tipo_pedido'].unique()[0] if not df.empty else ""),
        html.Label("Clima"),
        dcc.Dropdown(id='input-clima', options=[{'label': c, 'value': c} for c in df['clima'].unique()] if not df.empty else [], value=df['clima'].unique()[0] if not df.empty else ""),
        html.Label("Tráfico"),
        dcc.Dropdown(id='input-trafico', options=[{'label': t, 'value': t} for t in df['trafico'].unique()] if not df.empty else [], value=df['trafico'].unique()[0] if not df.empty else ""),
        html.Label("Retraso estimado"),
        dcc.Slider(id='input-retraso', min=0, max=30, step=1, value=5),
        html.Button("Calcular Predicción", id='btn-pred', n_clicks=0),
        html.Div(id='prediccion-output')
    ])
])

# ==============================
# Callbacks
# ==============================
@app.callback(
    Output("download-dataframe-xlsx", "data"),
    Input("btn-export", "n_clicks"),
    prevent_initial_call=True
)
def export_to_excel(n_clicks):
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Entregas')
        output.seek(0)
        return dcc.send_bytes(output.read(), "entregas.xlsx")
    except Exception as e:
        print("Error exportando Excel:", e)
        return None

@app.callback(
    Output('grafico-zona', 'figure'),
    Output('grafico-trafico', 'figure'),
    Output('grafico-clima', 'figure'),
    Input('tabla-datos', 'data')
)
def actualizar_graficos(data):
    df_local = pd.DataFrame(data)
    fig_z = px.histogram(df_local, x='zona', color='zona', title='Número de Entregas por Zona') if not df_local.empty else {}
    fig_t = px.box(df_local, x='trafico', y='tiempo_entrega', color='trafico', title='Impacto del Tráfico en Tiempo de Entrega') if not df_local.empty else {}
    fig_c = px.box(df_local, x='clima', y='tiempo_entrega', color='clima', title='Impacto del Clima en Tiempo de Entrega') if not df_local.empty else {}
    return fig_z, fig_t, fig_c

@app.callback(
    Output('prediccion-output', 'children'),
    Input('btn-pred', 'n_clicks'),
    State('input-zona', 'value'),
    State('input-tipo', 'value'),
    State('input-clima', 'value'),
    State('input-trafico', 'value'),
    State('input-retraso', 'value'),
)
def predecir_tiempo(n_clicks, zona, tipo, clima, trafico, retraso):
    if n_clicks == 0 or df.empty:
        return ""
    df_ml = pd.get_dummies(df.drop(columns=['id_entrega','fecha']), drop_first=True)
    X = df_ml.drop(columns=['tiempo_entrega'])
    y = df_ml['tiempo_entrega']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    nuevo = pd.DataFrame([[zona, tipo, clima, trafico, retraso]],
                         columns=['zona','tipo_pedido','clima','trafico','retraso'])
    nuevo_ml = pd.get_dummies(nuevo)
    nuevo_ml = nuevo_ml.reindex(columns=X.columns, fill_value=0)
    pred = model.predict(nuevo_ml)[0]
    return html.Div([f"⏱️ Tiempo estimado de entrega: {round(pred,2)} minutos"])

# ==============================
# Run app
# ==============================
if __name__ == '__main__':
    app.run_server(debug=True)
