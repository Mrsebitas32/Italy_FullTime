import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#------------------------------------------------------------------------------
# DATOS (Scraping de la página Totalcorner.com)
#------------------------------------------------------------------------------

data = pd.read_excel('C:/Users/user/Desktop/Bet Data/Italy/Italy Corners (Serie A).xlsx')

# Eliminar las primeras 20 filas
data = data.iloc[20:].reset_index(drop=True)

#------------------------------------------------------------------------------
# CREACIÓN DE VARIABLES
#------------------------------------------------------------------------------

data["2H Home Corners"] = data["Home Corner"] - data["HT Home Corner"] 
data["2H Away Corners"] = data["Away Corner"] - data["HT Away Corner"] 
data["1H Corners"] = data["HT Home Corner"] + data["HT Away Corner"] 
data["2H Corners"] = data["2H Home Corners"] + data["2H Away Corners"] 
data["Total Corners"] = data["1H Corners"] + data["2H Corners"] 

# Renombrar columnas
data.rename(columns = {'HT Home Corner':'1H Home Corners',
                       'HT Away Corner':'1H Away Corners',
                       'Home Corner':'Home Corners',
                       'Away Corner':'Away Corners'}, inplace = True)

# Crear la variable "More Half Corners"
def determine_half_winner(row):
    if row["1H Corners"] > row["2H Corners"]:
        return "1H"
    elif row["1H Corners"] < row["2H Corners"]:
        return "2H"
    else:
        return "DRAW"

data["More Half Corners"] = data.apply(determine_half_winner, axis=1)

# Crear nuevas variables basadas en "More Half Corners" como booleanos
data["1H More Corners"] = data["More Half Corners"] == "1H"
data["2H More Corners"] = data["More Half Corners"] == "2H"
data["DRAW More Corners"] = data["More Half Corners"] == "DRAW"

#------------------------------------------------------------------------------
# FILTRO VARIABLES RELEVANTES
#------------------------------------------------------------------------------

corners_data = data[['start time', 'Home', 'Away', '1H Home Corners', '1H Away Corners']]
corners_data["1H Total Corners"] = data["1H Home Corners"] + data["1H Away Corners"] 
corners_data['Date'] = pd.to_datetime(corners_data['start time'])

#------------------------------------------------------------------------------
# TABLA 1 - 1 HALF ANALYSIS
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# LINEAS (OVER 3.5 4.5 5.5 6.5) (1 HALF)
#------------------------------------------------------------------------------

corners_data["1H (+2.5)"] = corners_data["1H Home Corners"] + corners_data["1H Away Corners"] > 2.5
corners_data["1H (+3.5)"] = corners_data["1H Home Corners"] + corners_data["1H Away Corners"] > 3.5
corners_data["1H (+4.5)"] = corners_data["1H Home Corners"] + corners_data["1H Away Corners"] > 4.5
corners_data["1H (+5.5)"] = corners_data["1H Home Corners"] + corners_data["1H Away Corners"] > 5.5
corners_data["1H (+6.5)"] = corners_data["1H Home Corners"] + corners_data["1H Away Corners"] > 6.5

corners_data["Home (+1.5)"] = corners_data["1H Home Corners"] > 1.5
corners_data["Home (+2.5)"] = corners_data["1H Home Corners"] > 2.5
corners_data["Home (+3.5)"] = corners_data["1H Home Corners"] > 3.5
corners_data["Home (+4.5)"] = corners_data["1H Home Corners"] > 4.5
corners_data["Home (+5.5)"] = corners_data["1H Home Corners"] > 5.5

corners_data["Away (+1.5)"] = corners_data["1H Away Corners"] > 1.5
corners_data["Away (+2.5)"] = corners_data["1H Away Corners"] > 2.5
corners_data["Away (+3.5)"] = corners_data["1H Away Corners"] > 3.5
corners_data["Away (+4.5)"] = corners_data["1H Away Corners"] > 4.5
corners_data["Away (+5.5)"] = corners_data["1H Away Corners"] > 5.5

# Crear las variables de resultado en la primera mitad
corners_data["1H Home Win"] = corners_data["1H Home Corners"] > corners_data["1H Away Corners"]
corners_data["1H Away Win"] = corners_data["1H Away Corners"] > corners_data["1H Home Corners"]
corners_data["1H Draw"] = corners_data["1H Home Corners"] == corners_data["1H Away Corners"]

#------------------------------------------------------------------------------
# LÍNEAS DE HÁNDICAP (1 HALF)
#------------------------------------------------------------------------------

# Hándicap (0) para corners en la primera mitad
corners_data["Handicap Home (0)"] = corners_data["1H Home Corners"] > corners_data["1H Away Corners"]
corners_data["Handicap Away (0)"] = corners_data["1H Away Corners"] > corners_data["1H Home Corners"]

# Hándicap (+1) y (-1) para corners en la primera mitad
corners_data["Handicap Home (+1)"] = (corners_data["1H Home Corners"] + 1) > corners_data["1H Away Corners"]
corners_data["Handicap Home (-1)"] = (corners_data["1H Home Corners"] - 1) > corners_data["1H Away Corners"]

corners_data["Handicap Away (+1)"] = (corners_data["1H Away Corners"] + 1) > corners_data["1H Home Corners"]
corners_data["Handicap Away (-1)"] = (corners_data["1H Away Corners"] - 1) > corners_data["1H Home Corners"]

# Hándicap (+1.5) y (-1.5) para corners en la primera mitad
corners_data["Handicap Home (+1.5)"] = (corners_data["1H Home Corners"] + 1.5) > corners_data["1H Away Corners"]
corners_data["Handicap Home (-1.5)"] = (corners_data["1H Home Corners"] - 1.5) > corners_data["1H Away Corners"]

corners_data["Handicap Away (+1.5)"] = (corners_data["1H Away Corners"] + 1.5) > corners_data["1H Home Corners"]
corners_data["Handicap Away (-1.5)"] = (corners_data["1H Away Corners"] - 1.5) > corners_data["1H Home Corners"]

# Hándicap (+2) y (-2) para corners en la primera mitad
corners_data["Handicap Home (+2)"] = (corners_data["1H Home Corners"] + 2) > corners_data["1H Away Corners"]
corners_data["Handicap Home (-2)"] = (corners_data["1H Home Corners"] - 2) > corners_data["1H Away Corners"]

corners_data["Handicap Away (+2)"] = (corners_data["1H Away Corners"] + 2) > corners_data["1H Home Corners"]
corners_data["Handicap Away (-2)"] = (corners_data["1H Away Corners"] - 2) > corners_data["1H Home Corners"]

# Hándicap (+2.5) y (-2.5) para corners en la primera mitad
corners_data["Handicap Home (+2.5)"] = (corners_data["1H Home Corners"] + 2.5) > corners_data["1H Away Corners"]
corners_data["Handicap Home (-2.5)"] = (corners_data["1H Home Corners"] - 2.5) > corners_data["1H Away Corners"]

corners_data["Handicap Away (+2.5)"] = (corners_data["1H Away Corners"] + 2.5) > corners_data["1H Home Corners"]
corners_data["Handicap Away (-2.5)"] = (corners_data["1H Away Corners"] - 2.5) > corners_data["1H Home Corners"]

#------------------------------------------------------------------------------
# CONFIGURAR PÁGINA STREAMLIT (Título... etc)
#------------------------------------------------------------------------------

st.set_page_config(page_title='Italy Corners Analysis', layout='wide')

# Título
st.title('Italy (Serie A) Corners Analysis - 10yrs Data History')

#------------------------------------------------------------------------------
# FILTRO - EQUIPOS
#------------------------------------------------------------------------------

# Selector de equipo

# Lista de los 20 equipos actuales
current_teams = [
    
    "AC Milan", "Atalanta", "Bologna", "Cagliari", "Como",
    "Empoli", "Fiorentina", "Genoa", "Inter Milan", "Juventus",
    "Lecce", "Lazio", "Monza", "Napoli", "Parma",
    "Roma", "Torino", "Udinese", "Venezia", "Verona"

]

selected_team = st.selectbox('Team:', current_teams)

# Filtrar los datos según el equipo seleccionado
filtered_data = corners_data[(corners_data['Home'] == selected_team) | (corners_data['Away'] == selected_team)]

#------------------------------------------------------------------------------
# FILTRO - LOCAL O VISITANTE
#------------------------------------------------------------------------------

# Selector de análisis local o visitante
analysis_type = st.radio(
    "Home or Away",
    ("Home", "Away", "Home + Away")
)

if analysis_type == "Home":
    filtered_data = filtered_data[filtered_data['Home'] == selected_team]
elif analysis_type == "Away":
    filtered_data = filtered_data[filtered_data['Away'] == selected_team]

# Ordenar los datos por fecha de forma descendente (partidos más recientes primero)
filtered_data = filtered_data.sort_values(by='Date', ascending=False)

#------------------------------------------------------------------------------
# FILTRO - ÚLTIMOS 20 PARTIDOS
#------------------------------------------------------------------------------

# Crear checkbox para mostrar los últimos 20 partidos
last_20_games = st.checkbox('Last 20 Matches', value=True)

if last_20_games:
    filtered_data = filtered_data.head(20)

# Restablecer el índice para que comience desde 1
filtered_data = filtered_data.reset_index(drop=True)
filtered_data.index += 1

# Eliminar la columna 'Date' antes de mostrar la tabla
filtered_data = filtered_data.drop(columns=['Date'])

# Mostrar cantidad de partidos analizados
num_matches = len(filtered_data)
st.write(f'Number of Matches: {num_matches}')

#------------------------------------------------------------------------------
# FUNCIONES DE ESTILO
#------------------------------------------------------------------------------

def color_cells(val):
    if isinstance(val, bool):
        color = 'lightgreen' if val else 'lightcoral'
        return f'background-color: {color}; text-align: center; padding: 5px; font-size: 14px'
    return ''

def add_symbols(val):
    if isinstance(val, bool):
        return '✔️' if val else '❌'
    return val

def color_percentage(val):
    try:
        percentage = float(val.strip('%'))
        if percentage >= 85:
            color = '#2E8B57'  # Verde más oscuro, suavizado
            font_weight = 'bold'  # Negrilla para porcentajes altos
            font_color = 'black'  # Letra negra en fondos verdes
        elif percentage >= 75:
            color = '#3CB371'  # Verde intermedio, suavizado
            font_weight = 'bold'
            font_color = 'black'
        elif percentage >= 65:
            color = '#98FB98'  # Verde claro, suavizado
            font_weight = 'bold'
            font_color = 'black'
        elif percentage <= 10:
            color = '#A52A2A'  # Rojo más oscuro, suavizado
            font_weight = 'bold'  # Negrilla para porcentajes bajos
            font_color = 'white'  # Letra blanca en fondos rojos
        elif percentage <= 20:
            color = '#CD5C5C'  # Rojo intermedio, suavizado
            font_weight = 'bold'
            font_color = 'white'
        elif percentage <= 30:
            color = '#F08080'  # Rojo claro, suavizado
            font_weight = 'bold'
            font_color = 'white'
        else:
            color = 'white'  # Blanco para porcentajes intermedios
            font_weight = 'normal'
            font_color = 'black'  # Letra negra en fondos blancos
        
        return f'background-color: {color}; color: {font_color}; text-align: center; vertical-align: middle; padding: 5px; font-size: 14px; font-weight: {font_weight}'
    except:
        return ''

columns_to_style = [
    "1H (+2.5)", "1H (+3.5)", "1H (+4.5)", "1H (+5.5)", "1H (+6.5)",
    "Home (+1.5)", "Home (+2.5)", "Home (+3.5)", "Home (+4.5)", "Home (+5.5)",
    "Away (+1.5)", "Away (+2.5)", "Away (+3.5)", "Away (+4.5)", "Away (+5.5)",
    "1H Home Win", "1H Away Win", "1H Draw"
]

# Añadir las nuevas columnas a la lista de columnas a estilizar
columns_to_style.extend([
    
    "Handicap Home (0)", "Handicap Away (0)",
    
    "Handicap Home (+1)", "Handicap Home (-1)", 
    "Handicap Away (+1)", "Handicap Away (-1)",
    
    "Handicap Home (+1.5)", "Handicap Home (-1.5)",
    "Handicap Away (+1.5)", "Handicap Away (-1.5)",
    
    "Handicap Home (+2)", "Handicap Home (-2)", 
    "Handicap Away (+2)", "Handicap Away (-2)",
    
    "Handicap Home (+2.5)", "Handicap Home (-2.5)",
    "Handicap Away (+2.5)", "Handicap Away (-2.5)"
    
])

# Calcular porcentajes
percentages = {}
for col in columns_to_style:
    percentage = filtered_data[col].mean() * 100
    percentages[col] = f"{percentage:.2f}%"

# Crear encabezados con porcentajes
column_headers = {col: f"{col}\n({percentages[col]})" for col in columns_to_style}

# Añadir los nuevos encabezados con porcentajes
column_headers.update({col: f"{col}\n({percentages[col]})" for col in [
    
    "Handicap Home (0)", "Handicap Away (0)",
    
    "Handicap Home (+1)", "Handicap Home (-1)", 
    "Handicap Away (+1)", "Handicap Away (-1)",
    
    "Handicap Home (+1.5)", "Handicap Away (+1.5)",
    "Handicap Home (-1.5)", "Handicap Away (-1.5)",
    
    "Handicap Home (+2)", "Handicap Home (-2)",
    "Handicap Away (+2)", "Handicap Away (-2)",
    
    "Handicap Home (+2.5)", "Handicap Home (-2.5)",
    "Handicap Away (+2.5)", "Handicap Away (-2.5)"]})

# Mostrar la tabla de porcentajes transpuesta con colores
percentages_df = pd.DataFrame.from_dict(percentages, orient='index', columns=['Porcentaje'])
percentages_df = percentages_df.T  # Transponer la tabla para que sea horizontal
styled_percentages = percentages_df.style.applymap(color_percentage)

st.write("")
st.dataframe(styled_percentages)

# Aplicar estilos y añadir símbolos a la tabla
styled_table = filtered_data.style.applymap(color_cells, subset=columns_to_style)
styled_table = styled_table.format({col: add_symbols for col in columns_to_style})

#------------------------------------------------------------------------------
# TABLA DE POSIBLES APUESTAS - MEJORADA
#------------------------------------------------------------------------------

# Función para filtrar valores significativos que están en verde o rojo (alto o bajo)
def filter_significant_values(df, threshold_high=65, threshold_low=30):
    filtered_df = df.applymap(lambda x: x if (float(x.strip('%')) >= threshold_high or float(x.strip('%')) <= threshold_low) else None)
    filtered_df = filtered_df.dropna(how='all')  # Eliminar filas con valores no significativos (NaN)
    return filtered_df

# Crear el DataFrame de porcentajes
percentages = {}
for col in columns_to_style:
    percentage = filtered_data[col].mean() * 100
    percentages[col] = f"{percentage:.2f}%"  # Mostrar solo dos decimales

percentages_df = pd.DataFrame.from_dict(percentages, orient='index', columns=['Porcentaje'])

# Filtrar solo los valores que están en verde o rojo
filtered_percentages_df = filter_significant_values(percentages_df)

# Ordenar los resultados de mayor a menor en términos de porcentaje
sorted_filtered_percentages_df = filtered_percentages_df.copy()
# Asegurarse de que todos los valores se convierten a float para ordenarlos correctamente
sorted_filtered_percentages_df['Porcentaje'] = sorted_filtered_percentages_df['Porcentaje'].apply(lambda x: float(x.strip('%')))

# Ahora podemos ordenar
sorted_filtered_percentages_df = sorted_filtered_percentages_df.sort_values(by='Porcentaje', ascending=False)

# Crear un DataFrame vertical con los títulos y porcentajes
final_df = sorted_filtered_percentages_df.reset_index()
final_df.columns = ['Bet', '%']

# Convertir de nuevo los valores a string con el símbolo de porcentaje
final_df['%'] = final_df['%'].apply(lambda x: f"{x:.2f}%")

# Función para aplicar estilos de color
def color_percentage(val):
    try:
        percentage = float(val.strip('%'))
        if percentage >= 85:
            color = '#2E8B57'
            font_weight = 'bold'
            font_color = 'black'
        elif percentage >= 75:
            color = '#3CB371'
            font_weight = 'bold'
            font_color = 'black'
        elif percentage >= 65:
            color = '#98FB98'
            font_weight = 'bold'
            font_color = 'black'
        elif percentage <= 10:
            color = '#A52A2A'
            font_weight = 'bold'
            font_color = 'white'
        elif percentage <= 20:
            color = '#CD5C5C'
            font_weight = 'bold'
            font_color = 'white'
        elif percentage <= 30:
            color = '#F08080'
            font_weight = 'bold'
            font_color = 'white'
        else:
            color = 'white'
            font_weight = 'normal'
            font_color = 'black'
        
        return f'background-color: {color}; color: {font_color}; text-align: center; vertical-align: middle; padding: 5px; font-size: 14px; font-weight: {font_weight}'
    except:
        return ''

# Categorizar las columnas
home_bets = [col for col in columns_to_style if col.startswith('Home ') or 'Handicap Home' in col]
away_bets = [col for col in columns_to_style if col.startswith('Away ') or 'Handicap Away' in col]
other_bets = [col for col in columns_to_style if col not in home_bets and col not in away_bets]

# Crear DataFrames para cada categoría
def create_final_df(filtered_percentages_df, bet_category):
    df = filtered_percentages_df.copy()
    df = df[df.index.isin(bet_category)]
    df = df.sort_values(by='Porcentaje', ascending=False)
    # Eliminar filas con valores NaN para asegurarnos de que no haya celdas vacías
    df = df.dropna(how='any')
    final = df.reset_index()
    final.columns = ['Bet', '%']
    final['%'] = final['%'].apply(lambda x: f"{x:.2f}%")
    return final

final_home_df = create_final_df(sorted_filtered_percentages_df, home_bets)
final_away_df = create_final_df(sorted_filtered_percentages_df, away_bets)
final_other_df = create_final_df(sorted_filtered_percentages_df, other_bets)

# Aplicar estilos de color a cada DataFrame
styled_home_df = final_home_df.style.applymap(color_percentage, subset=['%'])
styled_away_df = final_away_df.style.applymap(color_percentage, subset=['%'])
styled_other_df = final_other_df.style.applymap(color_percentage, subset=['%'])

# Mostrar las tablas en Streamlit lado a lado
st.subheader('1H Corners Bets')

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Home Bets**")
    st.dataframe(styled_home_df, height=300)

with col2:
    st.markdown("**Away Bets**")
    st.dataframe(styled_away_df, height=300)

with col3:
    st.markdown("**Other Bets**")
    st.dataframe(styled_other_df, height=300)

#------------------------------------------------------------------------------
# TABLA PRINCIPAL
#------------------------------------------------------------------------------

# Mostrar tabla en Streamlit
st.subheader(f'1H CORNERS Analysis for {selected_team} - {analysis_type}')
st.dataframe(styled_table)

#------------------------------------------------------------------------------
# TABLA DE RANKING DE PROMEDIOS y DESVIACIONES DE CORNERS (LOCAL Y VISITANTE)
#------------------------------------------------------------------------------

# Filtrar los datos para solo los equipos actuales
home_data = corners_data[corners_data['Home'].isin(current_teams)]
away_data = corners_data[corners_data['Away'].isin(current_teams)]

if last_20_games:
    home_data = home_data.sort_values(by='Date', ascending=False).groupby('Home').head(20)
    away_data = away_data.sort_values(by='Date', ascending=False).groupby('Away').head(20)

# Calcular los agregados para home_data
home_agg_for = home_data.groupby('Home').agg(
    {'1H Home Corners': ['mean', 'std', 'sum', 'count']}
).reset_index()

home_agg_against = home_data.groupby('Home').agg(
    {'1H Away Corners': ['mean', 'std', 'sum', 'count']}
).reset_index()

# Calcular los agregados para away_data
away_agg_for = away_data.groupby('Away').agg(
    {'1H Away Corners': ['mean', 'std', 'sum', 'count']}
).reset_index()

away_agg_against = away_data.groupby('Away').agg(
    {'1H Home Corners': ['mean', 'std', 'sum', 'count']}
).reset_index()

# Renombrar las columnas para claridad
home_agg_for.columns = ['Team', 'μ', 'σ', '1H Corners For', '# Matches']
home_agg_against.columns = ['Team', 'μ', 'σ', '1H Corners Against', '# Matches']
away_agg_for.columns = ['Team', 'μ', 'σ', '1H Corners For', '# Matches']
away_agg_against.columns = ['Team', 'μ', 'σ', '1H Corners Against', '# Matches']

# Añadir columna de rango [μ - σ ; μ + σ]
home_agg_for['Range [μ - σ ; μ + σ]'] = home_agg_for.apply(lambda x: f"[{x['μ'] - x['σ']:.2f} ; {x['μ'] + x['σ']:.2f}]", axis=1)
home_agg_against['Range [μ - σ ; μ + σ]'] = home_agg_against.apply(lambda x: f"[{x['μ'] - x['σ']:.2f} ; {x['μ'] + x['σ']:.2f}]", axis=1)
away_agg_for['Range [μ - σ ; μ + σ]'] = away_agg_for.apply(lambda x: f"[{x['μ'] - x['σ']:.2f} ; {x['μ'] + x['σ']:.2f}]", axis=1)
away_agg_against['Range [μ - σ ; μ + σ]'] = away_agg_against.apply(lambda x: f"[{x['μ'] - x['σ']:.2f} ; {x['μ'] + x['σ']:.2f}]", axis=1)

# Reordenar columnas para que 'Range [μ - σ ; μ + σ]' esté después de 'σ'
home_agg_for = home_agg_for[['Team', 'μ', 'σ', 'Range [μ - σ ; μ + σ]', '1H Corners For', '# Matches']]
home_agg_against = home_agg_against[['Team', 'μ', 'σ', 'Range [μ - σ ; μ + σ]', '1H Corners Against', '# Matches']]
away_agg_for = away_agg_for[['Team', 'μ', 'σ', 'Range [μ - σ ; μ + σ]', '1H Corners For', '# Matches']]
away_agg_against = away_agg_against[['Team', 'μ', 'σ', 'Range [μ - σ ; μ + σ]', '1H Corners Against', '# Matches']]

# Ordenar por 'Average 1H Corners For' y 'Average 1H Corners Against' para el ranking
home_agg_for = home_agg_for.sort_values(by='μ', ascending=False).reset_index(drop=True)
home_agg_against = home_agg_against.sort_values(by='μ', ascending=False).reset_index(drop=True)
away_agg_for = away_agg_for.sort_values(by='μ', ascending=False).reset_index(drop=True)
away_agg_against = away_agg_against.sort_values(by='μ', ascending=False).reset_index(drop=True)

# Renombrar las columnas para que el texto esté en dos líneas
home_agg_for.columns = ['Team', 'μ', 'σ', 'Range [μ - σ ;\nμ + σ]', '1H\nCorners For', '#\nMatches']
home_agg_against.columns = ['Team', 'μ', 'σ', 'Range [μ - σ ;\nμ + σ]', '1H\nCorners Against', '#\nMatches']
away_agg_for.columns = ['Team', 'μ', 'σ', 'Range [μ - σ ;\nμ + σ]', '1H\nCorners For', '#\nMatches']
away_agg_against.columns = ['Team', 'μ', 'σ', 'Range [μ - σ ;\nμ + σ]', '1H\nCorners Against', '#\nMatches']

# Agregar índice comenzando desde 1
home_agg_for.index += 1
home_agg_against.index += 1
away_agg_for.index += 1
away_agg_against.index += 1

# Función para resaltar la fila del equipo seleccionado
def highlight_team(row, team):
    return ['background-color: lightgreen' if row.Team == team else '' for _ in row]

# Mostrar las tablas en Streamlit

# Crear las columnas para mostrar las 4 tablas
col1, col2 = st.columns(2)

with col1:
    st.subheader('HOME Corners For (1H)')
    st.dataframe(home_agg_for.style.apply(lambda x: highlight_team(x, selected_team), axis=1).format({
        'μ': "{:.2f}", 'σ': "{:.2f}", '1H Corners For': "{:.0f}", '# Matches': "{:.0f}"
    }))
    
    st.subheader('HOME Corners Against (1H)')
    st.dataframe(home_agg_against.style.apply(lambda x: highlight_team(x, selected_team), axis=1).format({
        'μ': "{:.2f}", 'σ': "{:.2f}", '1H Corners Against': "{:.0f}"
    }))

with col2:
    st.subheader('AWAY Corners For (1H)')
    st.dataframe(away_agg_for.style.apply(lambda x: highlight_team(x, selected_team), axis=1).format({
        'μ': "{:.2f}", 'σ': "{:.2f}", '1H Corners For': "{:.0f}", '# Matches': "{:.0f}"
    }))
    
    st.subheader('AWAY Corners Against (1H)')
    st.dataframe(away_agg_against.style.apply(lambda x: highlight_team(x, selected_team), axis=1).format({
        'μ': "{:.2f}", 'σ': "{:.2f}", '1H Corners Against': "{:.0f}" 
    }))

#------------------------------------------------------------------------------
# GRÁFICOS 
#------------------------------------------------------------------------------

#1
# Datos
total_corners = filtered_data['1H Total Corners']
home_corners = filtered_data['1H Home Corners']
away_corners = filtered_data['1H Away Corners']

# Calcular la media y desviación estándar
def get_stats(data):
    return data.mean(), data.std()

mean_total, std_dev_total = get_stats(total_corners)
mean_home, std_dev_home = get_stats(home_corners)
mean_away, std_dev_away = get_stats(away_corners)

# Crear figura y ejes
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

# Función para añadir porcentajes y números de partidos
def add_bar_info(ax, data, color, label_prefix):
    # Obtener los datos de las barras
    patches = ax.patches
    total_counts = sum(p.get_height() for p in patches)

    # Calcular los porcentajes y encontrar las 3 barras con mayores porcentajes
    percentages = [(p.get_height() / total_counts) * 100 for p in patches]
    top_indices = sorted(range(len(percentages)), key=lambda i: percentages[i], reverse=True)[:3]

    # Añadir los porcentajes y los números de partidos a cada barra y resaltar las 3 más altas
    for i, p in enumerate(patches):
        height = p.get_height()
        percentage = percentages[i]
        num_matches_in_bin = int(p.get_height())
        
        # Resaltar las barras con los mayores porcentajes
        if i in top_indices:
            p.set_color('palegreen')  # Cambia el color de la barra
            p.set_edgecolor('black')  # Añadir borde negro
            ax.text(p.get_x() + p.get_width() / 2, height, f'{percentage:.1f}%', 
                    ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')
            ax.text(p.get_x() + p.get_width() / 2, height - (height * 0.30), f'{num_matches_in_bin}', 
                    ha='center', va='bottom', fontsize=10, color='red', fontweight='bold')
        else:
            ax.text(p.get_x() + p.get_width() / 2, height, f'{percentage:.1f}%', 
                    ha='center', va='bottom', fontsize=10, color='black')
            ax.text(p.get_x() + p.get_width() / 2, height - (height * 0.40), f'{num_matches_in_bin}', 
                    ha='center', va='bottom', fontsize=10, color='red')

# Gráfico para Total Corners
sns.histplot(total_corners, bins=12, kde=True, ax=axes[0])
axes[0].axvline(mean_total, color='b', linestyle='--', linewidth=2, 
                label=f'Media: {mean_total:.2f}\nDesviación Estándar: {std_dev_total:.2f}')
axes[0].set_title(f'1H Total Corners Distribution ({selected_team})')
axes[0].set_xlabel('Total de Corners')
axes[0].set_ylabel('Frecuencia')
axes[0].legend(loc='upper left')
axes[0].text(0.95, 0.95, f'Partidos Analizados: {len(total_corners)}', 
              horizontalalignment='right', verticalalignment='top', 
              transform=axes[0].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
add_bar_info(axes[0], total_corners, 'b', 'Total Corners')

# Gráfico para Home Corners
sns.histplot(home_corners, bins=12, kde=True, ax=axes[1])
axes[1].axvline(mean_home, color='r', linestyle='--', linewidth=2, 
                label=f'Media: {mean_home:.2f}\nDesviación Estándar: {std_dev_home:.2f}')
axes[1].set_title(f'Home Corners FOR or AGAINST ({selected_team})')
axes[1].set_xlabel('Home Corners')
axes[1].set_ylabel('Frecuencia')
axes[1].legend(loc='upper left')
axes[1].text(0.95, 0.95, f'Partidos Analizados: {len(home_corners)}', 
              horizontalalignment='right', verticalalignment='top', 
              transform=axes[1].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
add_bar_info(axes[1], home_corners, 'r', 'Home Corners')

# Gráfico para Away Corners
sns.histplot(away_corners, bins=12, kde=True, ax=axes[2])
axes[2].axvline(mean_away, color='g', linestyle='--', linewidth=2, 
                label=f'Media: {mean_away:.2f}\nDesviación Estándar: {std_dev_away:.2f}')
axes[2].set_title(f'Away Corners FOR or AGAINST ({selected_team})')
axes[2].set_xlabel('Away Corners')
axes[2].set_ylabel('Frecuencia')
axes[2].legend(loc='upper left')
axes[2].text(0.95, 0.95, f'Partidos Analizados: {len(away_corners)}', 
              horizontalalignment='right', verticalalignment='top', 
              transform=axes[2].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
add_bar_info(axes[2], away_corners, 'g', 'Away Corners')

# Ajustar el espacio entre gráficos
plt.tight_layout()

# Mostrar gráficos
st.pyplot(fig)

# #2
# st.subheader('Promedio de Corners por Equipo')
# home_corners_avg = filtered_data.groupby('Home')['Home Corners'].mean().sort_values(ascending=False)
# away_corners_avg = filtered_data.groupby('Away')['Away Corners'].mean().sort_values(ascending=False)

# plt.figure(figsize=(8, 4))
# home_corners_avg.plot(kind='bar', color='blue', alpha=0.7, label='Home Corners')
# away_corners_avg.plot(kind='bar', color='red', alpha=0.7, label='Away Corners')
# plt.xlabel('Equipo')
# plt.ylabel('Promedio de Corners')
# plt.legend()
# plt.xticks(rotation=90)
# st.pyplot(plt)

# #3
# st.subheader('Porcentaje de Partidos que Superan Líneas de Corners Totales')
# total_matches = len(filtered_data)
# over_columns = ["Over 7.5 Corners", "Over 8.5 Corners", "Over 9.5 Corners", "Over 10.5 Corners", "Over 11.5 Corners"]
# over_percentages = {col: filtered_data[col].sum() / total_matches * 100 for col in over_columns}

# # Mostrar porcentajes como tarjetas
# for col, percentage in over_percentages.items():
#     st.metric(label=col, value=f"{percentage:.2f}%", delta="")

# #4
# st.subheader('Porcentaje de Partidos que Superan Líneas de Corners en la Primera Mitad')
# first_half_columns = ["1H Over 3.5 Corners", "1H Over 4.5 Corners", "1H Over 5.5 Corners", "1H Over 6.5 Corners"]
# first_half_percentages = {col: filtered_data[col].sum() / total_matches * 100 for col in first_half_columns}

# # Mostrar porcentajes como tarjetas
# for col, percentage in first_half_percentages.items():
#     st.metric(label=col, value=f"{percentage:.2f}%", delta="")
    
# #5
# st.subheader('Comparación de Porcentajes de Corners Totales y en la Primera Mitad')
# data_comparison = {
#     'Umbral': list(over_percentages.keys()) + list(first_half_percentages.keys()),
#     'Porcentaje': list(over_percentages.values()) + list(first_half_percentages.values()),
#     'Tipo': ['Total'] * len(over_percentages) + ['1H'] * len(first_half_percentages)
# }
# df_comparison = pd.DataFrame(data_comparison)

# plt.figure(figsize=(10, 5))
# ax = sns.barplot(data=df_comparison, x='Umbral', y='Porcentaje', hue='Tipo', palette='coolwarm')
# plt.xlabel('Umbral de Corners')
# plt.ylabel('Porcentaje (%)')
# plt.xticks(rotation=90)

# Añadir etiquetas de porcentaje en cada barra
# for p in ax.patches:
#     ax.annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
#                 ha='center', va='center', xytext=(0, 5), textcoords='offset points')
# st.pyplot(plt)

# #6
# st.subheader('Tendencia Mensual de Porcentajes de Corners Totales que Superan Líneas')
# corners_data['start time'] = pd.to_datetime(corners_data['start time'], errors='coerce')
# corners_data['Month'] = corners_data['start time'].dt.to_period('M')

# monthly_over_totals = filtered_data.groupby('Month')[over_columns].mean() * 100

# plt.figure(figsize=(8, 4))
# for col in over_columns:
#     plt.plot(monthly_over_totals.index.astype(str), monthly_over_totals[col], label=col)

# plt.xlabel('Fecha')
# plt.ylabel('Porcentaje (%)')
# plt.legend()
# plt.xticks(rotation=45)
# st.pyplot(plt)

# #7
# st.subheader('Tendencia Mensual de Porcentajes de Corners en la Primera Mitad que Superan Líneas')
# monthly_over_1h = filtered_data.groupby('Month')[first_half_columns].mean() * 100

# plt.figure(figsize=(8, 4))
# for col in first_half_columns:
#     plt.plot(monthly_over_1h.index.astype(str), monthly_over_1h[col], label=col)

# plt.xlabel('Fecha')
# plt.ylabel('Porcentaje (%)')
# plt.legend()
# plt.xticks(rotation=45)
# st.pyplot(plt)




#------------------------------------------------------------------------------
# RANKINGS
#------------------------------------------------------------------------------

import streamlit as st
import pandas as pd

# LINEAS

# Lista de los 20 equipos actuales
current_teams = [
    
    "AC Milan", "Atalanta", "Bologna", "Cagliari", "Como",
    "Empoli", "Fiorentina", "Genoa", "Inter Milan", "Juventus",
    "Lecce", "Lazio", "Monza", "Napoli", "Parma",
    "Roma", "Torino", "Udinese", "Venezia", "Verona"
]

# Definir las líneas de apuestas
lines = [
    "1H (+2.5)", "1H (+3.5)", "1H (+4.5)", "1H (+5.5)", "1H (+6.5)",  # Primera fila
    "Home (+1.5)", "Home (+2.5)", "Home (+3.5)", "Home (+4.5)", "Home (+5.5)",  # Segunda fila
    "Away (+1.5)", "Away (+2.5)", "Away (+3.5)", "Away (+4.5)", "Away (+5.5)",  # Tercera fila
]

# Nuevas líneas de apuestas
new_lines = [
    # Primera fila con 5 columnas
    ["1H Home Win", "1H Away Win", "1H Draw", "Handicap Home (0)", "Handicap Away (0)"],

    # Segunda fila con 4 columnas
    ["Handicap Home (+1)", "Handicap Home (-1)", "Handicap Away (+1)", "Handicap Away (-1)"],

    # Tercera fila con 4 columnas
    ["Handicap Home (+1.5)", "Handicap Home (-1.5)", "Handicap Away (+1.5)", "Handicap Away (-1.5)"],

    # Cuarta fila con 4 columnas
    ["Handicap Home (+2)", "Handicap Home (-2)", "Handicap Away (+2)", "Handicap Away (-2)"],

    # Quinta fila con 4 columnas
    ["Handicap Home (+2.5)", "Handicap Home (-2.5)", "Handicap Away (+2.5)", "Handicap Away (-2.5)"]
]


# FUNCION PARA CALCULAR %

import pandas as pd
import streamlit as st

# FUNCION PARA CALCULAR %
def calculate_percentage(df, line, role):
    # Filtrar los datos en función del rol (Home o Away)
    if role == "Home":
        percentage_df = df[df['Home'].isin(current_teams)].groupby('Home')[line].mean().reset_index()
    elif role == "Away":
        percentage_df = df[df['Away'].isin(current_teams)].groupby('Away')[line].mean().reset_index()

    # Asegurar que todos los equipos estén presentes en el ranking, aunque no tengan datos
    missing_teams = [team for team in current_teams if team not in percentage_df.iloc[:, 0].values]
    missing_data = pd.DataFrame({percentage_df.columns[0]: missing_teams, line: [0] * len(missing_teams)})
    
    # Combinar los datos originales con los datos de los equipos faltantes
    percentage_df = pd.concat([percentage_df, missing_data], ignore_index=True)

    # Convertir el resultado en porcentaje
    percentage_df[line] = (percentage_df[line] * 100)

    # Renombrar la columna de porcentaje
    percentage_df.rename(columns={line: f'{line} (%)'}, inplace=True)

    # Ordenar el DataFrame por porcentaje en orden descendente
    percentage_df.sort_values(by=f'{line} (%)', ascending=False, inplace=True)

    # Reiniciar el índice después de ordenar y eliminar el índice anterior
    percentage_df.reset_index(drop=True, inplace=True)

    # Añadir la columna 'Team' que contendrá el nombre del equipo
    percentage_df['Team'] = percentage_df[percentage_df.columns[0]]

    # Seleccionar las columnas que se mostrarán: el equipo y el porcentaje
    percentage_df = percentage_df[['Team', f'{line} (%)']]

    return percentage_df

# Función para colorear celdas en verde si el valor es superior a un umbral (ej: 60%)
def highlight_high_values(s):
    colors = []
    for v in s:
        if v >= 85:
            colors.append('background-color: #2E8B57; font-weight: bold; color: black')  # Verde más oscuro
        elif v >= 75:
            colors.append('background-color: #3CB371; font-weight: bold; color: black')  # Verde intermedio
        elif v >= 65:
            colors.append('background-color: #98FB98; font-weight: normal; color: black')  # Verde claro
        elif v < 10:
            colors.append('background-color: #A52A2A; font-weight: bold; color: white')  # Rojo más oscuro
        elif v < 20:
            colors.append('background-color: #CD5C5C; font-weight: bold; color: white')  # Rojo intermedio
        else:
            colors.append('')  # Sin color si no cumple con los umbrales
    return colors    


# Crear un DataFrame de rankings vacío
ranking_dfs = {}

# Línea con emojis temáticos
st.markdown("**─────────────────────────────────────────────────────────────────────────────────────────────────────────**")  # Línea oscura con símbolos
st.markdown("### ⚽⚽⚽ Corners Rankings ⚽⚽⚽")

#------------------------------------------------------------------------------ 
# FILTROS
#------------------------------------------------------------------------------ 

# Botón para filtrar los últimos 20 partidos para el equipo local
filter_last_20_home = st.checkbox("Use only the last 20 matches for Home Team")

# Botón para filtrar los últimos 20 partidos para el equipo visitante
filter_last_20_away = st.checkbox("Use only the last 20 matches for Away Team")

# Seleccionar equipo local y visitante
home_team = st.selectbox("Select Home Team:", current_teams)
away_team = st.selectbox("Select Away Team:", current_teams)

# Seleccionar el rol para el equipo local
home_role = st.radio("Select role for Home Team:", ("Home", "Away"), index=0)

# Seleccionar el rol para el equipo visitante
away_role = st.radio("Select role for Away Team:", ("Home", "Away"), index=1)

# Filtrar los datos según el checkbox de Home y Away
if filter_last_20_home:
    # Filtrar los últimos 20 partidos para el equipo local
    home_matches = corners_data[corners_data['Home'] == home_team]
    home_matches_last_20 = home_matches.sort_values(by='Date', ascending=False).head(20)
    corners_data = pd.concat([home_matches_last_20, corners_data[corners_data['Home'] != home_team]])
    
if filter_last_20_away:
    # Filtrar los últimos 20 partidos para el equipo visitante
    away_matches = corners_data[corners_data['Away'] == away_team]
    away_matches_last_20 = away_matches.sort_values(by='Date', ascending=False).head(20)
    corners_data = pd.concat([away_matches_last_20, corners_data[corners_data['Away'] != away_team]])

# Asegurarse de que solo los equipos seleccionados estén presentes en el DataFrame final
selected_teams = [home_team, away_team]
corners_data = corners_data[(corners_data['Home'].isin(selected_teams)) | (corners_data['Away'].isin(selected_teams))]

# Calcular y almacenar los rankings para cada línea original
for line in lines:
    # Calcular para el equipo local
    percentage_df_home = calculate_percentage(corners_data, line, home_role)
    percentage_df_home = percentage_df_home[percentage_df_home['Team'] == home_team]
    
    # Calcular para el equipo visitante
    percentage_df_away = calculate_percentage(corners_data, line, away_role)
    percentage_df_away = percentage_df_away[percentage_df_away['Team'] == away_team]
    
    # Combinar los DataFrames de local y visitante
    combined_df = pd.concat([percentage_df_home, percentage_df_away], ignore_index=True)
    
    ranking_dfs[f'{line}'] = combined_df

# Calcular y almacenar los rankings para cada nueva línea
for row in new_lines:
    for line in row:
        # Calcular para el equipo local
        percentage_df_home = calculate_percentage(corners_data, line, home_role)
        percentage_df_home = percentage_df_home[percentage_df_home['Team'] == home_team]
        
        # Calcular para el equipo visitante
        percentage_df_away = calculate_percentage(corners_data, line, away_role)
        percentage_df_away = percentage_df_away[percentage_df_away['Team'] == away_team]
        
        # Combinar los DataFrames de local y visitante
        combined_df = pd.concat([percentage_df_home, percentage_df_away], ignore_index=True)
        
        ranking_dfs[f'{line}'] = combined_df

#------------------------------------------------------------------------------
# Tablas de rankings
#------------------------------------------------------------------------------

# Mostrar los rankings en filas y columnas:
# Primera fila con 4 columnas (1H)
columns1 = st.columns(5)
for i, line in enumerate(lines[:5]):  # Los primeros 4 (1H)
    with columns1[i]:
        st.subheader(f'{line}:')
        # Aplicar formato de color y limitar los decimales a 2, agregando el símbolo %
        styled_df = ranking_dfs[line].style.format({
            f'{line} (%)': "{:.2f}%"
        }).apply(highlight_high_values, subset=[f'{line} (%)'])
        st.dataframe(styled_df)

# Segunda fila con 5 columnas (Home)
columns2 = st.columns(5)
for i, line in enumerate(lines[5:10]):  # Siguientes 5 (Home)
    with columns2[i]:
        st.subheader(f'{line}:')
        # Aplicar formato de color y limitar los decimales a 2, agregando el símbolo %
        styled_df = ranking_dfs[line].style.format({
            f'{line} (%)': "{:.2f}%"
        }).apply(highlight_high_values, subset=[f'{line} (%)'])
        st.dataframe(styled_df)

# Tercera fila con 5 columnas (Away)
columns3 = st.columns(5)
for i, line in enumerate(lines[10:]):  # Últimos 5 (Away)
    with columns3[i]:
        st.subheader(f'{line}:')
        # Aplicar formato de color y limitar los decimales a 2, agregando el símbolo %
        styled_df = ranking_dfs[line].style.format({
            f'{line} (%)': "{:.2f}%"
        }).apply(highlight_high_values, subset=[f'{line} (%)'])
        st.dataframe(styled_df)

# Primera fila con 5 columnas (nuevas líneas)
columns4 = st.columns(5)
for i, line in enumerate(new_lines[0]):  # Primera fila de nuevas líneas
    with columns4[i]:
        st.subheader(f'{line}:')
        styled_df = ranking_dfs[line].style.format({
            f'{line} (%)': "{:.2f}%"
        }).apply(highlight_high_values, subset=[f'{line} (%)'])
        st.dataframe(styled_df)

# Segunda fila con 4 columnas (nuevas líneas)
columns5 = st.columns(4)
for i, line in enumerate(new_lines[1]):  # Segunda fila de nuevas líneas
    with columns5[i]:
        st.subheader(f'{line}:')
        styled_df = ranking_dfs[line].style.format({
            f'{line} (%)': "{:.2f}%"
        }).apply(highlight_high_values, subset=[f'{line} (%)'])
        st.dataframe(styled_df)

# Tercera fila con 4 columnas (nuevas líneas)
columns6 = st.columns(4)
for i, line in enumerate(new_lines[2]):  # Tercera fila de nuevas líneas
    with columns6[i]:
        st.subheader(f'{line}:')
        styled_df = ranking_dfs[line].style.format({
            f'{line} (%)': "{:.2f}%"
        }).apply(highlight_high_values, subset=[f'{line} (%)'])
        st.dataframe(styled_df)

# Cuarta fila con 4 columnas (nuevas líneas)
columns7 = st.columns(4)
for i, line in enumerate(new_lines[3]):  # Cuarta fila de nuevas líneas
    with columns7[i]:
        st.subheader(f'{line}:')
        styled_df = ranking_dfs[line].style.format({
            f'{line} (%)': "{:.2f}%"
        }).apply(highlight_high_values, subset=[f'{line} (%)'])
        st.dataframe(styled_df)

# Quinta fila con 4 columnas (nuevas líneas)
columns8 = st.columns(4)
for i, line in enumerate(new_lines[4]):  # Quinta fila de nuevas líneas
    with columns8[i]:
        st.subheader(f'{line}:')
        styled_df = ranking_dfs[line].style.format({
            f'{line} (%)': "{:.2f}%"
        }).apply(highlight_high_values, subset=[f'{line} (%)'])
        st.dataframe(styled_df)

#------------------------------------------------------------------------------
# SUBTABLA ( HOME VS AWAY)
#------------------------------------------------------------------------------

# Crear selectores en Streamlit para el equipo local y visitante
selected_local_team = st.selectbox("Selecciona el equipo local", current_teams)
selected_away_team = st.selectbox("Selecciona el equipo visitante", current_teams)

subtabla = corners_data[
    (corners_data['Home'] == selected_local_team) & 
    (corners_data['Away'] == selected_away_team)
].reset_index(drop=True)  # Reinicia el índice y descarta el anterior

# Mostrar la subtabla filtrada en Streamlit
st.subheader(f'1H CORNERS Analysis for {selected_local_team} vs {selected_away_team}')
st.dataframe(subtabla)



















