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
data = data.iloc[0:].reset_index(drop=True)

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

corners_data = data[['start time', 'Home', 'Away',
                     'Home Corners', 'Away Corners', 'Total Corners']]
 
corners_data['Date'] = pd.to_datetime(corners_data['start time'])

#------------------------------------------------------------------------------
# TABLA 1 - FULL TIME ANALYSIS
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# LINEAS (OVER 3.5 4.5 5.5 6.5) (FULL TIME)
#------------------------------------------------------------------------------

corners_data["FT (+6.5)"] = corners_data["Home Corners"] + corners_data["Away Corners"] > 6.5
corners_data["FT (+7.5)"] = corners_data["Home Corners"] + corners_data["Away Corners"] > 7.5
corners_data["FT (+8.5)"] = corners_data["Home Corners"] + corners_data["Away Corners"] > 8.5
corners_data["FT (+9.5)"] = corners_data["Home Corners"] + corners_data["Away Corners"] > 9.5
corners_data["FT (+10.5)"] = corners_data["Home Corners"] + corners_data["Away Corners"] > 10.5
corners_data["FT (+11.5)"] = corners_data["Home Corners"] + corners_data["Away Corners"] > 11.5
corners_data["FT (+12.5)"] = corners_data["Home Corners"] + corners_data["Away Corners"] > 12.5

corners_data["FT Home (+2.5)"] = corners_data["Home Corners"] > 2.5
corners_data["FT Home (+3.5)"] = corners_data["Home Corners"] > 3.5
corners_data["FT Home (+4.5)"] = corners_data["Home Corners"] > 4.5
corners_data["FT Home (+5.5)"] = corners_data["Home Corners"] > 5.5
corners_data["FT Home (+6.5)"] = corners_data["Home Corners"] > 6.5

corners_data["FT Away (+2.5)"] = corners_data["Away Corners"] > 2.5
corners_data["FT Away (+3.5)"] = corners_data["Away Corners"] > 3.5
corners_data["FT Away (+4.5)"] = corners_data["Away Corners"] > 4.5
corners_data["FT Away (+5.5)"] = corners_data["Away Corners"] > 5.5
corners_data["FT Away (+6.5)"] = corners_data["Away Corners"] > 6.5

# Crear las variables de resultado en la primera mitad
corners_data["Home Win"] = corners_data["Home Corners"] > corners_data["Away Corners"]
corners_data["Away Win"] = corners_data["Away Corners"] > corners_data["Home Corners"]
corners_data["Draw"] = corners_data["Home Corners"] == corners_data["Away Corners"]

#------------------------------------------------------------------------------ 
# LÍNEAS DE HÁNDICAP (FULL TIME) - Desde -4.5 hasta +4.5 
#------------------------------------------------------------------------------ 

# Hándicap desde -4.5 hasta +4.5 para corners en la primera mitad (FT)

# handicap_values = [-4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5,
#                    0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]

# for h in handicap_values:
#     # Para el equipo local (Home)
#     corners_data[f"Handicap Home ({h})"] = (corners_data["Home Corners"] + h) > corners_data["Away Corners"]
    
#     # Para el equipo visitante (Away)
#     corners_data[f"Handicap Away ({h})"] = (corners_data["Away Corners"] + h) > corners_data["Home Corners"]

#------------------------------------------------------------------------------
# CONFIGURAR PÁGINA STREAMLIT (Título... etc)
#------------------------------------------------------------------------------

st.set_page_config(page_title='Italy Corners FT', layout='wide', initial_sidebar_state='expanded')

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

# Filtrar los datos según el equipo seleccionado - DATOS ORIGINALES CON ÚNICAMENTE LOS 20 EQUIPOS
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
# FILTRO - ÚLTIMOS 10, 20 PARTIDOS O TODOS
#------------------------------------------------------------------------------

# Crear radio button para seleccionar entre los últimos 10, 20 partidos o todos
last_matches = st.radio(
    'Select Number of Matches to Display:',
    options=['All Matches', 'Last 10 Matches', 'Last 20 Matches'],
    index=0  # Por defecto selecciona 'All Matches'
)

# Filtrar los datos según la selección
if last_matches == 'Last 10 Matches':
    filtered_data = filtered_data.head(10)
elif last_matches == 'Last 20 Matches':
    filtered_data = filtered_data.head(20)
else:  # 'All Matches'
    filtered_data = filtered_data  # No se realiza ningún filtrado adicional

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
        if percentage >= 90:
            color = '#2E8B57'  # Verde más oscuro, suavizado
            font_weight = 'bold'  # Negrilla para porcentajes altos
            font_color = 'black'  # Letra negra en fondos verdes
        elif percentage >= 80:
            color = '#3CB371'  # Verde intermedio, suavizado
            font_weight = 'bold'
            font_color = 'black'
        elif percentage >= 70:
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
    # Líneas FT para el total de corners
    "FT (+6.5)", "FT (+7.5)", "FT (+8.5)", "FT (+9.5)", "FT (+10.5)", "FT (+11.5)", "FT (+12.5)",

    # Líneas FT para el equipo local (Home)
    "FT Home (+2.5)", "FT Home (+3.5)", "FT Home (+4.5)", "FT Home (+5.5)", "FT Home (+6.5)",

    # Líneas FT para el equipo visitante (Away)
    "FT Away (+2.5)", "FT Away (+3.5)", "FT Away (+4.5)", "FT Away (+5.5)", "FT Away (+6.5)",

    # Resultados (ganador/empate) en tiempo completo
    "Home Win", "Away Win", "Draw"
]

# Añadir las columnas correspondientes a cada valor de hándicap
# for h in handicap_values:
#     columns_to_style.extend([
#         f"Handicap Home ({h})",  # Añadir el hándicap para el equipo local
#         f"Handicap Away ({h})"   # Añadir el hándicap para el equipo visitante
#     ])

# Calcular porcentajes
percentages = {}
for col in columns_to_style:
    percentage = filtered_data[col].mean() * 100
    percentages[col] = f"{percentage:.2f}%"

# Crear encabezados con porcentajes
column_headers = {col: f"{col}\n({percentages[col]})" for col in columns_to_style}

# Actualizar los encabezados con el formato "{col} ({percentages[col]})"
# column_headers.update({f"Handicap Home ({h})": f"Handicap Home ({h})\n({percentages[f'Handicap Home ({h})']})" for h in handicap_values})
# column_headers.update({f"Handicap Away ({h})": f"Handicap Away ({h})\n({percentages[f'Handicap Away ({h})']})" for h in handicap_values})

# Mostrar la tabla de porcentajes transpuesta con colores
percentages_df = pd.DataFrame.from_dict(percentages, orient='index', columns=['Porcentaje'])
percentages_df = percentages_df.T  # Transponer la tabla para que sea horizontal
styled_percentages = percentages_df.style.applymap(color_percentage)

st.write("")
st.dataframe(styled_percentages)

# Aplicar estilos y añadir símbolos a la tabla
styled_table = filtered_data.style.applymap(color_cells, subset=columns_to_style)
styled_table = styled_table.format({col: add_symbols for col in columns_to_style})

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# POSSIBLE BETS
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

st.markdown("**─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────**")  # Línea oscura con símbolos

# Función para filtrar valores significativos que están en verde o rojo (alto o bajo)
def filter_significant_values(df, threshold_high=70, threshold_low=30):
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
        if percentage >= 90:
            color = '#2E8B57'
            font_weight = 'bold'
            font_color = 'black'
        elif percentage >= 80:
            color = '#3CB371'
            font_weight = 'bold'
            font_color = 'black'
        elif percentage >= 70:
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

# Clean the DataFrames by removing empty rows (rows where all values are NaN)
final_home_df_cleaned = final_home_df.dropna(how='any')  # Drop rows with any NaN values
final_away_df_cleaned = final_away_df.dropna(how='any')
final_other_df_cleaned = final_other_df.dropna(how='any')

# Transpose the DataFrames for horizontal display
final_home_df_transposed = final_home_df.set_index('Bet').T
final_away_df_transposed = final_away_df.set_index('Bet').T
final_other_df_transposed = final_other_df.set_index('Bet').T

# Apply styles to the transposed DataFrames
styled_home_df_transposed = final_home_df_transposed.style.applymap(color_percentage)
styled_away_df_transposed = final_away_df_transposed.style.applymap(color_percentage)
styled_other_df_transposed = final_other_df_transposed.style.applymap(color_percentage)

# Mostrar las tablas en Streamlit lado a lado

st.subheader('💰💰💰 Possible Bets 💰💰💰')

st.markdown("**Home Bets**")
st.dataframe(styled_home_df_transposed, height=50)

st.markdown("**Away Bets**")
st.dataframe(styled_away_df_transposed, height=50)

st.markdown("**Other Bets**")
st.dataframe(styled_other_df_transposed, height=50)

st.markdown("**─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────**")  # Línea oscura con símbolos

#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# TABLA PRINCIPAL
#--------------------------------------------------------------------------------------------------------------------------------------------------------------

# Mostrar tabla en Streamlit
st.subheader( f'🕰️🕰️🕰️ Historical CORNERS Analysis for {selected_team} - {analysis_type} 🕰️🕰️🕰️')
st.dataframe(styled_table)

#------------------------------------------------------------------------------
# TABLA DE RANKING DE PROMEDIOS y DESVIACIONES DE CORNERS (LOCAL Y VISITANTE)
#------------------------------------------------------------------------------

# Filtrar los datos para solo los equipos actuales
home_data = corners_data[corners_data['Home'].isin(current_teams)]
away_data = corners_data[corners_data['Away'].isin(current_teams)]

# Filtrar según la selección de partidos
if last_matches == 'Last 10 Matches':
    home_data = home_data.sort_values(by='Date', ascending=False).groupby('Home').head(10)
    away_data = away_data.sort_values(by='Date', ascending=False).groupby('Away').head(10)
elif last_matches == 'Last 20 Matches':
    home_data = home_data.sort_values(by='Date', ascending=False).groupby('Home').head(20)
    away_data = away_data.sort_values(by='Date', ascending=False).groupby('Away').head(20)

# Calcular los agregados para home_data
home_agg_for = home_data.groupby('Home').agg(
    {'Home Corners': ['mean', 'std', 'sum', 'count']}
).reset_index()

home_agg_against = home_data.groupby('Home').agg(
    {'Away Corners': ['mean', 'std', 'sum', 'count']}
).reset_index()

# Calcular los agregados para away_data
away_agg_for = away_data.groupby('Away').agg(
    {'Away Corners': ['mean', 'std', 'sum', 'count']}
).reset_index()

away_agg_against = away_data.groupby('Away').agg(
    {'Home Corners': ['mean', 'std', 'sum', 'count']}
).reset_index()

# Renombrar las columnas para claridad
home_agg_for.columns = ['Team', 'μ', 'σ', 'FT Corners For', '# Matches']
home_agg_against.columns = ['Team', 'μ', 'σ', 'FT Corners Against', '# Matches']
away_agg_for.columns = ['Team', 'μ', 'σ', 'FT Corners For', '# Matches']
away_agg_against.columns = ['Team', 'μ', 'σ', 'FT Corners Against', '# Matches']

# Añadir columna de rango [μ - σ ; μ + σ]
home_agg_for['Range [μ - σ ; μ + σ]'] = home_agg_for.apply(lambda x: f"[{x['μ'] - x['σ']:.2f} ; {x['μ'] + x['σ']:.2f}]", axis=1)
home_agg_against['Range [μ - σ ; μ + σ]'] = home_agg_against.apply(lambda x: f"[{x['μ'] - x['σ']:.2f} ; {x['μ'] + x['σ']:.2f}]", axis=1)
away_agg_for['Range [μ - σ ; μ + σ]'] = away_agg_for.apply(lambda x: f"[{x['μ'] - x['σ']:.2f} ; {x['μ'] + x['σ']:.2f}]", axis=1)
away_agg_against['Range [μ - σ ; μ + σ]'] = away_agg_against.apply(lambda x: f"[{x['μ'] - x['σ']:.2f} ; {x['μ'] + x['σ']:.2f}]", axis=1)

# Reordenar columnas para que 'Range [μ - σ ; μ + σ]' esté después de 'σ'
home_agg_for = home_agg_for[['Team', 'μ', 'σ', 'Range [μ - σ ; μ + σ]', 'FT Corners For', '# Matches']]
home_agg_against = home_agg_against[['Team', 'μ', 'σ', 'Range [μ - σ ; μ + σ]', 'FT Corners Against', '# Matches']]
away_agg_for = away_agg_for[['Team', 'μ', 'σ', 'Range [μ - σ ; μ + σ]', 'FT Corners For', '# Matches']]
away_agg_against = away_agg_against[['Team', 'μ', 'σ', 'Range [μ - σ ; μ + σ]', 'FT Corners Against', '# Matches']]

# Ordenar por 'Average FT Corners For' y 'Average FT Corners Against' para el ranking
home_agg_for = home_agg_for.sort_values(by='μ', ascending=False).reset_index(drop=True)
home_agg_against = home_agg_against.sort_values(by='μ', ascending=False).reset_index(drop=True)
away_agg_for = away_agg_for.sort_values(by='μ', ascending=False).reset_index(drop=True)
away_agg_against = away_agg_against.sort_values(by='μ', ascending=False).reset_index(drop=True)

# Renombrar las columnas para que el texto esté en dos líneas
home_agg_for.columns = ['Team', 'μ', 'σ', 'Range [μ - σ ;\nμ + σ]', 'FT\nCorners For', '#\nMatches']
home_agg_against.columns = ['Team', 'μ', 'σ', 'Range [μ - σ ;\nμ + σ]', 'FT\nCorners Against', '#\nMatches']
away_agg_for.columns = ['Team', 'μ', 'σ', 'Range [μ - σ ;\nμ + σ]', 'FT\nCorners For', '#\nMatches']
away_agg_against.columns = ['Team', 'μ', 'σ', 'Range [μ - σ ;\nμ + σ]', 'FT\nCorners Against', '#\nMatches']

# Agregar índice comenzando desde 1
home_agg_for.index += 1
home_agg_against.index += 1
away_agg_for.index += 1
away_agg_against.index += 1

# Función para resaltar la fila del equipo seleccionado
def highlight_team(row, team):
    return ['background-color: lightgreen' if row.Team == team else '' for _ in row]

# Mostrar las tablas en Streamlit

st.markdown("**─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────**")  # Línea oscura con símbolos
st.markdown("### 📶📶📶 Rankings 📶📶📶")


# Crear las columnas para mostrar las 4 tablas
col1, col2 = st.columns(2)

with col1:
    st.subheader('HOME Corners For (FT)')
    st.dataframe(home_agg_for.style.apply(lambda x: highlight_team(x, selected_team), axis=1).format({
        'μ': "{:.2f}", 'σ': "{:.2f}", 'FT Corners For': "{:.0f}", '# Matches': "{:.0f}"
    }))
    
    st.subheader('HOME Corners Against (FT)')
    st.dataframe(home_agg_against.style.apply(lambda x: highlight_team(x, selected_team), axis=1).format({
        'μ': "{:.2f}", 'σ': "{:.2f}", 'FT Corners Against': "{:.0f}"
    }))

with col2:
    st.subheader('AWAY Corners For (FT)')
    st.dataframe(away_agg_for.style.apply(lambda x: highlight_team(x, selected_team), axis=1).format({
        'μ': "{:.2f}", 'σ': "{:.2f}", 'FT Corners For': "{:.0f}", '# Matches': "{:.0f}"
    }))
    
    st.subheader('AWAY Corners Against (FT)')
    st.dataframe(away_agg_against.style.apply(lambda x: highlight_team(x, selected_team), axis=1).format({
        'μ': "{:.2f}", 'σ': "{:.2f}", 'FT Corners Against': "{:.0f}" 
    }))

#---------------------------------------------------------------------------------------------------------------------------------------------
# GRÁFICOS 
#---------------------------------------------------------------------------------------------------------------------------------------------

# Línea con emojis temáticos
st.markdown("**─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────**")  # Línea oscura con símbolos
st.markdown("### 📊📊📊 Distributions 📊📊📊")

#---------------------------------------------------------------------------- 1 (DISTRIBUTIONS)
# Datos
total_corners = filtered_data['Total Corners']
home_corners = filtered_data['Home Corners']
away_corners = filtered_data['Away Corners']

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
axes[0].set_title(f'FT Total Corners Distribution ({selected_team})')
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

st.markdown("**─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────**")  # Línea oscura con símbolos

#---------------------------------------------------------------------------- 2 FOR (BAR CHART)

st.subheader('Average Corners FOR (Home & Away)')

# Calculate averages
home_corners_avg = home_data.groupby('Home')['Home Corners'].mean().sort_values(ascending=False)
away_corners_avg = away_data.groupby('Away')['Away Corners'].mean().sort_values(ascending=False)

# Set up the subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 5))  # Two graphs side by side

# Function to create transparent bars with black borders
def plot_transparent_bars(ax, averages, team_color, highlight_team):
    # Set gray background
    ax.set_facecolor('#f2f2f2')  # Smooth gray color
    ax.grid(color='white', linestyle='-', linewidth=0.5, axis='x')  # Subtle white grid for better readability
    for i, (team, value) in enumerate(averages.items()):
        if team == highlight_team:
            # Selected team with color
            ax.barh(i, value, color=team_color, edgecolor='black', alpha=1.0, height=0.8, label='Selected Team' if i == 0 else "")
        else:
            # Transparent bars with black borders
            ax.barh(i, value, color='none', edgecolor='black', alpha=1.0, height=0.8)
        # Add text above each bar
        ax.text(value + 0.05, i, f'{value:.2f}', color='black', va='center', fontweight='bold' if team == highlight_team else 'normal')

# Plot for home averages
plot_transparent_bars(axes[0], home_corners_avg, 'forestgreen', selected_team)
axes[0].set_title('Average Corners FOR - Home', fontsize=14)
axes[0].set_xlabel('', fontsize=12)
axes[0].set_ylabel('', fontsize=12)
axes[0].set_yticks(range(len(home_corners_avg)))
axes[0].set_yticklabels(home_corners_avg.index, fontsize=10)
axes[0].invert_yaxis()  # Invert to show highest averages on top

# Plot for away averages
plot_transparent_bars(axes[1], away_corners_avg, 'forestgreen', selected_team)
axes[1].set_title('Average Corners FOR - Away', fontsize=14)
axes[1].set_xlabel('', fontsize=12)
axes[1].set_ylabel('')  # Remove duplicate y-axis label
axes[1].set_yticks(range(len(away_corners_avg)))
axes[1].set_yticklabels(away_corners_avg.index, fontsize=10)
axes[1].invert_yaxis()  # Invert to show highest averages on top

# Adjust layout for better fit
plt.tight_layout()

# Render the plot in Streamlit
st.pyplot(fig)

# ---------------------------------------------------------------------------- FOR (HEATMAP)

# Define thresholds
corner_thresholds = [2.5, 3.5, 4.5, 5.5]

# Calculate probabilities for Home teams
home_probabilities = {
    team: [
        home_data[(home_data['Home'] == team) & (home_data['Home Corners'] > threshold)].shape[0] / 
        home_data[home_data['Home'] == team].shape[0] * 100
        for threshold in corner_thresholds
    ]
    for team in home_data['Home'].unique()
}

# Calculate probabilities for Away teams
away_probabilities = {
    team: [
        away_data[(away_data['Away'] == team) & (away_data['Away Corners'] > threshold)].shape[0] / 
        away_data[away_data['Away'] == team].shape[0] * 100
        for threshold in corner_thresholds
    ]
    for team in away_data['Away'].unique()
}

# Create DataFrames for both heatmaps
home_heatmap_data = pd.DataFrame(home_probabilities, index=[f"FT Home (+{t})" for t in corner_thresholds]).T
away_heatmap_data = pd.DataFrame(away_probabilities, index=[f"FT Away (+{t})" for t in corner_thresholds]).T

# Order both DataFrames based on the 2.5 threshold
home_heatmap_data = home_heatmap_data.sort_values(by="FT Home (+2.5)", ascending=False)
away_heatmap_data = away_heatmap_data.sort_values(by="FT Away (+2.5)", ascending=False)

# Add average row to the bottom of each heatmap
home_avg_row = home_heatmap_data.mean(axis=0)
home_heatmap_data.loc["Average"] = home_avg_row

away_avg_row = away_heatmap_data.mean(axis=0)
away_heatmap_data.loc["Average"] = away_avg_row

# Plotting both heatmaps side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))  # Adjusted to make both heatmaps visible

# Home heatmap
sns.heatmap(home_heatmap_data, annot=True, cmap="coolwarm", fmt=".1f", ax=ax1, linewidths=0.5, linecolor='gray')

# Make the "Average" row bold and thicker borders
for t in range(len(home_heatmap_data.columns)):
    if home_heatmap_data.index[-1] == "Average":
        ax1.add_patch(plt.Rectangle((t, len(home_heatmap_data) - 1), 1, 1, fill=False, edgecolor='black', lw=2))  # Thicker border for average row

ax1.set_title("Probability of Exceeding Corner Thresholds (FOR Home Teams)")
ax1.set_xlabel("Threshold")
ax1.set_ylabel("Team")

# Away heatmap
sns.heatmap(away_heatmap_data, annot=True, cmap="coolwarm", fmt=".1f", ax=ax2, linewidths=0.5, linecolor='gray')

# Make the "Average" row bold and thicker borders
for t in range(len(away_heatmap_data.columns)):
    if away_heatmap_data.index[-1] == "Average":
        ax2.add_patch(plt.Rectangle((t, len(away_heatmap_data) - 1), 1, 1, fill=False, edgecolor='black', lw=2))  # Thicker border for average row

ax2.set_title("Probability of Exceeding Corner Thresholds (FOR Away Teams)")
ax2.set_xlabel("Threshold")
ax2.set_ylabel("Team")

# Display the plot in Streamlit with adjusted size
st.pyplot(fig, use_container_width=True)


# #---------------------------------------------------------------------------- FOR (VIOLINPLOTS HOME)

# import plotly.express as px
# import pandas as pd
# import streamlit as st

# # Sample thresholds
# thresholds = [1.5, 2.5, 3.5, 4.5]

# # Create buttons with Streamlit's layout to display them horizontally in columns
# col1, col2, col3, col4 = st.columns(4)

# # Initialize a variable to track the selected threshold
# selected_threshold = None

# # Create buttons in columns (Side by Side)
# with col1:
#     if st.button("Home FOR 1.5"):
#         selected_threshold = 1.5

# with col2:
#     if st.button("Home FOR 2.5"):
#         selected_threshold = 2.5

# with col3:
#     if st.button("Home FOR 3.5"):
#         selected_threshold = 3.5

# with col4:
#     if st.button("Home FOR 4.5"):
#         selected_threshold = 4.5

# # Default threshold if none selected
# if selected_threshold is None:
#     selected_threshold = 2.5

# # Calculate the percentage of games each team exceeds the selected threshold
# home_data["exceeds_threshold"] = home_data["Home Corners"] > selected_threshold
# team_percentages = home_data.groupby("Home")["exceeds_threshold"].mean() * 100

# # Reorder the 'Home' column in the DataFrame based on the sorted team percentages (from highest to lowest)
# sorted_teams = team_percentages.sort_values(ascending=False).index
# home_data["Home"] = pd.Categorical(home_data["Home"], categories=sorted_teams, ordered=True)

# # # Filter data to focus on teams with Total Corners greater than 5
# filtered_data = home_data[home_data["Total Corners"] >= 1]

# # Create the violin plot using Plotly
# fig = px.violin(
#     filtered_data,
#     x="Home",
#     y="Home Corners",
#     color="Home",
#     title=f"(HOME) Distribution of FOR Corners by Team (Ordered by Percentage Exceeding {selected_threshold})",
#     labels={"Home": "Team", "AGAINST Corners": "FOR Corners"},
#     template="plotly_white",
#     box=True,  # Show box plot inside the violin plot for comparison
#     points="all",  # Show all individual points for better granularity
#     width=800,
#     height=500
# )

# # Add horizontal threshold lines for the selected threshold
# fig.add_hline(
#     y=selected_threshold, line_dash="dash", line_color="black", 
#     annotation_text=f"+{selected_threshold}", annotation_position="top left",
#     line_width=2
# )

# # Highlight teams with the highest percentage exceeding the selected threshold
# for team, percentage in team_percentages.items():
#     if percentage >= 70:  # Highlight teams with > 70% of games exceeding the threshold
#         # Place the annotation above the violin plots (very high)
#         fig.add_annotation(
#             x=team,
#             y=filtered_data["Total Corners"].max() + 2,  # Adjust y to be above the plot
#             text=f"{percentage:.1f}%",
#             showarrow=True,
#             arrowhead=2,
#             arrowcolor="green",  # Use green to highlight high percentage teams
#             font=dict(color="black", size=12),
#             ax=0,
#             ay=-40  # Position of the annotation (moving the arrow up)
#         )

# # Customize the layout for better visualization
# fig.update_layout(
#     xaxis=dict(
#         title="Team", 
#         tickangle=90,
#         categoryorder="array",  # Order categories based on the provided list
#         categoryarray=sorted_teams  # Set the order of the teams based on their percentage
#     ),
#     yaxis_title="Total Corners",
#     showlegend=False,  # Hide legend for cleaner output
#     plot_bgcolor="white",  # Make background clean
# )

# # Display the plot in Streamlit
# st.plotly_chart(fig, use_container_width=True)


# #---------------------------------------------------------------------------- FOR (VIOLINPLOTS AWAY)

# import plotly.express as px
# import pandas as pd
# import streamlit as st

# # Sample thresholds
# thresholds = [1.5, 2.5, 3.5, 4.5]

# # Create buttons with Streamlit's layout to display them horizontally in columns
# col1, col2, col3, col4 = st.columns(4)

# # Initialize a variable to track the selected threshold
# selected_threshold = None

# # Create buttons in columns (Side by Side)
# with col1:
#     if st.button("Away FOR 1.5"):
#         selected_threshold = 1.5

# with col2:
#     if st.button("Away FOR 2.5"):
#         selected_threshold = 2.5

# with col3:
#     if st.button("Away FOR 3.5"):
#         selected_threshold = 3.5

# with col4:
#     if st.button("Away FOR 4.5"):
#         selected_threshold = 4.5

# # Default threshold if none selected
# if selected_threshold is None:
#     selected_threshold = 2.5

# # Calculate the percentage of games each team exceeds the selected threshold
# away_data["exceeds_threshold"] = away_data["Away Corners"] > selected_threshold
# team_percentages = away_data.groupby("Away")["exceeds_threshold"].mean() * 100

# # Reorder the 'Home' column in the DataFrame based on the sorted team percentages (from highest to lowest)
# sorted_teams = team_percentages.sort_values(ascending=False).index
# away_data["Away"] = pd.Categorical(away_data["Away"], categories=sorted_teams, ordered=True)

# # # Filter data to focus on teams with Total Corners greater than 5
# filtered_data = away_data[away_data["Total Corners"] >= 1]

# # Create the violin plot using Plotly
# fig2 = px.violin(
#     filtered_data,
#     x="Away",
#     y="Away Corners",
#     color="Away",
#     title=f"(AWAY) Distribution of AGAINST Corners by Team (Ordered by Percentage Exceeding {selected_threshold})",
#     labels={"Away": "Team", "AGAINST Corners": "FOR Corners"},
#     template="plotly_white",
#     box=True,  # Show box plot inside the violin plot for comparison
#     points="all",  # Show all individual points for better granularity
#     width=800,
#     height=500
# )

# # Add horizontal threshold lines for the selected threshold
# fig2.add_hline(
#     y=selected_threshold, line_dash="dash", line_color="black", 
#     annotation_text=f"+{selected_threshold}", annotation_position="top left",
#     line_width=2
# )

# # Highlight teams with the highest percentage exceeding the selected threshold
# for team, percentage in team_percentages.items():
#     if percentage >= 70:  # Highlight teams with > 70% of games exceeding the threshold
#         # Place the annotation above the violin plots (very high)
#         fig2.add_annotation(
#             x=team,
#             y=filtered_data["Total Corners"].max() + 2,  # Adjust y to be above the plot
#             text=f"{percentage:.1f}%",
#             showarrow=True,
#             arrowhead=2,
#             arrowcolor="green",  # Use green to highlight high percentage teams
#             font=dict(color="black", size=12),
#             ax=0,
#             ay=-40  # Position of the annotation (moving the arrow up)
#         )

# # Customize the layout for better visualization
# fig2.update_layout(
#     xaxis=dict(
#         title="Team", 
#         tickangle=90,
#         categoryorder="array",  # Order categories based on the provided list
#         categoryarray=sorted_teams  # Set the order of the teams based on their percentage
#     ),
#     yaxis_title="Total Corners",
#     showlegend=False,  # Hide legend for cleaner output
#     plot_bgcolor="white",  # Make background clean
# )

# # Display the plot in Streamlit
# st.plotly_chart(fig2, use_container_width=True)




st.markdown("**─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────**")  # Línea oscura con símbolos

#---------------------------------------------------------------------------- 3 AGAINST (BAR CHART)

st.subheader('Average Corners AGAINST (Home & Away)')

# Calculate averages
home_corners_avg = home_data.groupby('Home')['Away Corners'].mean().sort_values(ascending=False)
away_corners_avg = away_data.groupby('Away')['Home Corners'].mean().sort_values(ascending=False)

# Set up the subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 5))  # Two graphs side by side

# Function to create transparent bars with black borders
def plot_transparent_bars(ax, averages, team_color, highlight_team):
    # Set gray background
    ax.set_facecolor('#f2f2f2')  # Smooth gray color
    ax.grid(color='white', linestyle='-', linewidth=0.5, axis='x')  # Subtle white grid for better readability
    for i, (team, value) in enumerate(averages.items()):
        if team == highlight_team:
            # Selected team with color
            ax.barh(i, value, color=team_color, edgecolor='black', alpha=1.0, height=0.8, label='Selected Team' if i == 0 else "")
        else:
            # Transparent bars with black borders
            ax.barh(i, value, color='none', edgecolor='black', alpha=1.0, height=0.8)
        # Add text above each bar
        ax.text(value + 0.05, i, f'{value:.2f}', color='black', va='center', fontweight='bold' if team == highlight_team else 'normal')

# Plot for home averages
plot_transparent_bars(axes[0], home_corners_avg, 'red', selected_team)
axes[0].set_title('Average Corners AGAINST - Home', fontsize=14)
axes[0].set_xlabel('', fontsize=12)
axes[0].set_ylabel('', fontsize=12)
axes[0].set_yticks(range(len(home_corners_avg)))
axes[0].set_yticklabels(home_corners_avg.index, fontsize=10)
axes[0].invert_yaxis()  # Invert to show highest averages on top

# Plot for away averages
plot_transparent_bars(axes[1], away_corners_avg, 'red', selected_team)
axes[1].set_title('Average Corners AGAINST - Away', fontsize=14)
axes[1].set_xlabel('', fontsize=12)
axes[1].set_ylabel('')  # Remove duplicate y-axis label
axes[1].set_yticks(range(len(away_corners_avg)))
axes[1].set_yticklabels(away_corners_avg.index, fontsize=10)
axes[1].invert_yaxis()  # Invert to show highest averages on top

# Adjust layout for better fit
plt.tight_layout()

# Render the plot in Streamlit
st.pyplot(fig)

# ---------------------------------------------------------------------------- AGAINST (HEATMAP)

# Define thresholds
corner_thresholds = [2.5, 3.5, 4.5, 5.5]

# Calculate probabilities for Home teams
home_probabilities = {
    team: [
        home_data[(home_data['Home'] == team) & (home_data['Away Corners'] > threshold)].shape[0] / 
        home_data[home_data['Home'] == team].shape[0] * 100
        for threshold in corner_thresholds
    ]
    for team in home_data['Home'].unique()
}

# Calculate probabilities for Away teams
away_probabilities = {
    team: [
        away_data[(away_data['Away'] == team) & (away_data['Home Corners'] > threshold)].shape[0] / 
        away_data[away_data['Away'] == team].shape[0] * 100
        for threshold in corner_thresholds
    ]
    for team in away_data['Away'].unique()
}

# Create DataFrames for both heatmaps
home_heatmap_data = pd.DataFrame(home_probabilities, index=[f"FT Away (+{t})" for t in corner_thresholds]).T
away_heatmap_data = pd.DataFrame(away_probabilities, index=[f"FT Home (+{t})" for t in corner_thresholds]).T

# Order both DataFrames based on the 6.5 threshold
home_heatmap_data = home_heatmap_data.sort_values(by="FT Away (+2.5)", ascending=False)
away_heatmap_data = away_heatmap_data.sort_values(by="FT Home (+2.5)", ascending=False)

# Add average row to the bottom of each heatmap
home_avg_row = home_heatmap_data.mean(axis=0)
home_heatmap_data.loc["Average"] = home_avg_row

away_avg_row = away_heatmap_data.mean(axis=0)
away_heatmap_data.loc["Average"] = away_avg_row

# Plotting both heatmaps side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))  # Adjusted to make both heatmaps visible

# Home heatmap
sns.heatmap(home_heatmap_data, annot=True, cmap="coolwarm", fmt=".1f", ax=ax1, linewidths=0.5, linecolor='gray')

# Make the "Average" row bold and thicker borders
for t in range(len(home_heatmap_data.columns)):
    if home_heatmap_data.index[-1] == "Average":
        ax1.add_patch(plt.Rectangle((t, len(home_heatmap_data) - 1), 1, 1, fill=False, edgecolor='black', lw=2))  # Thicker border for average row

ax1.set_title("Probability of Exceeding Corner Thresholds (AGAINST Home Teams)")
ax1.set_xlabel("Threshold")
ax1.set_ylabel("Team")

# Away heatmap
sns.heatmap(away_heatmap_data, annot=True, cmap="coolwarm", fmt=".1f", ax=ax2, linewidths=0.5, linecolor='gray')

# Make the "Average" row bold and thicker borders
for t in range(len(away_heatmap_data.columns)):
    if away_heatmap_data.index[-1] == "Average":
        ax2.add_patch(plt.Rectangle((t, len(away_heatmap_data) - 1), 1, 1, fill=False, edgecolor='black', lw=2))  # Thicker border for average row

ax2.set_title("Probability of Exceeding Corner Thresholds (AGAINST Away Teams)")
ax2.set_xlabel("Threshold")
ax2.set_ylabel("Team")

# Display the plot in Streamlit with adjusted size
st.pyplot(fig, use_container_width=True)


# #---------------------------------------------------------------------------- AGAINST (VIOLINPLOTS HOME)

# import plotly.express as px
# import pandas as pd
# import streamlit as st

# # Sample thresholds
# thresholds = [1.5, 2.5, 3.5, 4.5]

# # Create buttons with Streamlit's layout to display them horizontally in columns
# col1, col2, col3, col4 = st.columns(4)

# # Initialize a variable to track the selected threshold
# selected_threshold = None

# # Create buttons in columns (Side by Side)
# with col1:
#     if st.button("Home AGAINST 1.5"):
#         selected_threshold = 1.5

# with col2:
#     if st.button("Home AGAINST 2.5"):
#         selected_threshold = 2.5

# with col3:
#     if st.button("Home AGAINST 3.5"):
#         selected_threshold = 3.5

# with col4:
#     if st.button("Home AGAINST 4.5"):
#         selected_threshold = 4.5

# # Default threshold if none selected
# if selected_threshold is None:
#     selected_threshold = 2.5

# # Calculate the percentage of games each team exceeds the selected threshold
# home_data["exceeds_threshold"] = home_data["Away Corners"] > selected_threshold
# team_percentages = home_data.groupby("Home")["exceeds_threshold"].mean() * 100

# # Reorder the 'Home' column in the DataFrame based on the sorted team percentages (from highest to lowest)
# sorted_teams = team_percentages.sort_values(ascending=False).index
# home_data["Home"] = pd.Categorical(home_data["Home"], categories=sorted_teams, ordered=True)

# # # Filter data to focus on teams with Total Corners greater than 5
# filtered_data = home_data[home_data["Total Corners"] >= 1]

# # Create the violin plot using Plotly
# fig = px.violin(
#     filtered_data,
#     x="Home",
#     y="Away Corners",
#     color="Home",
#     title=f"(HOME) Distribution of AGAINST Corners by Team (Ordered by Percentage Exceeding {selected_threshold})",
#     labels={"Home": "Team", "AGAINST Corners": "FOR Corners"},
#     template="plotly_white",
#     box=True,  # Show box plot inside the violin plot for comparison
#     points="all",  # Show all individual points for better granularity
#     width=800,
#     height=500
# )

# # Add horizontal threshold lines for the selected threshold
# fig.add_hline(
#     y=selected_threshold, line_dash="dash", line_color="black", 
#     annotation_text=f"+{selected_threshold}", annotation_position="top left",
#     line_width=2
# )

# # Highlight teams with the highest percentage exceeding the selected threshold
# for team, percentage in team_percentages.items():
#     if percentage >= 70:  # Highlight teams with > 70% of games exceeding the threshold
#         # Place the annotation above the violin plots (very high)
#         fig.add_annotation(
#             x=team,
#             y=filtered_data["Total Corners"].max() + 2,  # Adjust y to be above the plot
#             text=f"{percentage:.1f}%",
#             showarrow=True,
#             arrowhead=2,
#             arrowcolor="green",  # Use green to highlight high percentage teams
#             font=dict(color="black", size=12),
#             ax=0,
#             ay=-40  # Position of the annotation (moving the arrow up)
#         )

# # Customize the layout for better visualization
# fig.update_layout(
#     xaxis=dict(
#         title="Team", 
#         tickangle=90,
#         categoryorder="array",  # Order categories based on the provided list
#         categoryarray=sorted_teams  # Set the order of the teams based on their percentage
#     ),
#     yaxis_title="Total Corners",
#     showlegend=False,  # Hide legend for cleaner output
#     plot_bgcolor="white",  # Make background clean
# )

# # Display the plot in Streamlit
# st.plotly_chart(fig, use_container_width=True)


# #---------------------------------------------------------------------------- AGAINST (VIOLINPLOTS AWAY)

# import plotly.express as px
# import pandas as pd
# import streamlit as st

# # Sample thresholds
# thresholds = [1.5, 2.5, 3.5, 4.5]

# # Create buttons with Streamlit's layout to display them horizontally in columns
# col1, col2, col3, col4 = st.columns(4)

# # Initialize a variable to track the selected threshold
# selected_threshold = None

# # Create buttons in columns (Side by Side)
# with col1:
#     if st.button("Away AGAINST 1.5"):
#         selected_threshold = 1.5

# with col2:
#     if st.button("Away AGAINST 2.5"):
#         selected_threshold = 2.5

# with col3:
#     if st.button("Away AGAINST 3.5"):
#         selected_threshold = 3.5

# with col4:
#     if st.button("Away AGAINST 4.5"):
#         selected_threshold = 4.5

# # Default threshold if none selected
# if selected_threshold is None:
#     selected_threshold = 2.5

# # Calculate the percentage of games each team exceeds the selected threshold
# away_data["exceeds_threshold"] = away_data["Home Corners"] > selected_threshold
# team_percentages = away_data.groupby("Away")["exceeds_threshold"].mean() * 100

# # Reorder the 'Home' column in the DataFrame based on the sorted team percentages (from highest to lowest)
# sorted_teams = team_percentages.sort_values(ascending=False).index
# away_data["Away"] = pd.Categorical(away_data["Away"], categories=sorted_teams, ordered=True)

# # # Filter data to focus on teams with Total Corners greater than 5
# filtered_data = away_data[away_data["Total Corners"] >= 1]

# # Create the violin plot using Plotly
# fig2 = px.violin(
#     filtered_data,
#     x="Away",
#     y="Home Corners",
#     color="Away",
#     title=f"(AWAY) Distribution of AGAINST Corners by Team (Ordered by Percentage Exceeding {selected_threshold})",
#     labels={"Away": "Team", "AGAINST Corners": "FOR Corners"},
#     template="plotly_white",
#     box=True,  # Show box plot inside the violin plot for comparison
#     points="all",  # Show all individual points for better granularity
#     width=800,
#     height=500
# )

# # Add horizontal threshold lines for the selected threshold
# fig2.add_hline(
#     y=selected_threshold, line_dash="dash", line_color="black", 
#     annotation_text=f"+{selected_threshold}", annotation_position="top left",
#     line_width=2
# )

# # Highlight teams with the highest percentage exceeding the selected threshold
# for team, percentage in team_percentages.items():
#     if percentage >= 70:  # Highlight teams with > 70% of games exceeding the threshold
#         # Place the annotation above the violin plots (very high)
#         fig2.add_annotation(
#             x=team,
#             y=filtered_data["Total Corners"].max() + 2,  # Adjust y to be above the plot
#             text=f"{percentage:.1f}%",
#             showarrow=True,
#             arrowhead=2,
#             arrowcolor="green",  # Use green to highlight high percentage teams
#             font=dict(color="black", size=12),
#             ax=0,
#             ay=-40  # Position of the annotation (moving the arrow up)
#         )

# # Customize the layout for better visualization
# fig2.update_layout(
#     xaxis=dict(
#         title="Team", 
#         tickangle=90,
#         categoryorder="array",  # Order categories based on the provided list
#         categoryarray=sorted_teams  # Set the order of the teams based on their percentage
#     ),
#     yaxis_title="Total Corners",
#     showlegend=False,  # Hide legend for cleaner output
#     plot_bgcolor="white",  # Make background clean
# )

# # Display the plot in Streamlit
# st.plotly_chart(fig2, use_container_width=True)



st.markdown("**─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────**")  # Línea oscura con símbolos

#---------------------------------------------------------------------------- 4 TOTAL (BAR CHART)

st.subheader('Average Corners TOTAL (Home & Away)')

# Calculate averages
home_corners_avg = home_data.groupby('Home')['Total Corners'].mean().sort_values(ascending=False)
away_corners_avg = away_data.groupby('Away')['Total Corners'].mean().sort_values(ascending=False)

# Set up the subplots
fig, axes = plt.subplots(1, 2, figsize=(22, 6))  # Two graphs side by side

# Function to create transparent bars with black borders
def plot_transparent_bars(ax, averages, team_color, highlight_team):
    # Set gray background
    ax.set_facecolor('#f2f2f2')  # Smooth gray color
    ax.grid(color='white', linestyle='-', linewidth=0.5, axis='y')  # Subtle white grid for better readability
    for i, (team, value) in enumerate(averages.items()):
        if team == highlight_team:
            # Selected team with color
            ax.bar(i, value, color=team_color, edgecolor='black', alpha=1.0, label='Selected Team' if i == 0 else "")
        else:
            # Transparent bars with black borders
            ax.bar(i, value, color='none', edgecolor='black', alpha=1.0)
        # Add text above each bar
        ax.text(i, value + 0.08, f'{value:.2f}', color='black', ha='center', fontweight='bold' if team == highlight_team else 'normal')

# Plot for home averages
plot_transparent_bars(axes[0], home_corners_avg, 'orange', selected_team)
axes[0].set_title('Average Corners TOTAL - Home', fontsize=18)
axes[0].set_xlabel('', fontsize=12)
axes[0].set_ylabel('Average Corners', fontsize=12)
axes[0].set_xticks(range(len(home_corners_avg)))
axes[0].set_xticklabels(home_corners_avg.index, fontsize=14, rotation=60)  # Rotate x-axis labels for better readability

# Plot for away averages
plot_transparent_bars(axes[1], away_corners_avg, 'orange', selected_team)
axes[1].set_title('Average Corners TOTAL - Away', fontsize=18)
axes[1].set_xlabel('', fontsize=12)
axes[1].set_ylabel('')  # Remove duplicate y-axis label
axes[1].set_xticks(range(len(away_corners_avg)))
axes[1].set_xticklabels(away_corners_avg.index, fontsize=14, rotation=60)  # Rotate x-axis labels for better readability

# Adjust layout for better fit
plt.tight_layout()

# Render the plot in Streamlit
st.pyplot(fig)

#---------------------------------------------------------------------------- TOTAL (HEATMAP)

# Define thresholds
corner_thresholds = [6.5, 7.5, 8.5, 9.5]

# Calculate probabilities for Home teams
home_probabilities = {
    team: [
        home_data[(home_data['Home'] == team) & (home_data['Total Corners'] > threshold)].shape[0] / 
        home_data[home_data['Home'] == team].shape[0] * 100
        for threshold in corner_thresholds
    ]
    for team in home_data['Home'].unique()
}

# Calculate probabilities for Away teams
away_probabilities = {
    team: [
        away_data[(away_data['Away'] == team) & (away_data['Total Corners'] > threshold)].shape[0] / 
        away_data[away_data['Away'] == team].shape[0] * 100
        for threshold in corner_thresholds
    ]
    for team in away_data['Away'].unique()
}

# Create DataFrames for both heatmaps
home_heatmap_data = pd.DataFrame(home_probabilities, index=[f"FT (+{t})" for t in corner_thresholds]).T
away_heatmap_data = pd.DataFrame(away_probabilities, index=[f"FT (+{t})" for t in corner_thresholds]).T

# Order both DataFrames based on the 6.5 threshold
home_heatmap_data = home_heatmap_data.sort_values(by="FT (+6.5)", ascending=False)
away_heatmap_data = away_heatmap_data.sort_values(by="FT (+6.5)", ascending=False)

# Add average row to the bottom of each heatmap
home_avg_row = home_heatmap_data.mean(axis=0)
home_heatmap_data.loc["Average"] = home_avg_row

away_avg_row = away_heatmap_data.mean(axis=0)
away_heatmap_data.loc["Average"] = away_avg_row

# Plotting both heatmaps side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))  # Adjusted to make both heatmaps visible

# Home heatmap
sns.heatmap(home_heatmap_data, annot=True, cmap="coolwarm", fmt=".1f", ax=ax1, linewidths=0.5, linecolor='gray')

# Make the "Average" row bold and thicker borders
for t in range(len(home_heatmap_data.columns)):
    if home_heatmap_data.index[-1] == "Average":
        ax1.add_patch(plt.Rectangle((t, len(home_heatmap_data) - 1), 1, 1, fill=False, edgecolor='black', lw=2))  # Thicker border for average row

ax1.set_title("Probability of Exceeding Corner Thresholds (Home Teams)")
ax1.set_xlabel("Threshold")
ax1.set_ylabel("Team")

# Away heatmap
sns.heatmap(away_heatmap_data, annot=True, cmap="coolwarm", fmt=".1f", ax=ax2, linewidths=0.5, linecolor='gray')

# Make the "Average" row bold and thicker borders
for t in range(len(away_heatmap_data.columns)):
    if away_heatmap_data.index[-1] == "Average":
        ax2.add_patch(plt.Rectangle((t, len(away_heatmap_data) - 1), 1, 1, fill=False, edgecolor='black', lw=2))  # Thicker border for average row

ax2.set_title("Probability of Exceeding Corner Thresholds (Away Teams)")
ax2.set_xlabel("Threshold")
ax2.set_ylabel("Team")

# Display the plot in Streamlit with adjusted size
st.pyplot(fig, use_container_width=True)

# #---------------------------------------------------------------------------- 5 (TENDENCES OVER TIME)

# st.subheader('Tendencia Mensual de Porcentajes de Corners Totales que Superan Líneas')

# # Ensure 'start time' is in datetime format and create 'Month' column
# corners_data['start time'] = pd.to_datetime(corners_data['start time'], errors='coerce')
# corners_data['Month'] = corners_data['start time'].dt.to_period('M')

# # Define the columns representing OVER lines
# over_columns = ["FT (+6.5)", "FT (+7.5)", "FT (+8.5)"]  # Update this as per your dataset

# # Calculate monthly averages for OVER columns
# monthly_over_totals = corners_data.groupby('Month')[over_columns].mean() * 100

# # Calculate the total number of matches per month
# monthly_match_counts = corners_data.groupby('Month').size()

# # Filter out months with fewer than 5 matches
# min_matches_threshold = 5
# valid_months = monthly_match_counts[monthly_match_counts >= min_matches_threshold].index
# monthly_over_totals = monthly_over_totals.loc[valid_months]
# monthly_match_counts = monthly_match_counts.loc[valid_months]

# # Extract the start of each year
# start_of_years = monthly_over_totals.index[monthly_over_totals.index.month == 1]

# import plotly.graph_objects as go

# # Create interactive line plot
# fig = go.Figure()
# for col in over_columns:
#     fig.add_trace(go.Scatter(
#         x=monthly_over_totals.index.astype(str),
#         y=monthly_over_totals[col],
#         mode='lines',
#         name=col
#     ))

# # Add match count annotations above all the lines
# for i, month in enumerate(monthly_match_counts.index.astype(str)):
#     # Calculate the maximum y-value for all OVER lines at this month
#     max_y_value = monthly_over_totals.iloc[i].max()
#     y_position = max_y_value + 5  # Place the annotation 5% above the maximum line

#     fig.add_trace(go.Scatter(
#         x=[month],
#         y=[y_position],
#         mode="text",
#         text=f"{monthly_match_counts.iloc[i]}",
#         textposition="top center",
#         showlegend=False,
#     ))

# # Add vertical dotted lines for the start of each year
# for start_year in start_of_years:
#     fig.add_shape(
#         type="line",
#         x0=str(start_year),
#         y0=0,
#         x1=str(start_year),
#         y1=monthly_over_totals.values.max() + 10,
#         line=dict(color="gray", width=1, dash="dot"),
#         xref="x", yref="y"
#     )

# # Add horizontal highlight for y-axis ranges (from 100 to 80 and 80 to 60)
# fig.add_shape(
#     type="rect",
#     x0=0,
#     y0=80,
#     x1=1,
#     y1=90,
#     fillcolor="green",
#     opacity=0.3,
#     layer="below",
#     xref="paper",
#     yref="y"
# )

# fig.add_shape(
#     type="rect",
#     x0=0,
#     y0=70,
#     x1=1,
#     y1=80,
#     fillcolor="#98FB98",
#     opacity=0.3,
#     layer="below",
#     xref="paper",
#     yref="y"
# )

# # Update layout with improved title and labels
# fig.update_layout(
#     title="Tendencia Mensual de Porcentajes que Superan Líneas OVER en Corners Totales",
#     xaxis_title="",
#     yaxis_title="Porcentaje (%)",
#     legend_title="Líneas OVER",
#     xaxis=dict(tickangle=45),
#     yaxis=dict(range=[30, 100], title="Porcentaje (%)"),
#     annotations=[
#         dict(
#             xref="paper",
#             yref="paper",
#             text="",
#             showarrow=False,
#             x=0.5,
#             y=-0.2,
#             font=dict(size=10, color="gray")
#         )
#     ]
# )

# st.plotly_chart(fig)


# #---------------------------------------------------------------------------- 7 VIOLINPLOTS (HOME)

# import plotly.express as px
# import pandas as pd
# import streamlit as st

# # Sample thresholds
# thresholds = [6.5, 7.5, 8.5, 9.5]

# # Create buttons with Streamlit's layout to display them horizontally in columns
# col1, col2, col3, col4 = st.columns(4)

# # Initialize a variable to track the selected threshold
# selected_threshold = None

# # Create buttons in columns (Side by Side)
# with col1:
#     if st.button("Home 6.5"):
#         selected_threshold = 6.5

# with col2:
#     if st.button("Home 7.5"):
#         selected_threshold = 7.5

# with col3:
#     if st.button("Home 8.5"):
#         selected_threshold = 8.5

# with col4:
#     if st.button("Home 9.5"):
#         selected_threshold = 9.5

# # Default threshold if none selected
# if selected_threshold is None:
#     selected_threshold = 6.5

# # Calculate the percentage of games each team exceeds the selected threshold
# home_data["exceeds_threshold"] = home_data["Total Corners"] > selected_threshold
# team_percentages = home_data.groupby("Home")["exceeds_threshold"].mean() * 100

# # Reorder the 'Home' column in the DataFrame based on the sorted team percentages (from highest to lowest)
# sorted_teams = team_percentages.sort_values(ascending=False).index
# home_data["Home"] = pd.Categorical(home_data["Home"], categories=sorted_teams, ordered=True)

# # Filter data to focus on teams with Total Corners greater than 5
# filtered_data = home_data[home_data["Total Corners"] >= 5]

# # Create the violin plot using Plotly
# fig = px.violin(
#     filtered_data,
#     x="Home",
#     y="Total Corners",
#     color="Home",
#     title=f"(HOME) Distribution of Total Corners by Team (Ordered by Percentage Exceeding {selected_threshold})",
#     labels={"Home": "Team", "Total Corners": "Total Corners"},
#     template="plotly_white",
#     box=True,  # Show box plot inside the violin plot for comparison
#     points="all",  # Show all individual points for better granularity
#     width=800,
#     height=500
# )

# # Add horizontal threshold lines for the selected threshold
# fig.add_hline(
#     y=selected_threshold, line_dash="dash", line_color="black", 
#     annotation_text=f"+{selected_threshold}", annotation_position="top left",
#     line_width=2
# )

# # Highlight teams with the highest percentage exceeding the selected threshold
# for team, percentage in team_percentages.items():
#     if percentage >= 70:  # Highlight teams with > 70% of games exceeding the threshold
#         # Place the annotation above the violin plots (very high)
#         fig.add_annotation(
#             x=team,
#             y=filtered_data["Total Corners"].max() + 2,  # Adjust y to be above the plot
#             text=f"{percentage:.1f}%",
#             showarrow=True,
#             arrowhead=2,
#             arrowcolor="green",  # Use green to highlight high percentage teams
#             font=dict(color="black", size=12),
#             ax=0,
#             ay=-40  # Position of the annotation (moving the arrow up)
#         )

# # Customize the layout for better visualization
# fig.update_layout(
#     xaxis=dict(
#         title="Team", 
#         tickangle=90,
#         categoryorder="array",  # Order categories based on the provided list
#         categoryarray=sorted_teams  # Set the order of the teams based on their percentage
#     ),
#     yaxis_title="Total Corners",
#     showlegend=False,  # Hide legend for cleaner output
#     plot_bgcolor="white",  # Make background clean
# )

# # Display the plot in Streamlit
# st.plotly_chart(fig, use_container_width=True)


# #---------------------------------------------------------------------------- 8 VIOLINPLOTS (HOME)

# import plotly.express as px
# import pandas as pd
# import streamlit as st

# # Sample thresholds
# thresholds = [6.5, 7.5, 8.5, 9.5]

# # Create buttons with Streamlit's layout to display them horizontally in columns
# col1, col2, col3, col4 = st.columns(4)

# # Initialize a variable to track the selected threshold
# selected_threshold = None

# # Create buttons in columns (Side by Side)
# with col1:
#     if st.button("Away 6.5"):
#         selected_threshold = 6.5

# with col2:
#     if st.button("Away 7.5"):
#         selected_threshold = 7.5

# with col3:
#     if st.button("Away 8.5"):
#         selected_threshold = 8.5

# with col4:
#     if st.button("Away 9.5"):
#         selected_threshold = 9.5

# # Default threshold if none selected
# if selected_threshold is None:
#     selected_threshold = 6.5

# # Calculate the percentage of games each team exceeds the selected threshold
# away_data["exceeds_threshold"] = away_data["Total Corners"] > selected_threshold
# team_percentages = away_data.groupby("Away")["exceeds_threshold"].mean() * 100

# # Reorder the 'Home' column in the DataFrame based on the sorted team percentages (from highest to lowest)
# sorted_teams = team_percentages.sort_values(ascending=False).index
# away_data["Away"] = pd.Categorical(away_data["Away"], categories=sorted_teams, ordered=True)

# # Filter data to focus on teams with Total Corners greater than 5
# filtered_data = away_data[away_data["Total Corners"] >= 5]

# # Create the violin plot using Plotly
# fig = px.violin(
#     filtered_data,
#     x="Away",
#     y="Total Corners",
#     color="Away",
#     title=f"(AWAY) Distribution of Total Corners by Team (Ordered by Percentage Exceeding {selected_threshold})",
#     labels={"Away": "Team", "Total Corners": "Total Corners"},
#     template="plotly_white",
#     box=True,  # Show box plot inside the violin plot for comparison
#     points="all",  # Show all individual points for better granularity
#     width=800,
#     height=500
# )

# # Add horizontal threshold lines for the selected threshold
# fig.add_hline(
#     y=selected_threshold, line_dash="dash", line_color="black", 
#     annotation_text=f"+{selected_threshold}", annotation_position="top left",
#     line_width=2
# )

# # Highlight teams with the highest percentage exceeding the selected threshold
# for team, percentage in team_percentages.items():
#     if percentage >= 70:  # Highlight teams with > 70% of games exceeding the threshold
#         # Place the annotation above the violin plots (very high)
#         fig.add_annotation(
#             x=team,
#             y=filtered_data["Total Corners"].max() + 2,  # Adjust y to be above the plot
#             text=f"{percentage:.1f}%",
#             showarrow=True,
#             arrowhead=2,
#             arrowcolor="green",  # Use green to highlight high percentage teams
#             font=dict(color="black", size=12),
#             ax=0,
#             ay=-40  # Position of the annotation (moving the arrow up)
#         )

# # Customize the layout for better visualization
# fig.update_layout(
#     xaxis=dict(
#         title="Team", 
#         tickangle=90,
#         categoryorder="array",  # Order categories based on the provided list
#         categoryarray=sorted_teams  # Set the order of the teams based on their percentage
#     ),
#     yaxis_title="Total Corners",
#     showlegend=False,  # Hide legend for cleaner output
#     plot_bgcolor="white",  # Make background clean
# )

# # Display the plot in Streamlit
# st.plotly_chart(fig, use_container_width=True)


# #---------------------------------------------------------------------------- 


#---------------------------------------------------------------------------------------------------------------------------------------------
# HOME VS AWAY 1
#---------------------------------------------------------------------------------------------------------------------------------------------

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

# Definir las líneas de apuestas con los nuevos títulos
lines = [
    # Líneas FT para el total de corners
    "FT (+6.5)", "FT (+7.5)", "FT (+8.5)", "FT (+9.5)", "FT (+10.5)", "FT (+11.5)", "FT (+12.5)",

    # Líneas FT para el equipo local (Home)
    "FT Home (+2.5)", "FT Home (+3.5)", "FT Home (+4.5)", "FT Home (+5.5)", "FT Home (+6.5)",

    # Líneas FT para el equipo visitante (Away)
    "FT Away (+2.5)", "FT Away (+3.5)", "FT Away (+4.5)", "FT Away (+5.5)", "FT Away (+6.5)",

]

# Nuevas líneas de apuestas
new_lines = [
    # Primera fila con 5 columnas
    # ["Home Win", "Away Win", "Draw", "Handicap Home (0)", "Handicap Away (0)"],

    # # Segunda fila con 4 columnas
    # ["Handicap Home (1)", "Handicap Home (-1)", "Handicap Away (1)", "Handicap Away (-1)"],

    # # Tercera fila con 4 columnas
    # ["Handicap Home (1.5)", "Handicap Home (-1.5)", "Handicap Away (1.5)", "Handicap Away (-1.5)"],

    # # Cuarta fila con 4 columnas
    # ["Handicap Home (2)", "Handicap Home (-2)", "Handicap Away (2)", "Handicap Away (-2)"],

    # # Quinta fila con 4 columnas
    # ["Handicap Home (2.5)", "Handicap Home (-2.5)", "Handicap Away (2.5)", "Handicap Away (-2.5)"],

    # # Sexta fila con 4 columnas - Añadimos las nuevas líneas de handicap desde -4.5 hasta +4.5
    # ["Handicap Home (3)", "Handicap Home (-3)", "Handicap Away (3)", "Handicap Away (-3)"],

    # # Séptima fila con 4 columnas
    # ["Handicap Home (3.5)", "Handicap Home (-3.5)", "Handicap Away (3.5)", "Handicap Away (-3.5)"],

    # # Octava fila con 4 columnas
    # ["Handicap Home (4)", "Handicap Home (-4)", "Handicap Away (4)", "Handicap Away (-4)"],

    # # Novena fila con 4 columnas
    # ["Handicap Home (4.5)", "Handicap Home (-4.5)", "Handicap Away (4.5)", "Handicap Away (-4.5)"]
]


#------------------------------------------------------ FUNCION PARA CALCULAR %

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

#-------------------------------------------------------- FUNCIÓN PARA COLOREAR

def highlight_high_values(s):
    colors = []
    for v in s:
        if v >= 90:
            colors.append('background-color: #2E8B57; font-weight: bold; color: black')  # Verde más oscuro
        elif v >= 80:
            colors.append('background-color: #3CB371; font-weight: bold; color: black')  # Verde intermedio
        elif v >= 70:
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
st.markdown("**─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────**")  # Línea oscura con símbolos
st.markdown("### 💲💲💲 Home vs Away 💲💲💲")

#---------------------------------------------------------------------------------------------------------------------------------------------
# HOME VS AWAY 2
#---------------------------------------------------------------------------------------------------------------------------------------------

#------------------------------------ BOTONES DE EQUIPOS-ROL-NÚMERO DE PARTIDOS

col1, col2 = st.columns(2)

with col1:
    # Seleccionar equipo local
    home_team = st.selectbox("Select Home Team:", current_teams)
    # Seleccionar el rol para el equipo local
    home_role = st.radio("Select role for Home Team:", ("Home", "Away"), index=0)
    # Seleccionar el filtro para el equipo local
    home_filter = st.radio("Select match filter for Home Team:", 
                           ("All Matches", "Last 10 Matches", "Last 20 Matches"), index=0)

with col2:
    # Seleccionar equipo visitante
    away_team = st.selectbox("Select Away Team:", current_teams)
    # Seleccionar el rol para el equipo visitante
    away_role = st.radio("Select role for Away Team:", ("Home", "Away"), index=1)
    # Seleccionar el filtro para el equipo visitante
    away_filter = st.radio("Select match filter for Away Team:", 
                           ("All Matches", "Last 10 Matches", "Last 20 Matches"), index=0)

#--------------------------------------------------------------- APLICAR FILTRO

home_matches = home_data[home_data['Home'] == home_team]
away_matches = away_data[away_data['Away'] == away_team]

# Aplicar el filtro para el equipo local
if home_filter == "Last 10 Matches":
    home_matches = home_matches.sort_values(by='Date', ascending=False).head(10)
elif home_filter == "Last 20 Matches":
    home_matches = home_matches.sort_values(by='Date', ascending=False).head(20)

# Aplicar el filtro para el equipo visitante
if away_filter == "Last 10 Matches":
    away_matches = away_matches.sort_values(by='Date', ascending=False).head(10)
elif away_filter == "Last 20 Matches":
    away_matches = away_matches.sort_values(by='Date', ascending=False).head(20)

# Calculate the number of matches
num_home_matches = len(home_matches)
num_away_matches = len(away_matches)

# Step 3: Display the results in new columns
result_col1, result_col2 = st.columns(2)

with result_col1:
    st.write(f'Number of Matches for Home Team ({home_team}): {num_home_matches}')

with result_col2:
    st.write(f'Number of Matches for Away Team ({away_team}): {num_away_matches}')

# Filtrar el DataFrame original para asegurar que solo se mantengan las filas relevantes
corners_data2 = pd.concat([home_matches, away_matches])

# Asegurarse de que solo los equipos seleccionados estén presentes en el DataFrame final
selected_teams = [home_team, away_team]

corners_data2 = corners_data2[(corners_data2['Home'].isin(selected_teams)) | (corners_data2['Away'].isin(selected_teams))]

# Calcular y almacenar los rankings para cada línea original
for line in lines:
    # Calcular para el equipo local
    percentage_df_home = calculate_percentage(corners_data2, line, home_role)
    percentage_df_home = percentage_df_home[percentage_df_home['Team'] == home_team]
    
    # Calcular para el equipo visitante
    percentage_df_away = calculate_percentage(corners_data2, line, away_role)
    percentage_df_away = percentage_df_away[percentage_df_away['Team'] == away_team]
    
    # Combinar los DataFrames de local y visitante
    combined_df = pd.concat([percentage_df_home, percentage_df_away], ignore_index=True)
    
    ranking_dfs[f'{line}'] = combined_df

# # Calcular y almacenar los rankings para cada nueva línea
# for row in new_lines:
#     for line in row:
#         # Calcular para el equipo local
#         percentage_df_home = calculate_percentage(corners_data2, line, home_role)
#         percentage_df_home = percentage_df_home[percentage_df_home['Team'] == home_team]
        
#         # Calcular para el equipo visitante
#         percentage_df_away = calculate_percentage(corners_data2, line, away_role)
#         percentage_df_away = percentage_df_away[percentage_df_away['Team'] == away_team]
        
#         # Combinar los DataFrames de local y visitante
#         combined_df = pd.concat([percentage_df_home, percentage_df_away], ignore_index=True)
        
#         ranking_dfs[f'{line}'] = combined_df


#---------------------------------------------------------------------------------------------------------------------------------------------
# HOME VS AWAY 3
#---------------------------------------------------------------------------------------------------------------------------------------------


# Mostrar los rankings en filas y columnas:
# Primera fila con 4 columnas (1H)
columns1 = st.columns(7)
for i, line in enumerate(lines[:7]):  # Los primeros 7 (FT)
    with columns1[i]:
        st.subheader(f'{line}:')
        # Aplicar formato de color y limitar los decimales a 2, agregando el símbolo %
        styled_df = ranking_dfs[line].style.format({
            f'{line} (%)': "{:.2f}%"
        }).apply(highlight_high_values, subset=[f'{line} (%)'])
        st.dataframe(styled_df)

# Segunda fila con 5 columnas (Home)
columns2 = st.columns(5)
for i, line in enumerate(lines[7:12]):  # Siguientes 5 (Home)
    with columns2[i]:
        st.subheader(f'{line}:')
        # Aplicar formato de color y limitar los decimales a 2, agregando el símbolo %
        styled_df = ranking_dfs[line].style.format({
            f'{line} (%)': "{:.2f}%"
        }).apply(highlight_high_values, subset=[f'{line} (%)'])
        st.dataframe(styled_df)

# Tercera fila con 5 columnas (Away)
columns3 = st.columns(5)
for i, line in enumerate(lines[12:]):  # Últimos 5 (Away)
    with columns3[i]:
        st.subheader(f'{line}:')
        # Aplicar formato de color y limitar los decimales a 2, agregando el símbolo %
        styled_df = ranking_dfs[line].style.format({
            f'{line} (%)': "{:.2f}%"
        }).apply(highlight_high_values, subset=[f'{line} (%)'])
        st.dataframe(styled_df)



# # Primera fila con 5 columnas (nuevas líneas)
# columns4 = st.columns(5)
# for i, line in enumerate(new_lines[0]):  # Primera fila de nuevas líneas
#     with columns4[i]:
#         st.subheader(f'{line}:')
#         styled_df = ranking_dfs[line].style.format({
#             f'{line} (%)': "{:.2f}%"
#         }).apply(highlight_high_values, subset=[f'{line} (%)'])
#         st.dataframe(styled_df)

# # Segunda fila con 4 columnas (nuevas líneas)
# columns5 = st.columns(4)
# for i, line in enumerate(new_lines[1]):  # Segunda fila de nuevas líneas
#     with columns5[i]:
#         st.subheader(f'{line}:')
#         styled_df = ranking_dfs[line].style.format({
#             f'{line} (%)': "{:.2f}%"
#         }).apply(highlight_high_values, subset=[f'{line} (%)'])
#         st.dataframe(styled_df)

# # Tercera fila con 4 columnas (nuevas líneas)
# columns6 = st.columns(4)
# for i, line in enumerate(new_lines[2]):  # Tercera fila de nuevas líneas
#     with columns6[i]:
#         st.subheader(f'{line}:')
#         styled_df = ranking_dfs[line].style.format({
#             f'{line} (%)': "{:.2f}%"
#         }).apply(highlight_high_values, subset=[f'{line} (%)'])
#         st.dataframe(styled_df)

# # Cuarta fila con 4 columnas (nuevas líneas)
# columns7 = st.columns(4)
# for i, line in enumerate(new_lines[3]):  # Cuarta fila de nuevas líneas
#     with columns7[i]:
#         st.subheader(f'{line}:')
#         styled_df = ranking_dfs[line].style.format({
#             f'{line} (%)': "{:.2f}%"
#         }).apply(highlight_high_values, subset=[f'{line} (%)'])
#         st.dataframe(styled_df)

# # Quinta fila con 4 columnas (nuevas líneas)
# columns8 = st.columns(4)
# for i, line in enumerate(new_lines[4]):  # Quinta fila de nuevas líneas
#     with columns8[i]:
#         st.subheader(f'{line}:')
#         styled_df = ranking_dfs[line].style.format({
#             f'{line} (%)': "{:.2f}%"
#         }).apply(highlight_high_values, subset=[f'{line} (%)'])
#         st.dataframe(styled_df)
        
# # Sexta fila con 4 columnas (nuevas líneas)
# columns9 = st.columns(4)
# for i, line in enumerate(new_lines[5]):  # Sexta fila de nuevas líneas
#     with columns9[i]:
#         st.subheader(f'{line}:')
#         styled_df = ranking_dfs[line].style.format({
#             f'{line} (%)': "{:.2f}%"
#         }).apply(highlight_high_values, subset=[f'{line} (%)'])
#         st.dataframe(styled_df)        

# # Septima fila con 4 columnas (nuevas líneas)
# columns10 = st.columns(4)
# for i, line in enumerate(new_lines[6]):  # Septima fila de nuevas líneas
#     with columns10[i]:
#         st.subheader(f'{line}:')
#         styled_df = ranking_dfs[line].style.format({
#             f'{line} (%)': "{:.2f}%"
#         }).apply(highlight_high_values, subset=[f'{line} (%)'])
#         st.dataframe(styled_df) 
        
# # Octava fila con 4 columnas (nuevas líneas)
# columns11 = st.columns(4)
# for i, line in enumerate(new_lines[7]):  # Octava fila de nuevas líneas
#     with columns11[i]:
#         st.subheader(f'{line}:')
#         styled_df = ranking_dfs[line].style.format({
#             f'{line} (%)': "{:.2f}%"
#         }).apply(highlight_high_values, subset=[f'{line} (%)'])
#         st.dataframe(styled_df)        
        
# # Novena fila con 4 columnas (nuevas líneas)
# columns12 = st.columns(4)
# for i, line in enumerate(new_lines[8]):  # Novena fila de nuevas líneas
#     with columns12[i]:
#         st.subheader(f'{line}:')
#         styled_df = ranking_dfs[line].style.format({
#             f'{line} (%)': "{:.2f}%"
#         }).apply(highlight_high_values, subset=[f'{line} (%)'])
#         st.dataframe(styled_df) 
        
        
#----------------------------------------------------------------------------------------------------------------------------------------------
# DIRECT MATCHES
#----------------------------------------------------------------------------------------------------------------------------------------------


# Línea con emojis temáticos
st.markdown("**─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────**")  # Línea oscura con símbolos
st.markdown("### ⚽⚽⚽  Direct Matches ⚽⚽⚽")

# Crear selectores en Streamlit para el equipo local y visitante
selected_local_team = st.selectbox("Selecciona el equipo local", current_teams)
selected_away_team = st.selectbox("Selecciona el equipo visitante", current_teams)

subtabla = corners_data[
    (corners_data['Home'] == selected_local_team) & 
    (corners_data['Away'] == selected_away_team)
].reset_index(drop=True)  # Reinicia el índice y descarta el anterior

# Mostrar la subtabla filtrada en Streamlit
st.dataframe(subtabla)



















