from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

# Configuración del driver de Chrome (SERIE A)
options = webdriver.ChromeOptions()
prefs = {"download.default_directory": "C:\\Users\\user\\Desktop\\Bet Data\\Italy"}
options.add_experimental_option("prefs", prefs)
driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)

# Función para extraer la tabla de una página
def extract_table_from_page(url):
    driver.get(url)
    time.sleep(5)  # Esperar a que la página cargue completamente
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    # Buscar la tabla en la página
    table = soup.find('table', class_='table table-striped table-bordered table-condensed table-hover background_table')
    
    if table:
        df = pd.read_html(str(table))[0]
        return df
    else:
        print(f"Tabla no encontrada en {url}")
        return None

# Generar las URLs de todas las páginas (1 a 97)
base_url = "https://www.totalcorner.com/league/view/12/page:{}?copy=yes"
urls = [base_url.format(i) for i in range(1, 98)]  # Páginas de la 1 a la 97

# Lista para almacenar los DataFrames
dfs = []

# Iterar sobre cada URL y extraer la tabla
for url in urls:
    df = extract_table_from_page(url)
    if df is not None:
        dfs.append(df)
    
    # Pausa aleatoria entre 5 y 15 segundos
    pause_duration = random.randint(5, 15)
    print(f"Pausando por {pause_duration} segundos antes de la siguiente solicitud...")
    time.sleep(pause_duration)

# Concatenar todos los DataFrames en uno solo
if dfs:
    final_df = pd.concat(dfs, ignore_index=True)

    # Eliminar las columnas no deseadas
    columns_to_drop = ["Unnamed: 1", "Handicap", "Asian Corn.", "Corner O/U", "Goals", "Goals O/U"]
    final_df = final_df.drop(columns=columns_to_drop)

    # Guardar el DataFrame en un archivo Excel
    output_path = "C:\\Users\\user\\Desktop\\Bet Data\\Costa Rica\\Italy Corners (Serie A).xlsx"
    final_df.to_excel(output_path, index=False)

    print("Datos extraídos y guardados en Excel con éxito.")
else:
    print("No se encontraron tablas en ninguna de las páginas.")

# Cerrar el driver
driver.quit()
