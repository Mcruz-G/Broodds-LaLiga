
import pandas as pd
import sqlite3
import time
from decimal import Decimal
import numpy as np 
import re

# Define a function to determine the SeasonStage
def determine_season_stage(date, round_info):
    if "Apertura" in round_info and "Regular Season" in round_info:
        return "Apertura"
    elif "Clausura" in round_info and "Regular Season" in round_info:
        return "Clausura"
    elif "Guardianes" in round_info:
        return "Guardianes"
    else:
        return "Liguilla"

# Define a function to extract the year
def extract_year(round_info, date):
    if "Quarter-finals" in round_info or "Semi-finals" in round_info or "Repechaje" or "Reclasificacion" in round_info or "Finals" in round_info:
        year = pd.to_datetime(date).year
        year = str(year)
        return year
    return ''.join(filter(str.isdigit, round_info))

# Define a function to determine the Season Type
def determine_season_type(round_info):
    if len(round_info.split('— ')) > 1:
        return round_info.split('— ')[-1]
    if len(round_info.split(' ')) > 1:
        return round_info.split(' ')[-2] + " " + round_info.split(' ')[-1]
    return round_info.split(' ')[-1]

# Función para extraer la temporada del enlace
def extraer_temporada(link):
    temporada = link.split('/')[-2]
    return temporada

def fbref_pull_and_store_data(metadata):
    # Procesar cada fila y actualizar/crear bases de datos SQL
    # complete this ditionary with the metadata of the tables you want to update
    unique_teams  = metadata.MetaEquipo.unique().tolist()
    datos_equipo = []  # Inicializar el DataFrame para los datos del equipo
    data_dict = {x : metadata[metadata.MetaEquipo == x] for x in unique_teams}

    for _, grupo in data_dict.items():
        print(f"Procesando base de datos: ")

        for _, fila in grupo.iterrows():
            tabla_nombre = fila['Tabla']
            link = fila['Link']
            metaequipo = fila['MetaEquipo']
            temporada = extraer_temporada(link) 

            print(f"Descargando datos de {tabla_nombre}, {metaequipo}.{temporada} desde {link}")
            
            time.sleep(3)  # Espera 1 segundo entre cada solicitud
            # try:
            datos_tabla = pd.read_html(link)
        
            datos_tabla = datos_tabla[1]
            datos_tabla['Temporada'] = temporada  # Agregar la temporada como columna
            datos_tabla['MetaEquipo'] = metaequipo
            datos_equipo.append(datos_tabla)

            # except:
            #     pass
    
    name_mapping = {
        "Barcelona" : "Barcelona",
        "Alavés" : "Alaves",
        "Leganés" : "Leganes",
        "Athletic Club" : "Athletic_Club",
        "Atlético Madrid" : "Atletico_Madrid",
        "Real Betis" : "Real_Betis",
        "Espanyol" : "Espanyol",
        "Celta Vigo" : "Celta_Vigo",
        "Getafe" : "Getafe",
        "Girona" : "Girona", 
        "Valladolid" : "Valladolid",
        "Las Palmas" : "Las_Palmas",
        "Mallorca" : "Mallorca", 
        "Osasuna" : "Osasuna",
        "Rayo Vallecano" : "Rayo_Vallecano", 
        "Real Madrid" : "Real_Madrid",
        "Real Sociedad" : "Real_Sociedad", 
        "Sevilla" : "Sevilla",
        "Valencia" : "Valencia", 
        "Villarreal" : "Villarreal"
    }

    data = pd.concat(datos_equipo, ignore_index=True, sort=False)
    data = data[data.Opponent.isin(name_mapping.keys())]
    data["Opponent"] = data["Opponent"].apply(lambda row: name_mapping[row])

    return data


def clean_goals(x):
    if type(x) != str:
        if np.isnan(x):
            return 0.0
        if type(x) == int or type(x) == float:
            return int(x)
        else: 
            return Decimal(x)       
    if ( "(" not in x) and ("." not in x):
        return int(x)
    if "(" in x:
        return int(x.split("(")[0])
    if "." in x:
        return int(x.split(".")[0])
    return int(x)
 

def data_cleaning(df):

       
    df['GF'] = df['GF'].apply(lambda x: x if x != None else '0')
    # df['GF'] = df['GF'].apply(lambda x: str(re.sub(r'\D', '', x)))
    df['GF'] = df['GF'].apply(clean_goals)
    df['GF'] = df['GF'].astype('int')

    df['GA'] = df['GA'].apply(lambda x: x if x != None else '0')
    # df['GA'] = df['GA'].apply(lambda x: str(re.sub(r'\D', '', x)))
    df['GA'] = df['GA'].apply(clean_goals)
    df['GA'] = df['GA'].astype('int')
    
    df['xG'] = df['xG'].apply(lambda x: x if x != None else '0')
    df['xG'] = df['xG'].astype(float)

    df['xGA'] = df['xGA'].apply(lambda x: x if x != None else '0')
    df['xGA'] = df['xGA'].astype(float)

    df['xG'] = df['xG'].fillna(0)
    df['xGA'] = df['xGA'].fillna(0)

    return df

def add_current_points(df):
    # Ordena los datos por temporada, seasonstage, Jornada y fecha para asegurar un orden correcto
    df.sort_values(by=['Temporada', 'MetaEquipo', 'Date'], inplace=True)
    # Inicializa las columnas 'current_points', 'current_wins' y 'current_goals' en NaN (espacios vacíos)
    df['current_points'] = ''
    df['current_points_home'] = ''
    df['current_points_away'] = ''
    df['current_exp_points'] = ''
    df['current_exp_points_home'] = ''
    df['current_exp_points_away'] = ''
    df['current_wins'] = ''
    df['current_wins_home'] = ''
    df['current_wins_away'] = ''
    df['current_draws'] = ''
    df['current_draws_home'] = ''
    df['current_draws_away'] = ''
    df['current_losses'] = ''
    df['current_losses_home'] = ''
    df['current_losses_away'] = ''
    df['current_goals'] = ''
    df['current_goals_home'] = ''
    df['current_goals_away'] = ''
    df['current_exp_goals'] = ''
    df['current_exp_goals_away'] = ''
    df['current_exp_goals_home'] = ''
    df['current_goals_against'] = ''
    df['current_goals_against_home'] = ''
    df['current_goals_against_away'] = ''
    df['current_exp_goals_against'] = ''
    df['current_exp_goals_against_home'] = ''
    df['current_exp_goals_against_away'] = ''
    df['current_ranking_points'] = ''
    df['current_ranking_points_home'] = ''
    df['current_ranking_points_away'] = ''
    df['current_ranking_wins'] = ''
    df['current_ranking_wins_home'] = ''
    df['current_ranking_wins_away'] = ''
    df['current_ranking_goals'] = ''
    df['current_ranking_goals_home'] = ''
    df['current_ranking_goals_away'] = ''
    df['current_ranking_score'] = ''
    df['current_ranking_score_home'] = ''
    df['current_ranking_score_away'] = ''
    df['current_goals_difference'] = ''
    df['current_goals_difference_home'] = ''
    df['current_goals_difference_away'] = ''
    df['ranking'] = ''
    df['partidos_jugados'] = ''
    df['partidos_jugados_home'] = ''
    df['partidos_jugados_away'] = ''

    df = df[df.Comp == 'La Liga']
    df['Jornada'] = df.apply(lambda row: int(row['Round'].split(" ")[-1]), axis=1) 

    temporadas = df.Temporada.unique().tolist()
    # stages = ['Apertura','Clausura']
    jornadas = df.Jornada.unique().tolist()

    for temp in temporadas:
        # for stage in stages:
        puntos_acumulados = {}
        puntos_acumulados_home = {}
        puntos_acumulados_away = {}
        exp_puntos_acumulados = {}
        exp_puntos_acumulados_home = {}
        exp_puntos_acumulados_away = {}
        wins_acumulados = {}
        wins_acumulados_home = {}
        wins_acumulados_away = {}
        draws_acumulados = {}
        draws_acumulados_home = {}
        draws_acumulados_away = {}
        losses_acumulados = {}
        losses_acumulados_home = {}
        losses_acumulados_away = {}
        goals_acumulados = {}
        goals_acumulados_home = {}
        goals_acumulados_away = {}
        exp_goals_acumulados = {}
        exp_goals_acumulados_home = {}
        exp_goals_acumulados_away = {}
        goals_against_acumulados = {}
        goals_against_acumulados_home = {}
        goals_against_acumulados_away = {}
        exp_goals_against_acumulados = {}
        exp_goals_against_acumulados_home = {}
        exp_goals_against_acumulados_away = {}
        diferencia_goles_acumulados = {}
        diferencia_goles_acumulados_home = {}
        diferencia_goles_acumulados_away = {}
        partidos_jugados = {}
        partidos_jugados_home = {}
        partidos_jugados_away = {}
        sample = df[(df.Temporada == temp)].sort_values(by='Date')
            
        for index, row in sample.iterrows():
            equipo = row['MetaEquipo']

            # Verifica si el equipo ya está en el diccionario de puntos acumulados
            if equipo not in puntos_acumulados:
                puntos_acumulados[equipo] = 0
                puntos_acumulados_home[equipo] = 0
                puntos_acumulados_away[equipo] = 0
                exp_puntos_acumulados[equipo] = 0
                exp_puntos_acumulados_home[equipo] = 0
                exp_puntos_acumulados_away[equipo] = 0
                wins_acumulados[equipo] = 0
                wins_acumulados_home[equipo] = 0
                wins_acumulados_away[equipo] = 0
                draws_acumulados[equipo] = 0
                draws_acumulados_home[equipo] = 0
                draws_acumulados_away[equipo] = 0
                losses_acumulados[equipo] = 0
                losses_acumulados_home[equipo] = 0
                losses_acumulados_away[equipo] = 0
                goals_acumulados[equipo] = 0
                goals_acumulados_home[equipo] = 0
                goals_acumulados_away[equipo] = 0
                exp_goals_acumulados[equipo] = 0
                exp_goals_acumulados_home[equipo] = 0
                exp_goals_acumulados_away[equipo] = 0
                goals_against_acumulados[equipo] = 0
                goals_against_acumulados_home[equipo] = 0
                goals_against_acumulados_away[equipo] = 0
                exp_goals_against_acumulados[equipo] = 0
                exp_goals_against_acumulados_home[equipo] = 0
                exp_goals_against_acumulados_away[equipo] = 0
                diferencia_goles_acumulados[equipo] = 0
                diferencia_goles_acumulados_home[equipo] = 0
                diferencia_goles_acumulados_away[equipo] = 0
                partidos_jugados[equipo] = 0
                partidos_jugados_home[equipo] = 0
                partidos_jugados_away[equipo] = 0

            # Asigna los puntos según el resultado del partido
            if row['Venue'] == 'Home':
                partidos_jugados_home[equipo] += 1
                goals_acumulados_home[equipo] += row['GF']
                goals_against_acumulados_home[equipo] += row['GA']
                exp_goals_acumulados_home[equipo] += row['xG']
                exp_goals_against_acumulados_home[equipo] += row['xGA']
                diferencia_goles_acumulados_home[equipo] += row['GF'] - row['GA']
            else:
                partidos_jugados_away[equipo] += 1
                goals_acumulados_away[equipo] += row['GF']
                goals_against_acumulados_away[equipo] += row['GA']
                exp_goals_acumulados_away[equipo] += row['xG']
                exp_goals_against_acumulados_away[equipo] += row['xGA']
                diferencia_goles_acumulados_away[equipo] += row['GF'] - row['GA']

            if row['Result'] == 'W':
                puntos_acumulados[equipo] += 3
                wins_acumulados[equipo] += 1
                if row['Venue'] == 'Home':
                    puntos_acumulados_home[equipo] += 3
                    wins_acumulados_home[equipo] += 1
                    
                else:
                    puntos_acumulados_away[equipo] += 3
                    wins_acumulados_away[equipo] += 1


            elif row['Result'] == 'D':
                puntos_acumulados[equipo] += 1
                draws_acumulados[equipo] += 1
                if row['Venue'] == 'Home':
                    puntos_acumulados_home[equipo] += 1
                    draws_acumulados_home[equipo] += 1
                else:
                    puntos_acumulados_away[equipo] += 1
                    draws_acumulados_away[equipo] += 1
            else:
                losses_acumulados[equipo] += 1
                if row['Venue'] == 'Home':
                    losses_acumulados_home[equipo] += 1
                else:
                    losses_acumulados_away[equipo] += 1
            
            if row['Expected_Results'] == "W":
                exp_puntos_acumulados[equipo] += 3
                if row['Venue'] == 'Home':
                    exp_puntos_acumulados_home[equipo] += 3
                else:
                    exp_puntos_acumulados_away[equipo] += 3
            elif row['Expected_Results'] == "D":
                exp_puntos_acumulados[equipo] += 1
                if row['Venue'] == 'Home':
                    exp_puntos_acumulados_home[equipo] += 1
                else:
                    exp_puntos_acumulados_away[equipo] += 1

            # Suma los goles
            partidos_jugados[equipo] += 1
            goals_acumulados[equipo] += row['GF']
            exp_goals_acumulados[equipo] += row['xG']
            goals_against_acumulados[equipo] += row['GA']
            exp_goals_against_acumulados[equipo] += row['xGA']
            diferencia_goles_acumulados[equipo] += row['GF'] - row['GA']

            # Asigna los valores acumulados a las columnas correspondientes
            sample.at[index, 'current_points'] = puntos_acumulados[equipo]
            sample.at[index, 'current_points_home'] = puntos_acumulados_home[equipo]
            sample.at[index, 'current_points_away'] = puntos_acumulados_away[equipo]
            sample.at[index, 'current_exp_points'] = exp_puntos_acumulados[equipo]
            sample.at[index, 'current_exp_points_home'] = exp_puntos_acumulados_home[equipo]
            sample.at[index, 'current_exp_points_away'] = exp_puntos_acumulados_away[equipo]
            sample.at[index, 'current_wins'] = wins_acumulados[equipo]
            sample.at[index, 'current_wins_home'] = wins_acumulados_home[equipo]
            sample.at[index, 'current_wins_away'] = wins_acumulados_away[equipo]
            sample.at[index, 'current_draws'] = draws_acumulados[equipo]
            sample.at[index, 'current_draws_home'] = draws_acumulados_home[equipo]
            sample.at[index, 'current_draws_away'] = draws_acumulados_away[equipo]
            sample.at[index, 'current_losses'] = losses_acumulados[equipo]
            sample.at[index, 'current_losses_home'] = losses_acumulados_home[equipo]
            sample.at[index, 'current_losses_away'] = losses_acumulados_away[equipo]
            sample.at[index, 'current_goals'] = goals_acumulados[equipo]
            sample.at[index, 'current_goals_home'] = goals_acumulados_home[equipo]
            sample.at[index, 'current_goals_away'] = goals_acumulados_away[equipo]
            sample.at[index, 'current_exp_goals'] = exp_goals_acumulados[equipo]
            sample.at[index, 'current_exp_goals_home'] = exp_goals_acumulados_home[equipo]
            sample.at[index, 'current_exp_goals_away'] = exp_goals_acumulados_away[equipo]
            sample.at[index, 'current_goals_against'] = goals_against_acumulados[equipo]
            sample.at[index, 'current_goals_against_home'] = goals_against_acumulados_home[equipo]
            sample.at[index, 'current_goals_against_away'] = goals_against_acumulados_away[equipo]
            sample.at[index, 'current_exp_goals_against'] = exp_goals_against_acumulados[equipo]
            sample.at[index, 'current_exp_goals_against_home'] = exp_goals_against_acumulados_home[equipo]
            sample.at[index, 'current_exp_goals_against_away'] = exp_goals_against_acumulados_away[equipo]
            sample.at[index, 'current_goals_difference'] = diferencia_goles_acumulados[equipo]
            sample.at[index, 'current_goals_difference_home'] = diferencia_goles_acumulados_home[equipo]
            sample.at[index, 'current_goals_difference_away'] = diferencia_goles_acumulados_away[equipo]
            sample.at[index, 'partidos_jugados'] = partidos_jugados[equipo]
            sample.at[index, 'partidos_jugados_home'] = partidos_jugados_home[equipo]
            sample.at[index, 'partidos_jugados_away'] = partidos_jugados_away[equipo]
        

        for jornada in jornadas:
            subsample = sample[sample.Jornada == jornada]
            subsample.sort_values(by=['current_points','current_goals_difference','current_goals'], inplace=True, ascending=False)
            subsample['ranking'] = range(1,len(subsample) + 1)
            sample.loc[subsample.index] = subsample
            
        df.loc[sample.index] = sample
    

    aux_df = df[['Date','MetaEquipo','ranking']]
    aux_df = aux_df.rename(columns={'MetaEquipo':'Opponent', 'ranking':'opponent_ranking'})

    df = df.merge(aux_df, on=['Date','Opponent'], how='left')
    return df

def add_columns(df):

    df['TotalGoals'] = df['GF'] + df['GA']
    df['GoalsDifference'] = df['GF'] - df['GA']
    df['AA'] = df.apply(lambda row: 1 if row['GF'] > 0 and row['GA'] > 0 else 0, axis=1)


    df['Expected_GF'] = df['xG'].apply(lambda x: round(x))
    df['Expected_GA'] = df['xGA'].apply(lambda x: round(x))

    # Write the column "Expected_Results" Which is "W" if round(xG) > round(xGA), "D" if round(xG) == round(xGA) and "L" if round(xG) < round(xGA)
    df['Expected_Results'] = df.apply(lambda row: 'W' if row['Expected_GF'] > row['Expected_GA'] else 'D' if row['Expected_GF'] == row['Expected_GA'] else 'L', axis=1)
    df['Expected_Goals_Difference'] = df['Expected_GF'] - df['Expected_GA']

    df['match_points'] = df.apply(lambda row: 3 if row['Result'] == 'W' else 1 if row['Result'] == 'D' else 0, axis=1)
    df['match_expected_points'] = df.apply(lambda row: 3 if row['Expected_Results'] == 'W' else 1 if row['Result'] == 'D' else 0, axis=1)
    df = add_current_points(df)
    return df

def update_scores_and_fixtures(season):
    #Filter metadata to control the game seasons you want to retrieve
    metadata = pd.read_csv("data/csvdata/metadata.csv")
    metadata['Temporada'] = metadata.apply(lambda row: row['Link'].split('/')[-2], axis=1)
    metadata = metadata[metadata.Temporada == season]

    retrieved_data = fbref_pull_and_store_data(metadata)

    retrieved_data = data_cleaning(retrieved_data)
    retrieved_data = add_columns(retrieved_data)

    db_path = "data/sqldata/Historic_scores_and_fixtures.db"

    conn = sqlite3.connect(db_path)

    current_total_data = pd.read_sql_query("SELECT * FROM scores_and_fixture", conn)
    current_total_data = current_total_data[~current_total_data.Temporada.isin([season])]

    new_data = pd.concat([current_total_data, retrieved_data], ignore_index=True, sort=False)
    new_data.to_sql('scores_and_fixture', conn, if_exists='replace', index=False)
    new_data.to_csv('data/csvdata/scores_and_fixtures.csv')

if __name__ == '__main__':

    update_scores_and_fixtures(season='2024-2025')
