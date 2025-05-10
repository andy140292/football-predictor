import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

# Cargar los datos
df = pd.read_csv('results.csv', parse_dates=['date'])

# Filtrar solo eliminatorias
df = df[df['tournament'] == 'FIFA World Cup qualification'].copy()

# Ordenar por fecha
df = df.sort_values('date')

# Crear historial de partidos
home_matches = df[['date', 'home_team']].rename(columns={'home_team': 'team', 'date': 'match_date'})
away_matches = df[['date', 'away_team']].rename(columns={'away_team': 'team', 'date': 'match_date'})
team_matches = pd.concat([home_matches, away_matches])
team_matches = team_matches.sort_values(['team', 'match_date'])
team_matches['previous_match_date'] = team_matches.groupby('team')['match_date'].shift(1)
team_matches['days_rest'] = (team_matches['match_date'] - team_matches['previous_match_date']).dt.days

# Mapeo de días de descanso
rest_lookup = team_matches.set_index(['team', 'match_date'])['days_rest']
df['home_days_rest'] = df.apply(lambda row: rest_lookup.get((row['home_team'], row['date'])), axis=1)
df['away_days_rest'] = df.apply(lambda row: rest_lookup.get((row['away_team'], row['date'])), axis=1)

# Diferencia de días de descanso
df['rest_diff'] = df['home_days_rest'] - df['away_days_rest']

# Resultado del partido: 1 si gana local, 0 si empate o derrota
df['home_win'] = (df['home_score'] > df['away_score']).astype(int)

# ---------------------------------------------------------------
teams = ['Argentina','Brazil','Uruguay','Colombia','Chile','Ecuador','Paraguay','Peru','Bolivia','Venezuela']  # Cambias aquí el nombre cuando quieras

for team in teams:
    # Filtrar partidos donde juega el equipo
    df_team = df[
        (df['home_team'] == team) | (df['away_team'] == team)
    ].copy()

    # Definir días de descanso para el equipo
    df_team['team_days_rest'] = df_team.apply(
        lambda row: row['home_days_rest'] if row['home_team'] == team else row['away_days_rest'], 
        axis=1
    )

    # Calcular días de descanso para el rival
    df_team['opponent_days_rest'] = df_team.apply(
        lambda row: row['away_days_rest'] if row['home_team'] == team else row['home_days_rest'],
        axis=1
)

    # Definir resultado del equipo
    def get_team_result(row):
        if row['home_team'] == team:
            if row['home_score'] > row['away_score']:
                return 1  # Ganó
            elif row['home_score'] < row['away_score']:
                return -1  # Perdió
            else:
                return 0  # Empató
        else:
            if row['away_score'] > row['home_score']:
                return 1  # Ganó
            elif row['away_score'] < row['home_score']:
                return -1  # Perdió
            else:
                return 0  # Empató

    df_team['team_result'] = df_team.apply(get_team_result, axis=1)

    # # 1. Todos los partidos (baseline)
    # all_matches_win_rate = (df_team['team_result'] == 1).mean()

    # # 2. Cuando descansó 4 días o menos
    # low_rest_win_rate = (df_team[df_team['team_days_rest'] <= 5]['team_result'] == 1).mean()

    # # 3. Cuando descansó más de 4 días
    # high_rest_win_rate = (df_team[df_team['team_days_rest'] > 5]['team_result'] == 1).mean()

    # # Mostrar resultados
    # print(f"Tasa de victoria de {team} en todos los partidos: {all_matches_win_rate:.2%}")
    # print(f"Tasa de victoria de {team} con <= 5 días de descanso: {low_rest_win_rate:.2%}")
    # print(f"Tasa de victoria de {team} con > 5 días de descanso: {high_rest_win_rate:.2%}")

    # Crear columna para agrupar
    def categorize_rest(x):
        if x <= 5:
            return int(x)  # Días 1 a 7 individuales
        else:
            return '>5'    # Agrupar todo lo demás

    df_team['rest_group'] = df_team['team_days_rest'].apply(categorize_rest)

    # Calcular porcentaje de victorias por grupo
    rest_group_win_rate = df_team.groupby('rest_group')['team_result'].apply(lambda x: (x == 1).mean())

    # Calcular número de partidos jugados por grupo
    rest_group_match_count = df_team.groupby('rest_group')['team_result'].count()

    # Ordenar el índice para que salgan 1-2-3...7-">7"
    ordered_index = [4,5,'>5']
    rest_group_win_rate = rest_group_win_rate.reindex(ordered_index)
    rest_group_match_count = rest_group_match_count.reindex(ordered_index)

    # Creamos una lista de posiciones numéricas
    x_pos = np.arange(len(rest_group_win_rate))

    # # Graficar
    # plt.figure(figsize=(12,6))

    # # Dibujar barras en posiciones numéricas
    # plt.bar(x_pos, rest_group_win_rate.values, width=0.6)

    # # Añadir etiquetas (número de partidos)
    # for i, (y, count) in enumerate(zip(rest_group_win_rate.values, rest_group_match_count.values)):
    #     plt.text(x_pos[i], y + 0.02, f'partidos={count}', ha='center', fontsize=12)

    # # Cambiar los ticks del eje X a las categorías de descanso
    # plt.xticks(x_pos, rest_group_win_rate.index.astype(str))

    # # Configurar gráfico
    # plt.title(f'Porcentaje de victorias de {team} por días de descanso', fontsize=16)
    # plt.xlabel('Días de descanso',fontsize=12)
    # plt.ylabel('Porcentaje de victorias',fontsize=12)
    # plt.ylim(0, 1.1)
    # plt.grid(axis='y')
    # plt.show()


    # Casos donde el equipo tuvo más descanso que el rival
    df_more_rest = df_team[df_team['team_days_rest'] > df_team['opponent_days_rest']]

    # Casos donde el equipo tuvo menos descanso que el rival
    df_less_rest = df_team[df_team['team_days_rest'] < df_team['opponent_days_rest']]

    # Casos donde descansaron igual
    df_same_rest = df_team[df_team['team_days_rest'] == df_team['opponent_days_rest']]

    # % victorias cuando tuvo más descanso
    win_rate_more_rest = (df_more_rest['team_result'] == 1).mean()

    # % victorias cuando tuvo menos descanso
    win_rate_less_rest = (df_less_rest['team_result'] == 1).mean()

    # % victorias cuando descansaron igual
    win_rate_same_rest = (df_same_rest['team_result'] == 1).mean()

    # Mostrar resultados
    print(f"Tasa de victorias de {team} con MÁS descanso que el rival: {win_rate_more_rest:.2%}")
    print(f"Tasa de victorias de {team} con MENOS descanso que el rival: {win_rate_less_rest:.2%}")
    print(f"Tasa de victorias de {team} con IGUAL descanso que el rival: {win_rate_same_rest:.2%}")


    # Función para identificar si el equipo jugó como local
    df_team['is_home'] = df_team['home_team'] == team

    # Agrupar partidos
    df_more_rest = df_team[df_team['team_days_rest'] > df_team['opponent_days_rest']]
    df_less_rest = df_team[df_team['team_days_rest'] < df_team['opponent_days_rest']]
    df_same_rest = df_team[df_team['team_days_rest'] == df_team['opponent_days_rest']]

    # Crear función para calcular tasas
    def calculate_home_away_win_rates(df_group):
        home_win_rate = ((df_group[df_group['is_home']]['team_result']) == 1).mean()
        away_win_rate = ((df_group[~df_group['is_home']]['team_result']) == 1).mean()
        n_home = len(df_group[df_group['is_home']])
        n_away = len(df_group[~df_group['is_home']])
        return home_win_rate, away_win_rate, n_home, n_away

    # Calcular tasas
    home_more, away_more, n_home_more, n_away_more = calculate_home_away_win_rates(df_more_rest)
    home_less, away_less, n_home_less, n_away_less = calculate_home_away_win_rates(df_less_rest)
    home_same, away_same, n_home_same, n_away_same = calculate_home_away_win_rates(df_same_rest)

    # Datos para graficar
    categories = ['Menos descanso', 'Más descanso',]
    home_win_rates = [home_less, home_more]
    away_win_rates = [away_less, away_more]
    n_home_counts = [n_home_less, n_home_more]
    n_away_counts = [n_away_less,n_away_more]

    x = np.arange(len(categories))  # posiciones de las categorías
    width = 0.35  # ancho de las barras

    # Graficar
    plt.figure(figsize=(12,7))
    bars_home = plt.bar(x - width/2, home_win_rates, width, label='Local')
    bars_away = plt.bar(x + width/2, away_win_rates, width, label='Visitante')

    # Añadir etiquetas de número de partidos
    for i in range(len(categories)):
        plt.text(x[i] - width/2, home_win_rates[i] + 0.02, f'partidos={n_home_counts[i]}', ha='center', fontsize=12)
        plt.text(x[i] + width/2, away_win_rates[i] + 0.02, f'partidos={n_away_counts[i]}', ha='center', fontsize=12)

    # Configurar gráfico
    plt.xticks(x, categories, fontsize=14)
    plt.title(f'% de victorias de {team} según descanso y condición (L/V)', fontsize=18)
    plt.ylabel('Porcentaje de victorias', fontsize=14)
    plt.ylim(0, 1.1)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(axis='y')
    plt.show()

# ---------------------------------------------------------------

# # Preparamos las variables
# X = df[['rest_diff']].fillna(0)
# X = sm.add_constant(X)
# y = df['home_win']

# # Modelo
# model = sm.Logit(y, X).fit()

# # Resultados
# print(model.summary())

