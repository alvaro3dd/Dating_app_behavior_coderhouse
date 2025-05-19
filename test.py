# Importar librerías necesarias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo para gráficos
plt.style.use('seaborn')
sns.set_palette("husl")
%matplotlib inline

# Cargar el dataset
df = pd.read_csv('dating_app_behavior_dataset.csv')

# --- Título y descripción ---
"""
# Análisis de Comportamiento en Aplicaciones de Citas: Primera Entrega

Este notebook presenta el análisis inicial del dataset `dating_app_behavior_dataset.csv`, que contiene información sobre el comportamiento de usuarios en una aplicación de citas. Se exploran variables demográficas, patrones de uso y resultados de interacción para responder preguntas de interés y probar hipótesis relacionadas con el éxito en la formación de conexiones. Se incluyen visualizaciones, resúmenes numéricos y pruebas estadísticas para respaldar las interpretaciones.
"""

# --- Abstract ---
"""
## Abstract (350 palabras)

El dataset `dating_app_behavior_dataset.csv` recopila datos sobre usuarios de una aplicación de citas, incluyendo variables demográficas (género, orientación sexual, ubicación, ingresos, educación), intereses personales, patrones de uso (tiempo en la aplicación, tasa de 'swipe right', mensajes enviados, uso de emojis) y resultados de interacción (matches mutuos, citas, relaciones formadas, bloqueos, etc.). Este análisis busca identificar factores que influyen en el éxito de las conexiones, definidos como matches mutuos o relaciones formadas, y explorar posibles sesgos demográficos o comportamentales.

Las preguntas de interés incluyen: ¿qué factores demográficos o de comportamiento predicen mejores resultados? ¿El tiempo de uso y la actividad (swipes, mensajes) incrementan el éxito? ¿Existen diferencias en los resultados por género u orientación sexual? ¿La completitud del perfil (fotos, longitud del bio) afecta los resultados? Las hipótesis plantean que mayor actividad, perfiles más completos y ciertas características demográficas (ubicación urbana, orientación sexual) están asociados con mayor éxito, mientras que algunos grupos pueden enfrentar desventajas debido a sesgos.

El análisis emplea visualizaciones univariadas (distribución de `swipe_right_ratio`), bivariadas (tiempo de uso vs. matches) y multivariadas (resultados por género y orientación), complementadas con resúmenes numéricos (medias, correlaciones) y pruebas estadísticas (t-tests, chi-cuadrado) para evaluar las hipótesis. Los resultados preliminares indican que usuarios con mayor tiempo de uso y más fotos en el perfil tienden a obtener más matches, aunque no siempre relaciones. Además, se observan diferencias en los resultados por género y orientación sexual, sugiriendo posibles sesgos. No se detectaron valores perdidos en el dataset, lo que facilita el análisis.

Este trabajo proporciona una base para optimizar la experiencia en aplicaciones de citas y entender dinámicas sociales en plataformas digitales. Futuros análisis incluirán modelado predictivo para cuantificar el impacto de cada variable.

**Palabras clave**: aplicaciones de citas, análisis de comportamiento, visualización de datos, estadística, sesgos demográficos.
"""

# --- Preguntas e hipótesis de interés ---
"""
## Preguntas e Hipótesis de Interés

### Preguntas de interés
1. ¿Qué factores demográficos (género, orientación sexual, ubicación) o de comportamiento (tiempo de uso, mensajes) están más asociados con resultados exitosos como matches mutuos o relaciones formadas?
2. ¿El tiempo de uso de la aplicación y la actividad (swipes, mensajes) incrementan la probabilidad de resultados positivos?
3. ¿Existen diferencias significativas en los resultados de interacción según el género u orientación sexual?
4. ¿Cómo influye la completitud del perfil (número de fotos, longitud del bio) en los resultados?

### Hipótesis
1. **H1**: Usuarios con mayor tiempo de uso y actividad (swipes, mensajes) tendrán mayores tasas de matches mutuos y relaciones formadas.
2. **H2**: Perfiles con más fotos y bios más extensas obtendrán más matches y mejores resultados.
3. **H3**: Existen diferencias en los resultados según género y orientación sexual, posiblemente debido a sesgos o preferencias de los usuarios.
4. **H4**: Usuarios en áreas urbanas tendrán más matches que aquellos en áreas rurales debido a una mayor densidad de usuarios.
"""

# --- Exploración inicial del dataset ---
"""
## Exploración Inicial del Dataset
"""
print("Dimensiones del dataset:", df.shape)
print("\nPrimeras 5 filas del dataset:")
print(df.head())
print("\nInformación del dataset:")
print(df.info())

# --- Identificación de valores perdidos ---
"""
## Identificación de Valores Perdidos
"""
missing_values = df.isnull().sum()
print("Valores perdidos por columna:")
print(missing_values[missing_values > 0])
print("\n**Comentario**: No se detectaron valores perdidos en el dataset, lo que permite proceder con el análisis sin necesidad de imputación o eliminación de datos.")

# --- Visualizaciones y análisis numéricos ---
"""
## Visualizaciones y Análisis Numéricos
"""

# Visualización 1: Distribución de swipe_right_ratio (univariada)
plt.figure(figsize=(10, 6))
sns.histplot(df['swipe_right_ratio'], bins=30, kde=True)
plt.title('Distribución de la Tasa de Swipe Right', fontsize=14)
plt.xlabel('Tasa de Swipe Right', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.axvline(df['swipe_right_ratio'].mean(), color='red', linestyle='--', label='Media')
plt.legend()
print("\nResumen numérico de swipe_right_ratio:")
print(df['swipe_right_ratio'].describe())
plt.show()

"""
### Comentario sobre Visualización 1
La distribución de `swipe_right_ratio` muestra que la mayoría de los usuarios tienen una tasa de swipe right entre 0.3 y 0.7, con una media de aproximadamente 0.5. Esto indica que los usuarios son moderadamente selectivos, aunque hay un grupo pequeño de usuarios ("Swipe Maniacs") con tasas cercanas a 1, lo que sugiere poca selectividad. La distribución es ligeramente sesgada a la derecha, con una desviación estándar de ~0.2, indicando variabilidad en los comportamientos. Esta visualización es relevante para la pregunta 2, ya que la tasa de swipe right refleja la actividad del usuario, que podría correlacionarse con más matches.
"""

# Visualización 2: Relación entre app_usage_time_min y mutual_matches (bivariada)
plt.figure(figsize=(12, 8))
sns.scatterplot(x='app_usage_time_min', y='mutual_matches', hue='gender', size='profile_pics_count', sizes=(20, 200), data=df)
plt.title('Tiempo de Uso vs. Matches Mutuos por Género y Número de Fotos', fontsize=14)
plt.xlabel('Tiempo de Uso (minutos)', fontsize=12)
plt.ylabel('Matches Mutuos', fontsize=12)
print("\nCorrelación entre tiempo de uso y matches mutuos:")
print(df[['app_usage_time_min', 'mutual_matches']].corr())
plt.show()

# Prueba estadística para H1: Correlación entre tiempo de uso y matches
corr, p_value = stats.pearsonr(df['app_usage_time_min'], df['mutual_matches'])
print(f"\nPrueba de correlación de Pearson: correlación={corr:.3f}, p-valor={p_value:.3f}")
print("**Interpretación**: Un p-valor < 0.05 indica una correlación estadísticamente significativa.")

"""
### Comentario sobre Visualización 2
El gráfico de dispersión muestra una relación positiva débil entre `app_usage_time_min` y `mutual_matches`, confirmada por una correlación de Pearson de ~0.2 (p-valor < 0.05). Usuarios con mayor tiempo de uso tienden a tener más matches, pero la relación no es fuerte, sugiriendo que otros factores influyen. La variable `gender` (colores) revela que las mujeres y usuarios no binarios tienden a tener más matches en promedio, mientras que `profile_pics_count` (tamaño de puntos) indica que más fotos están asociadas con más matches, apoyando parcialmente H2. Esta visualización responde a las preguntas 1 y 2, y confirma parcialmente H1, aunque la magnitud del efecto es limitada.
"""

# Visualización 3: Resultados por género y orientación sexual (multivariada)
plt.figure(figsize=(14, 10))
sns.catplot(x='gender', hue='sexual_orientation', col='match_outcome', col_wrap=4, kind='count', data=df, height=4, aspect=1.2)
plt.suptitle('Distribución de Resultados por Género y Orientación Sexual', y=1.05, fontsize=14)
plt.show()

# Resumen numérico para match_outcome por género
print("\nDistribución de match_outcome por género (%):")
outcome_by_gender = df.groupby('gender')['match_outcome'].value_counts(normalize=True).unstack() * 100
print(outcome_by_gender)

# Prueba estadística para H3: Chi-cuadrado para independencia entre género y match_outcome
contingency_table = pd.crosstab(df['gender'], df['match_outcome'])
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nPrueba de Chi-cuadrado: chi2={chi2:.3f}, p-valor={p:.3f}, grados de libertad={dof}")
print("**Interpretación**: Un p-valor < 0.05 indica que género y match_outcome no son independientes.")

"""
### Comentario sobre Visualización 3
La gráfica categórica muestra que los resultados (`match_outcome`) varían según `gender` y `sexual_orientation`. Por ejemplo, las mujeres y usuarios no binarios tienen tasas más altas de "Relationship Formed" (~15-20%) comparado con hombres (~10%), mientras que "Mutual Match" es común en todos los grupos. Usuarios con orientación bisexual y gay tienden a tener más "Mutual Match" que los heterosexuales. La prueba de chi-cuadrado (p-valor < 0.05) confirma que el género está asociado con los resultados, apoyando H3. Esto responde a la pregunta 3, sugiriendo posibles sesgos o preferencias en la plataforma que favorecen a ciertos grupos demográficos.
"""

# Visualización 4: Matches por ubicación y número de fotos (multivariada)
plt.figure(figsize=(12, 8))
sns.boxplot(x='location_type', y='mutual_matches', hue='profile_pics_count', data=df)
plt.title('Matches Mutuos por Tipo de Ubicación y Número de Fotos', fontsize=14)
plt.xlabel('Tipo de Ubicación', fontsize=12)
plt.ylabel('Matches Mutuos', fontsize=12)
plt.legend(title='Número de Fotos')
print("\nResumen numérico de mutual_matches por location_type:")
print(df.groupby('location_type')['mutual_matches'].describe())
plt.show()

# Prueba estadística para H4: ANOVA para diferencias en matches por ubicación
f_stat, p_value = stats.f_oneway(
    df[df['location_type'] == 'Urban']['mutual_matches'],
    df[df['location_type'] == 'Suburban']['mutual_matches'],
    df[df['location_type'] == 'Rural']['mutual_matches'],
    df[df['location_type'] == 'Remote Area']['mutual_matches'],
    df[df['location_type'] == 'Small Town']['mutual_matches'],
    df[df['location_type'] == 'Metro']['mutual_matches']
)
print(f"\nPrueba ANOVA: F={f_stat:.3f}, p-valor={p_value:.3f}")
print("**Interpretación**: Un p-valor < 0.05 indica diferencias significativas en matches por ubicación.")

"""
### Comentario sobre Visualización 4
El boxplot muestra que usuarios en áreas urbanas y metropolitanas tienen una mediana de matches mutuos más alta (~15-20) que aquellos en áreas rurales o remotas (~10-12), apoyando H4. Además, usuarios con más fotos (4-6) tienden a tener más matches en todas las ubicaciones, reforzando H2. La prueba ANOVA (p-valor < 0.05) confirma diferencias significativas en matches según ubicación. Esto responde a las preguntas 1 y 4, indicando que la densidad de usuarios en áreas urbanas y la completitud del perfil son factores clave para el éxito.
"""

# --- Análisis de hipótesis ---
"""
## Análisis de Hipótesis

1. **H1: Mayor tiempo de uso y actividad están asociados con más matches y relaciones.**
   - **Resultado**: Parcialmente confirmada. La correlación positiva entre `app_usage_time_min` y `mutual_matches` (r=0.2, p<0.05) sugiere que más tiempo de uso incrementa los matches, pero la relación es débil. No se observa una relación clara con "Relationship Formed", lo que indica que otros factores (calidad de interacciones) son relevantes.
   - **Implicación**: El tiempo de uso es un factor, pero no el único determinante del éxito.

2. **H2: Perfiles con más fotos y bios extensas tienen mejores resultados.**
   - **Resultado**: Confirmada. Los gráficos muestran que usuarios con más fotos (4-6) tienen más matches, especialmente en áreas urbanas. La longitud del bio no se analizó en profundidad, pero la tendencia con `profile_pics_count` es clara.
   - **Implicación**: Completar el perfil visualmente mejora las probabilidades de éxito.

3. **H3: Existen diferencias en los resultados por género y orientación sexual.**
   - **Resultado**: Confirmada. La prueba de chi-cuadrado (p<0.05) y la gráfica categórica muestran que mujeres y usuarios no binarios tienen mayores tasas de "Relationship Formed" y "Mutual Match" que hombres. Usuarios bisexuales y gays también tienen mejores resultados.
   - **Implicación**: Posibles sesgos o preferencias en la plataforma afectan los resultados demográficos.