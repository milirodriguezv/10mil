import numpy as np
from template import EstadoDiezMil, AmbienteDiezMil, AgenteQLearning, JugadorEntrenado
from utils import JUGADAS_STR

# Definimos el estado inicial
dados_iniciales = [1, 2, 3, 4, 5, 6]
estado_inicial = EstadoDiezMil(dados_iniciales, puntaje_total=0, puntaje_turno=0)

# Inicializamos el ambiente con el estado inicial
ambiente = AmbienteDiezMil(estado_inicial)

# Definimos el agente con sus hiperparámetros
alpha = 0.1   # Tasa de aprendizaje
gamma = 0.9   # Factor de descuento
epsilon = 0.1 # Probabilidad de exploración

agente = AgenteQLearning(ambiente, estado_inicial, alpha, gamma, epsilon)

# Entrenamos al agente por una cantidad de episodios
episodios = 1000
agente.entrenar(episodios, verbose=True)

# Guardamos la política entrenada
agente.guardar_politica("politica_entrenada.txt")

# Ahora cargamos la política entrenada y jugamos con un JugadorEntrenado
jugador_entrenado = JugadorEntrenado("AgenteEntrenado", "politica_entrenada.txt")

# Simulamos un turno
dados = [1, 2, 3, 4, 5, 6]
puntaje_total = 0
puntaje_turno = 0

# El jugador entrenado toma una decisión basada en la política entrenada
jugada, dados_a_tirar = jugador_entrenado.jugar(puntaje_total, puntaje_turno, dados)

# Imprimimos el resultado de la jugada
print(f"Jugada seleccionada: {JUGADAS_STR[jugada]}")
print(f"Dados a tirar: {dados_a_tirar}")
