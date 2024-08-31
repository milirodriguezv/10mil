from diezmil import JuegoDiezMil
from jugador import JugadorAleatorio, JugadorSiempreSePlanta
from template import JugadorEntrenado
import numpy as np
from statistics import mean

def jugar_y_registrar(jugador, nombre, datos):
    juego = JuegoDiezMil(jugador)
   
    cantidad_turnos, puntaje_final = juego.jugar(verbose=False)
    datos["CANT_TURNOS"].append(cantidad_turnos)
    datos["PUNTAJE_FINAL"].append(puntaje_final)
  

def calcular_promedios(datos):
    return {
        "CANT_TURNOS_PROMEDIO": mean(datos["CANT_TURNOS"]),
        "PUNTAJE_PROMEDIO": mean(datos["PUNTAJE_FINAL"])
    }

def main():
    datos_aleatorio = {"CANT_TURNOS": [], "PUNTAJE_FINAL": []}
    datos_plantar = {"CANT_TURNOS": [], "PUNTAJE_FINAL": []}
    datos_qlearning = {"CANT_TURNOS": [], "PUNTAJE_FINAL": []}

    for _ in range(100):
        jugar_y_registrar(JugadorAleatorio('random'), "Aleatorio", datos_aleatorio)
        jugar_y_registrar(JugadorSiempreSePlanta('plantón'), "Plantón", datos_plantar)
        jugar_y_registrar(JugadorEntrenado("QLearning", "politica_500000.csv"), "Q-Learning", datos_qlearning)
    
    resultados = {
        "ALEATORIO": calcular_promedios(datos_aleatorio),
        "PLANTON": calcular_promedios(datos_plantar),
        "Q-LEARNING": calcular_promedios(datos_qlearning)
    }

    print(resultados)

if __name__ == '__main__':
    main()