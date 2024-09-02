from diezmil import JuegoDiezMil
from jugador import JugadorAleatorio, JugadorSiempreSePlanta
from template import JugadorEntrenado
from statistics import mean

def jugar_y_registrar(jugador, nombre, num_juegos):
    turnos = []
    puntajes = []

    for _ in range(num_juegos):
        juego = JuegoDiezMil(jugador)
        cantidad_turnos, puntaje_final = juego.jugar(verbose=False)
        turnos.append(cantidad_turnos)
        puntajes.append(puntaje_final)

    print(f"La cantidad promedio de turnos en {num_juegos} juegos del jugador {nombre} es de {mean(turnos):.2f}.")
    print(f"El puntaje promedio de {nombre} es de {mean(puntajes):.2f}.\n")

def main():
    num_juegos = 1000
    jugadores = [
        (JugadorAleatorio('random'), "Aleatorio"),
        (JugadorSiempreSePlanta('plantón'), "Plantón"),
        (JugadorEntrenado("QLearning_Alpha0.1", "politica_100000_alpha_0.1.csv"), "Q-Learning_Alpha0.1"),
        (JugadorEntrenado("QLearning_Alpha0.2", "politica_100000_alpha_0.2.csv"), "Q-Learning_Alpha0.2"),
        (JugadorEntrenado("QLearning_Alpha0.3", "politica_100000_alpha_0.3.csv"), "Q-Learning_Alpha0.3")
    ]
        

    for jugador, nombre in jugadores:
        jugar_y_registrar(jugador, nombre, num_juegos)

if __name__ == '__main__':
    main()
