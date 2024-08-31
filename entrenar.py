import argparse
from template import AmbienteDiezMil, AgenteQLearning, EstadoDiezMil

def main(episodios, verbose):
    dados = [1,2,3,4,5,6]
    puntaje_total = 0
    puntaje_turno = 0

    estado = EstadoDiezMil(dados, puntaje_total, puntaje_turno)
    # Crear una instancia del ambiente
    ambiente = AmbienteDiezMil(estado)

    # Crear un agente de Q-learning
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.1
    agente = AgenteQLearning(ambiente, estado, alpha, gamma, epsilon)

    # Entrenar al agente con un número de episodios
    agente.entrenar(episodios, verbose=verbose)
    agente.guardar_politica(f"politica_{episodios}.csv")


if __name__ == '__main__':
    # Crear un analizador de argumentos
    parser = argparse.ArgumentParser(description="Entrenar un agente usando Q-learning en el ambiente de 'Diez Mil'.")

    # Agregar argumentos
    parser.add_argument('-e', '--episodios', type=int, default=500000, help='Número de episodios para entrenar al agente (default: 10000)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Activar modo verbose para ver más detalles durante el entrenamiento')

    # Parsear los argumentos
    args = parser.parse_args()

    # Llamar a la función principal con los argumentos proporcionados
    main(args.episodios, args.verbose)
