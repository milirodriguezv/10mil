import numpy as np
from utils import puntaje_y_no_usados, JUGADA_PLANTARSE, JUGADA_TIRAR, JUGADAS_STR
from collections import defaultdict
from tqdm import tqdm
from jugador import Jugador

class AmbienteDiezMil:
    
    def __init__(self):
        """Definir las variables de instancia de un ambiente.
        ¿Qué es propio de un ambiente de 10.000?
        """
        self.puntaje_acumulado = 0
        self.dados = np.random.randint(1,7,size=6)  # se inicializa con una primera tirada
        self.turno_terminado = False
        self.recompensa = 0

    def reset(self):
        """Reinicia el ambiente para volver a realizar un episodio.
        """
        self.puntaje_acumulado = 0
        self.dados = np.random.randint(1,7,size=6) 
        self.turno_terminado = False
        self.recompensa = 0

    def step(self, accion):
        """Dada una acción devuelve una recompensa.
        El estado es modificado acorde a la acción y su interacción con el ambiente.
        Podría ser útil devolver si terminó o no el turno.

        Args:
            accion: Acción elegida por un agente.

        Returns:
            tuple[int, bool]: Una recompensa y un flag que indica si terminó el turno. 
        """
        if accion == JUGADA_PLANTARSE:
            # self.turno_terminado = True
            self.dados = np.random.randint(1,7,size=6) # PREG
            
        elif accion == JUGADA_TIRAR:
            resultado = puntaje_y_no_usados(self.dados)
            if resultado.first == 0: # si en esa jugada no se suma nada
                self.puntaje_acumulado = 0
                self.dados = np.random.randint(1,7,size=6)
            else:
                self.puntaje_acumulado += resultado.first
                self.dados = resultado.second

            self.turno_terminado = False # PREG
        
        elif self.puntaje_acumulado >= 10000:
            self.turno_terminado = True
            self.recompensa = 1
        
        return (self.recompensa, self.turno_terminado)


class EstadoDiezMil:
    def __init__(self, dados, puntaje_acumulado, turno_terminado):
        """Definir qué hace a un estado de diez mil.
        Recordar que la complejidad del estado repercute en la complejidad de la tabla del agente de q-learning.
        """
        self.dados = dados
        self.puntaje_acumulado = puntaje_acumulado
        self.turno_terminado = turno_terminado
        

    def actualizar_estado(self, *args, **kwargs) -> None:
        """Modifica las variables internas del estado luego de una tirada.

        Args:
            ... (_type_): _description_
            ... (_type_): _description_
        """
        self.dados = kwargs.get('dados', self.dados)
        self.puntaje_acumulado = kwargs.get('puntaje_acumulado', self.puntaje_acumulado)
        self.turno_terminado = kwargs.get('turno_terminado', self.turno_terminado)
    
    def fin_turno(self):  # PREG
        """Modifica el estado al terminar el turno.
        """
        self.turno_terminado = True

    def __str__(self):
        """Representación en texto de EstadoDiezMil.
        Ayuda a tener una versión legible del objeto.

        Returns:
            str: Representación en texto de EstadoDiezMil.
        """
        pass   

class AgenteQLearning:
    def __init__(
        self,
        ambiente: AmbienteDiezMil,
        alpha: float,
        gamma: float,
        epsilon: float,
        *args,
        **kwargs
    ):
        """Definir las variables internas de un Agente que implementa el algoritmo de Q-Learning.

        Args:
            ambiente (AmbienteDiezMil): Ambiente con el que interactuará el agente.
            alpha (float): Tasa de aprendizaje.
            gamma (float): Factor de descuento.
            epsilon (float): Probabilidad de explorar.
        """
        pass

    def elegir_accion(self):
        """Selecciona una acción de acuerdo a una política ε-greedy.
        """
        pass

    def entrenar(self, episodios: int, verbose: bool = False) -> None:
        """Dada una cantidad de episodios, se repite el ciclo del algoritmo de Q-learning.
        Recomendación: usar tqdm para observar el progreso en los episodios.

        Args:
            episodios (int): Cantidad de episodios a iterar.
            verbose (bool, optional): Flag para hacer visible qué ocurre en cada paso. Defaults to False.
        """
        pass

    def guardar_politica(self, filename: str):
        """Almacena la política del agente en un formato conveniente.

        Args:
            filename (str): Nombre/Path del archivo a generar.
        """
        pass

class JugadorEntrenado(Jugador):
    def __init__(self, nombre: str, filename_politica: str):
        self.nombre = nombre
        self.politica = self._leer_politica(filename_politica)
        
    def _leer_politica(self, filename:str, SEP:str=','):
        """Carga una politica entrenada con un agente de RL, que está guardada
        en el archivo filename en un formato conveniente.

        Args:
            filename (str): Nombre/Path del archivo que contiene a una política almacenada. 
        """
        pass
    
    def jugar(
        self,
        puntaje_total:int,
        puntaje_turno:int,
        dados:list[int],
    ) -> tuple[int,list[int]]:
        """Devuelve una jugada y los dados a tirar.

        Args:
            puntaje_total (int): Puntaje total del jugador en la partida.
            puntaje_turno (int): Puntaje en el turno del jugador
            dados (list[int]): Tirada del turno.

        Returns:
            tuple[int,list[int]]: Una jugada y la lista de dados a tirar.
        """
        pass
        # puntaje, no_usados = puntaje_y_no_usados(dados)
        # COMPLETAR
        # estado = ...
        # jugada = self.politica[estado]
       
        # if jugada==JUGADA_PLANTARSE:
        #     return (JUGADA_PLANTARSE, [])
        # elif jugada==JUGADA_TIRAR:
        #     return (JUGADA_TIRAR, no_usados)