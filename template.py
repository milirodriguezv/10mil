import numpy as np
from utils import puntaje_y_no_usados, JUGADA_PLANTARSE, JUGADA_TIRAR, JUGADAS_STR
from collections import defaultdict
from tqdm import tqdm
from jugador import Jugador
import random
from random import randint

class AmbienteDiezMil:
    
    def __init__(self, estado):
        """Definir las variables de instancia de un ambiente.
        ¿Qué es propio de un ambiente de 10.000?
        """
        self.estado = estado 
        self.cant_turnos = 0
        self.recompensa = 0
        self.turno_terminado = False


    def reset(self):
        """Reinicia el ambiente para volver a realizar un episodio.
        """
        # self.estado.puntaje_total = 0
        # self.puntaje_turno = 0
        # self.estado.dados = [1, 2, 3, 4, 5, 6]  
        # self.estado.turno_terminado = False
        self.estado.actualizar_estado(puntaje_total=0, puntaje_turno=0, dados=[1,2,3,4,5,6])
        self.recompensa = 0
        self.cant_turnos = 0
        self.turno_terminado = False

    def step(self, accion):
        """Dada una acción devuelve una recompensa.
        El estado es modificado acorde a la acción y su interacción con el ambiente.
        Podría ser útil devolver si terminó o no el turno.

        Args:
            accion: Acción elegida por un agente.

        Returns:
            tuple[int, bool]: Una recompensa y un flag que indica si terminó el turno. 
        """
        # chequeamos que no haya llegado al tope de turnos
        if self.cant_turnos == 1000:
            self.estado.fin_turno()
            self.recompensa = -1
            self.turno_terminado = True

        if accion == JUGADA_PLANTARSE: 
            # self.estado.turno_terminado = True
            # self.estado.dados = [1, 2, 3, 4, 5, 6] # vuelve a tener todos los dados 
            # self.estado.puntaje_total += self.puntaje_turno # sumamos el puntaje del turno cuando decide plantarse
            # self.puntaje_turno = 0
            self.recompensa = self.estado.puntaje_turno / 10000
            self.estado.fin_turno()
            self.turno_terminado = True  
            
        elif accion == JUGADA_TIRAR:

            #self.estado.dados = [randint(1, 6) for _ in range(len(self.estado.dados))]
            puntaje, no_usados = puntaje_y_no_usados(self.estado.dados)
            
            if puntaje == 0: # si en esa jugada no se suma nada
                # self.puntaje_turno = 0
                # self.estado.dados = [1, 2, 3, 4, 5, 6]
                # self.estado.turno_terminado = True
                self.estado.fin_turno()
                self.recompensa = -1
                self.turno_terminado = True
              

            else:
                nuevo_puntaje_turno = self.estado.puntaje_turno + puntaje
                nuevo_puntaje_total = self.estado.puntaje_total + self.estado.puntaje_turno
                self.estado.actualizar_estado(puntaje_total=nuevo_puntaje_total, puntaje_turno=nuevo_puntaje_turno, dados=no_usados)
                self.recompensa = nuevo_puntaje_turno

                if len(self.estado.dados) == 0:  # si usó todos los dados, puede volver a tirar todos
                    self.estado.dados = [1, 2, 3, 4, 5, 6]
                
                self.turno_terminado = False

        # condición de que ganó
        if self.estado.puntaje_total >= 10000:  
            self.estado.fin_turno()
            self.recompensa = 1  
            self.turno_terminado = True

        self.cant_turnos += 1 # aumentamos la cantidad de turnos

        return (self.recompensa, self.turno_terminado)


class EstadoDiezMil:
    def __init__(self, dados, puntaje_total, puntaje_turno):
        """Definir qué hace a un estado de diez mil.
        Recordar que la complejidad del estado repercute en la complejidad de la tabla del agente de q-learning.
        """
        self.dados = dados 
        self.puntaje_total = puntaje_total 
        self.puntaje_turno = puntaje_turno
        

    def actualizar_estado(self, *args, **kwargs) -> None: 
        # tener en cuenta: que dados me quedan despues de tirar?, que puntaje obtengo despues de tirar?, termina el turno?
        """Modifica las variables internas del estado luego de una tirada.
        Args:
            ... (_type_): _description_
            ... (_type_): _description_
        """
        self.dados = kwargs.get('dados', self.dados)
        self.puntaje_total = kwargs.get('puntaje_total', self.puntaje_total)
        self.puntaje_turno = kwargs.get('puntaje_turno', self.puntaje_turno)

        
    
    def fin_turno(self):  # preguntar si la actualizacion de los puntos esta bien 
        """Modifica el estado al terminar el turno.
        """
        self.dados = [1, 2, 3, 4, 5, 6]
        self.puntaje_total += self.puntaje_turno
        self.puntaje_turno = 0

    def __repr__(self):
        """Representación en texto de EstadoDiezMil.
        Ayuda a tener una versión legible del objeto.

        Returns:
            str: Representación en texto de EstadoDiezMil.
        """
        cant_dados = len(self.dados)

        return (cant_dados, self.puntaje_turno)  # representamos a cada estado como la cant de dados en ese estado y el puntaje acumulado
                                                 # porque una decision se toma en base a la cantidad de estados disponibles para tirar y 
                                                 # el puntaje del turno (me arriesgo o no dependiendo cuantos puntos tengo hasta ahora)
    

class AgenteQLearning:
    def __init__(
        self,
        ambiente: AmbienteDiezMil,
        estado: EstadoDiezMil,
        alpha: float,
        gamma: float,
        epsilon: float,
    ):
        """Definir las variables internas de un Agente que implementa el algoritmo de Q-Learning.

        Args:
            ambiente (AmbienteDiezMil): Ambiente con el que interactuará el agente.
            alpha (float): Tasa de aprendizaje.
            gamma (float): Factor de descuento.
            epsilon (float): Probabilidad de explorar.
        """
        self.ambiente = ambiente
        self.estado = estado 
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.qtable = dict()  # (estado, accion) -> Q(estado, accion)


    def elegir_accion(self):   
        """Selecciona una acción de acuerdo a una política ε-greedy.
        """
        numero_random = np.random.rand()

        # inicializamos Q(estado, accion) si no existe en la tabla

        if (self.estado, JUGADA_PLANTARSE) not in self.qtable:
            self.qtable[((self.estado, JUGADA_PLANTARSE))] = 0
        
        if (self.estado, JUGADA_TIRAR) not in self.qtable:
            self.qtable[((self.estado, JUGADA_TIRAR))] = 0
        
        # politica e-greedy
        if numero_random < self.epsilon: 
           accion = random.choice([JUGADA_PLANTARSE, JUGADA_TIRAR])

        else: # acción greedy (explotacion)
            _, accion = max([(self.estado, JUGADA_PLANTARSE), (self.estado, JUGADA_TIRAR)], key=self.qtable.get)

        return accion 

        

    def entrenar(self, episodios: int, verbose: bool = False) -> None:
        """Dada una cantidad de episodios, se repite el ciclo del algoritmo de Q-learning.
        Recomendación: usar tqdm para observar el progreso en los episodios.

        Args:
            episodios (int): Cantidad de episodios a iterar.
            verbose (bool, optional): Flag para hacer visible qué ocurre en cada paso. Defaults to False.
        """
        for episodio in tqdm(range(episodios), disable=not verbose):
            # reiniciamos el ambiente y obtenemos el estado inicial
            self.ambiente.reset()
            
            # mientras el turno no haya terminado
            while not self.estado.turno_terminado:
                # elegimos una acción según la política e-greedy
                accion = self.elegir_accion()
                
                # ejecutamos la acción en el ambiente y obtenemos recompensa y nuevo estado
                recompensa, turno_terminado = self.ambiente.step(accion)
                nuevo_estado = self.estado  # el ambiente ya actualiza el estado internamente
                
                # actualizamos Q(s, a) usando la fórmula de Q-learning
                clave_actual = (self.estado, accion)
                if turno_terminado:
                    valor_q_siguiente = 0
                else:
                    valor_q_siguiente = max(self.qtable.get((nuevo_estado, a), 0) for a in [JUGADA_PLANTARSE, JUGADA_TIRAR])
                
                self.qtable[clave_actual] = (1 - self.alpha) * self.qtable.get(clave_actual, 0) + \
                                            self.alpha * (recompensa + self.gamma * valor_q_siguiente)
                
                # pasamos al nuevo estado
                self.estado = nuevo_estado


    def guardar_politica(self, filename: str):
        """Almacena la política del agente en un formato conveniente.

        Args:
            filename (str): Nombre/Path del archivo a generar.
        """
        with open(filename, 'w') as file:
            for (estado, accion), valor_q in self.qtable.items():
                file.write(f"{estado},{accion},{valor_q}\n")


class JugadorEntrenado(Jugador):
    def __init__(self, nombre: str, filename_politica: str):
        self.nombre = nombre
        self.politica = self._leer_politica(filename_politica)
        
    def _leer_politica(self, filename: str, SEP: str = ','):
        """Carga una política entrenada con un agente de RL desde un archivo.

        Args:
            filename (str): Nombre/Path del archivo que contiene a una política almacenada.
            SEP (str): Separador utilizado en el archivo, por defecto ','.
        """
        politica = {}

        with open(filename, 'r') as file:
            for line in file:
                estado_str, accion_str, valor_q_str = line.strip().split(SEP)
                estado = eval(estado_str)  # convierte el estado a tupla
                accion = int(accion_str)
                valor_q = float(valor_q_str)

                politica[(estado, accion)] = valor_q

        return politica

    
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
        puntaje, no_usados = puntaje_y_no_usados(dados)
        
        if puntaje == 0:
            puntaje_turno = 0
        else:
            puntaje_turno += puntaje

        puntaje_total += puntaje_turno

        estado = EstadoDiezMil(no_usados, puntaje_total, puntaje_turno)
        jugada = self.politica[estado]
       
        if jugada==JUGADA_PLANTARSE:
            return (JUGADA_PLANTARSE, [])
        elif jugada==JUGADA_TIRAR:
            return (JUGADA_TIRAR, no_usados)


