import numpy as np
from utils import puntaje_y_no_usados, JUGADA_PLANTARSE, JUGADA_TIRAR, JUGADAS_STR
from collections import defaultdict
from tqdm import tqdm
from jugador import Jugador
import random
from random import randint
import csv

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
        self.estado.actualizar_estado(0, 0, [1,2,3,4,5,6])
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
        

        if accion == JUGADA_PLANTARSE: 
            self.estado.fin_turno()
            self.turno_terminado = True  
            self.recompensa = 0
            self.cant_turnos += 1
            
        elif accion == JUGADA_TIRAR:

            dados = [randint(1, 6) for _ in range(len(self.estado.dados))]
            puntaje, no_usados = puntaje_y_no_usados(dados)
            
            if puntaje == 0: # si en esa jugada no se suma nada
                self.recompensa = - self.estado.puntaje_turno 
                self.estado.puntaje_turno = 0
                self.estado.fin_turno()
                self.cant_turnos += 1
              
            else:
                nuevo_puntaje_total = self.estado.puntaje_total + puntaje 
                nuevo_puntaje_turno = self.estado.puntaje_turno + puntaje 

                if len(no_usados) == 0:
                    self.estado.actualizar_estado(nuevo_puntaje_total, nuevo_puntaje_turno, [1,2,3,4,5,6]) # si usó todos los dados, puede volver a tirarlos 
                else:
                    self.estado.actualizar_estado(nuevo_puntaje_total, nuevo_puntaje_turno, no_usados)

                self.recompensa = self.estado.puntaje_turno


        # chequeamos que no haya llegado al tope de turnos o que haya ganado
        if self.cant_turnos == 1000 or self.estado.puntaje_total >= 10000:
            self.turno_terminado = True

        return (self.recompensa, self.turno_terminado)


class EstadoDiezMil:
    def __init__(self, dados, puntaje_total, puntaje_turno):
        """Definir qué hace a un estado de diez mil.
        Recordar que la complejidad del estado repercute en la complejidad de la tabla del agente de q-learning.
        """
        self.dados = dados 
        self.puntaje_total = puntaje_total 
        self.puntaje_turno = puntaje_turno
        self.cant_dados = len(dados)
        

    def actualizar_estado(self, nuevo_puntaje_total, nuevo_puntaje_turno, dados) -> None: 
      
        """Modifica las variables internas del estado luego de una tirada.
        Args:
            ... (_type_): _description_
            ... (_type_): _description_
        """
        self.dados = dados
        self.cant_dados = len(self.dados)
        self.puntaje_total = nuevo_puntaje_total
        self.puntaje_turno = nuevo_puntaje_turno

    
    def fin_turno(self):  
        """Modifica el estado al terminar el turno.
        """
        self.dados = [1, 2, 3, 4, 5, 6]
        self.puntaje_total += self.puntaje_turno
        self.puntaje_turno = 0
        self.cant_dados = len(self.dados)

    
    def crear_bins(self):
        if self.puntaje_turno == 0:
            return 0
        elif 1 <= self.puntaje_turno <= 100:
            return 1
        elif 101 <= self.puntaje_turno <= 200:
            return 2
        elif 201 <= self.puntaje_turno <= 300:
            return 3
        elif 301 <= self.puntaje_turno <= 400:
            return 4
        elif 401 <= self.puntaje_turno <= 500:
            return 5
        elif 501 <= self.puntaje_turno <= 600:
            return 6
        elif 601 <= self.puntaje_turno <= 1000:
            return 7
        elif 1001 <= self.puntaje_turno <= 3000:
            return 8
        elif 3001 <= self.puntaje_turno <= 6501:
            return 9
        else:
            return 10

    def __str__(self):
        """Representación en texto de EstadoDiezMil.
        Ayuda a tener una versión legible del objeto.

        Returns:
            str: Representación en texto de EstadoDiezMil.
        """
        cant_dados = self.cant_dados
        bin_puntaje = self.crear_bins()

        return f"Cantidad de dados:{cant_dados}, Bin: {bin_puntaje}"
                                                # representamos a cada estado como la cant de dados en ese estado y el puntaje acumulado
                                                # porque una decision se toma en base a la cantidad de estados disponibles para tirar y 
                                                # el bin del puntaje del turno (me arriesgo o no dependiendo cuantos puntos tengo hasta ahora)
    

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
        self.qtable = self.inicializar_qtable()  # {estado: {JUGADA_PLANTARSE: Q(estado, JUGADA_PLANTARSE), JUGADA_TIRAR: Q(estado, JUGADA_TIRAR)}}
        
    def inicializar_qtable(self):
        qtable = {}
        
        bins_puntaje = range(11)  # 11 bins posibles (0 a 10)
        cant_dados_posibles = range(1,7)  # entre 1 y 6 dados disponibles

        # se recorren todas las combinaciones posibles de bin de puntaje y cantidad de dados
        for bin_puntaje in bins_puntaje:
            for cant_dados in cant_dados_posibles:
                
                estado = (cant_dados, bin_puntaje)
                qtable[estado] = {accion: 0 for accion in [JUGADA_PLANTARSE, JUGADA_TIRAR]}
        
        return qtable
    
    
    def elegir_accion(self):   
        """Selecciona una acción de acuerdo a una política ε-greedy.
        """
        numero_random = np.random.rand()

        # politica e-greedy

        tupla_estado = (self.estado.cant_dados, self.estado.crear_bins())

        if numero_random < self.epsilon: # exploracion
            accion = random.choice([JUGADA_PLANTARSE, JUGADA_TIRAR])

        else: # acción greedy (explotacion)
            if self.qtable[tupla_estado][JUGADA_PLANTARSE] == self.qtable[tupla_estado][JUGADA_TIRAR]:
                accion = random.choice([JUGADA_PLANTARSE, JUGADA_TIRAR])
            else:
                accion = max(self.qtable[tupla_estado], key=self.qtable[tupla_estado].get)
            
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
            while not self.ambiente.turno_terminado:
                estado_actual = (self.estado.cant_dados, self.estado.crear_bins())
                # elegimos una acción según la política e-greedy
                accion = self.elegir_accion()
                
                # ejecutamos la acción en el ambiente y obtenemos recompensa y nuevo estado
                recompensa, turno_terminado = self.ambiente.step(accion)

                estado_siguiente = (self.estado.cant_dados, self.estado.crear_bins())
                
                # actualizamos Q(s, a) usando la fórmula de Q-learning  
                if turno_terminado:
                    valor_q_siguiente = 0
                else:
                    valor_q_siguiente = max(self.qtable[estado_siguiente], key=self.qtable[estado_siguiente].get)
                
                self.qtable[estado_actual][accion] += self.alpha * (recompensa + self.gamma * valor_q_siguiente - self.qtable[estado_actual][accion])
                

    def guardar_politica(self, filename: str):
        """Almacena la política del agente en un formato conveniente.

        Args:
            filename (str): Nombre/Path del archivo a generar.
        """
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Estado', 'Accion_Optima'])  

            for estado, acciones in self.qtable.items():
                # obtenemos la acción con el valor Q más alto
                accion_optima = max(acciones, key=acciones.get)
                writer.writerow([estado, accion_optima])
            
  

class JugadorEntrenado(Jugador):
    def __init__(self, nombre: str, filename_politica: str):
        self.nombre = nombre
        self.politica = self._leer_politica(filename_politica)
        
    def _leer_politica(self, filename: str, SEP: str = ','):
        """Carga una política entrenada con un agente de RL desde un archivo."""
        politica = {}

        with open(filename, mode='r') as file:
            reader = csv.reader(file, delimiter=SEP)
            next(reader)  
                
            for row in reader:
                estado = eval(row[0])  
                accion_optima = int(row[1])
                
                # Asignar la acción óptima al estado en el diccionario
                politica[estado] = accion_optima  
    
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

        if len(no_usados) == 0:
            estado = EstadoDiezMil([1,2,3,4,5,6], puntaje_total, puntaje_turno)
        else:
            estado = EstadoDiezMil(no_usados, puntaje_total, puntaje_turno)

        estado_tupla = (estado.cant_dados, estado.crear_bins())
    
        jugada = self.politica[estado_tupla]
    
        if jugada==JUGADA_PLANTARSE:
            return (JUGADA_PLANTARSE, [])
        elif jugada==JUGADA_TIRAR:
            return (JUGADA_TIRAR, no_usados)
        

    
        


