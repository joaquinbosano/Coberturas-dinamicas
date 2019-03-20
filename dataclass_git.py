import quandl
import pandas as pd

class Dataholder:



    def __init__(self, subyacente, tiempo_inicio, tiempo_terminal, api_key, cantidad_contratos = 1):

        # mi api =  "oXtqxBB1YNCLvfnoLyea"
        self.__api_key = str(api_key)
        self.__subyacente = subyacente
        self.__cantidad_contratos = cantidad_contratos
        self.__tiempo_inicio = tiempo_inicio
        self.__tiempo_terminal = tiempo_terminal
        self.__dict_aceptados = {"crudo": "CHRIS/ICE_T", "cobre": "CHRIS/CME_HG", "oro": "CHRIS/CME_GC", "soja":"CHRIS/ICE_IS"}


    def Crear(self):
        quandl.ApiConfig.api_key = self.__api_key
        lista_convencional = []
        lista_settle = []
        lista_last = []
        iterador = 1

        if self.__subyacente in list(self.__dict_aceptados.keys()):

            if self.__subyacente == list(self.__dict_aceptados.keys())[0]:
                lista_convencional = ["EIA/PET_RWTC_D"]
                while iterador < self.__cantidad_contratos + 1:

                    nombre_convencional = str(self.__dict_aceptados[self.__subyacente]) + str(iterador)
                    lista_convencional.append(nombre_convencional)
                    lista_settle.append(nombre_convencional + " - Settle")
                    iterador += 1

                placeholder = quandl.get(lista_convencional , start_date = self.__tiempo_inicio, end_date = self.__tiempo_terminal)
                return placeholder.loc[:,lista_settle].dropna(axis = 0, how = "any")

            else:
                while iterador < self.__cantidad_contratos + 1:
                    nombre_convencional = str(self.__dict_aceptados[self.__subyacente]) + str(iterador)
                    lista_convencional.append(nombre_convencional)
                    lista_settle.append(nombre_convencional +" - Settle" )
                    lista_last.append(nombre_convencional + " - Last")
                    iterador += 1

                placeholder = quandl.get(lista_convencional, start_date = self.__tiempo_inicio, end_date = self.__tiempo_terminal)
                try:
                    return placeholder.loc[:,lista_settle].dropna(axis = 0, how = "any")
                except:
                    return placeholder.loc[:,lista_last].dropna(axis = 0, how = "any")


    def Cambiar_Diccionario (self, claves, codigos):

        if type(claves) is type([1,2]):

            if type (codigos) is type([1,2]):

                newdict = dict(zip([claves],[codigos]))

                self.__dict_aceptados.update(newdict)
        else:

            self.__dict_aceptados.update({str(claves):str(codigos)})


    def ver(self):

        return self.__dict_aceptados
