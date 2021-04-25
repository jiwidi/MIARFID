import re
import sys

"""
******************************************
**    Práctica 1 ALC                    **
**    Autor: Javier Martínez Bernia     **
**    2021 @ MIARFID, UPV               **
******************************************
"""


if (len(sys.argv) != 3):
    print("Usage: python tokenizador.py <Fichero_Entrada> <Fichero_Salida>")
    quit()
else:
    input_file = sys.argv[1]
    output_file = sys.argv[2]


""" 
************************************
EXPRESIONES REGULARES
************************************
"""

# Fecha en formato "12 de Mayo de 2020"
re_date = re.compile(r'(\d{1,2}\s+\w{2}\s+\w+\s+\w{2,3}\s+\d{2,4})')

# Fecha en formato sin espacios "12-05-2020, 12/05, 12/05/2020 ..."
re_date_2 = re.compile(r'(\d{1,2}/\d{1,2}(/\d{2,4})?) | (\d{1,2}-\d{1,2}(-\d{2,4})?)', re.X)

# Números Decimales
re_decimal = re.compile(r'\d+[\.,]\d+')

# Horas
re_horas = re.compile(r'\d{1,2}:\d{2}')

# Direcciones Web y correos electrónicos
re_web = re.compile(r'(https?://(w{3}.)?\S*) | ([^@]+@[^@]+\.[^@]+)', re.X)

# Usuarios y hashtags (@fulanito #Lunes)
re_twitter = re.compile(r'(@[^@]+)|(#[^#]+)')

# Acronimos (EE.UU.)
re_acronimos = re.compile(r'([A-Z]{1,2}\.)+')

# Simbolos no alfanumericos "().'"?¿!¡...;:"
re_guion = re.compile(r'\w+-\w+')
re_simbolos = re.compile(r'(\W*)(\w+)(\W*)')
re_trespuntos = re.compile(r'(\W*)(\.\.\.)(\W*)')



""" 
************************************
PROCESADO DE LA ENTRADA
************************************
"""

res = ""

with open(input_file, "r") as f:
    # Procesamos cada linea
    for line in f.readlines():
        res += line + '\n'

        # Primero detectamos las fechas y las cambiamos por un token especial "TOKENFECHA"
        fechas = re_date.findall(line)
        line = re_date.sub("TOKENFECHA", line)

        # Luego separamos la cadena en espacios y detectamos todos los demas patrones
        for word in line.split():
            if word == 'TOKENFECHA':
                res += fechas.pop(0) + '\n'
            #
            elif re_decimal.match(word):
                res += re_decimal.match(word)[0] + '\n'
                for char in word:
                    if char not in re_decimal.match(word)[0]:
                        res += char + '\n'
            #
            elif re_date_2.match(word):
                res += re_date_2.match(word)[0] + '\n'
                for char in word:
                    if char not in re_date_2.match(word)[0]:
                        res += char + '\n'
            #
            elif re_horas.match(word):
                res += re_horas.match(word)[0] + '\n'
                for char in word:
                    if char not in re_horas.match(word)[0]:
                        res += char + '\n'
            #
            elif re_web.match(word):
                res += word + '\n'
            #
            elif re_twitter.match(word):
                res += word + '\n'
            #
            elif re_acronimos.match(word):
                res += re_acronimos.match(word)[0] + '\n'
                for char in word:
                    if char not in re_acronimos.match(word)[0]:
                        res += char + '\n'
            #            
            elif re_guion.match(word):
                res += re_guion.match(word)[0] + '\n'
                for char in word:
                    if char not in re_guion.match(word)[0]:
                        res += char + '\n'
            #
            else:
                simbolos = re_simbolos.findall(word)
                if len(simbolos) > 0:
                    for item in simbolos:
                        if len(item[0]) > 0:
                            # Separar cualquier simbolo excepto los tres puntos "..."
                            trespuntos = re_trespuntos.match(item[0])
                            if trespuntos:
                                #
                                if len(trespuntos.group(1)) > 0:
                                    for char in trespuntos.group(1): res += char + '\n'
                                #
                                res += trespuntos.group(2) + '\n'
                                #
                                if len(trespuntos.group(3)) > 0:
                                    for char in trespuntos.group(3): res += char + '\n'
                                #
                            else:
                                for char in item[0]: res += char + '\n'

                        
                        res += item[1] + '\n'
                        
                        if len(item[2]) > 0:
                            # Separar cualquier simbolo excepto los tres puntos "..."
                            trespuntos = re_trespuntos.match(item[2])
                            if trespuntos:
                                #
                                if len(trespuntos.group(1)) > 0:
                                    for char in trespuntos.group(1): res += char + '\n'
                                #
                                res += trespuntos.group(2) + '\n'
                                #
                                if len(trespuntos.group(3)) > 0:
                                    for char in trespuntos.group(3): res += char + '\n'
                                #
                            else:
                                for char in item[2]: res += char + '\n'

res += '\n'


with open(output_file, 'w') as out:
    out.write(res)