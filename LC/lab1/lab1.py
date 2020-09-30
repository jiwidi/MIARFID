from collections import defaultdict


def ej1(st):
    r = {}
    # Contamos cuantas ocurrencias hay de cada categoria y almacenamos en un diccionario
    for subst in st.split(" "):
        word, category = subst.split("/")
        r[category] = r.get(category, 0) + 1
    return r


def ej2(st):
    r = {}
    cat_f = {}
    words_cats = defaultdict(dict)
    # Hacemos splits en substring de "{PALABRA} {CATEGORIA}"" y procesamos la substring para almacenar la informacion necesaria para generar el resultado en O(N)
    for subst in st.split(" "):
        word, category = subst.split("/")
        word = word.lower()
        r[word] = r.get(word, 0) + 1
        words_cats[word][category] = 1
        cat_f[(word, category)] = cat_f.get((word, category), 0) + 1
    for word in r.keys():
        r[word] = (
            f"{r[word]}  " + "".join([f"{category} {cat_f[(word,category)]} " for category in words_cats[word]]).strip()
        )
    return r


def ej3(st):
    # Hacemos splits en substrings de "{PALABRA} {CATEGORIA}"" y nos quedamos con una lista SOLO de categorias. Con esta calculamos los bigramas y su frecuencia
    cats = ["<S>"] + [subst.split("/")[1] for subst in st.split(" ")] + ["</S>"]
    r = {}
    prev_cat = cats[0]
    for i in range(1, len(cats)):
        r[(prev_cat, cats[i])] = r.get((prev_cat, cats[i]), 0) + 1
        prev_cat = cats[i]
    return r


def ej4(st, w):
    # Ejecutamos el ej1 y ej2 para usar los resultados en el calculo de probabilidades
    r1 = ej1(st)
    r2 = ej2(st)
    # Checkeamos que la palabra este contenida en la cadena
    if w not in r2.keys():
        print("Palabra desconocida")
        return
    # Calculamos las probabilidades de emision de la palabra para todas sus categorias
    # P(W|c)
    c_count = r2[w][0]
    info = r2[w].split(" ")[2:]  # Nos saltamos los 2 primeros ya que no corresponden a informacion de categorias
    info = zip(info[::2], info[1::2])  # Juntamos categoria y frecuencia en tuplas
    tmp = {}
    for ocurrence in info:
        c_tmp, n_tmp = ocurrence
        tmp[c_tmp] = n_tmp

    for c in tmp.keys():
        p_w_c = int(tmp[c]) / int(c_count)
        print(f"P( {c} | {w} )= {p_w_c}")

    # Calculamos las probabilidades lexicas de la palabra para todas sus categorias
    # P(C|W)
    for c in tmp.keys():
        p_c_w = int(tmp[c]) / int(r1[c])
        print(f"P( {c} | {w} )= {p_c_w}")


def main():
    st = "La/DT mamá/N de/P Pedro/N tiene/V tres/DNC tristes/Adj tigres/N que/C comen/V trigo/N en/P un/Pr triste/Adj trigal/N ./Fp ./Fp ./Fp Son/V tres/Pr que/C cantan/V en/P clave/N de/P la/N ./Fp La/DT mamá/N vino/V a/P beber/V vino/N ./Fp"
    print("Ej 1 \n------------------")
    r = ej1(st)
    rs = sorted(r.keys(), key=lambda x: x.lower())
    for key in rs:
        print(f"{key} {r[key]}")
    print("\nEj 2 \n------------------")
    r = ej2(st)
    rs = sorted(r.keys(), key=lambda x: x.lower())
    for key in rs:
        print(f"{key} {r[key]}")
    print("\nEj 3 \n------------------")
    r = ej3(st)
    for key in r:
        print(f"{key} {r[key]}")
    print("\nEj 4 \n------------------")
    ej4(st, "la")


if __name__ == "__main__":
    main()