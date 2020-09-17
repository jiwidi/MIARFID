from collections import defaultdict


def ej1(st):
    r = {}
    for subst in st.split(" "):
        word, category = subst.split("/")
        r[category] = r.get(category, 0) + 1
    return r


def ej2(st):
    r = {}
    cat_f = {}
    words_cats = defaultdict(dict)
    for subst in st.split(" "):
        word, category = subst.split("/")
        word = word.lower()
        r[word] = r.get(word, 0) + 1
        words_cats[word][category] = 1
        cat_f[(word, category)] = cat_f.get((word, category), 0) + 1
    for word in r.keys():
        r[word] = (
            f"{r[word]}  "
            + "".join([f"{category} {cat_f[(word,category)]}  " for category in words_cats[word]]).strip()
        )
    return r


def ej3(st):
    cats = [subst.split("/")[1] for subst in st.split(" ")]
    r = {}
    prev_cat = cats[0]
    for i in range(1, len(cats)):
        r[(prev_cat, cats[i])] = r.get((prev_cat, cats[i]), 0) + 1
        prev_cat = cats[i]
    return r


def ej4(st, w, c):
    r1 = ej1(st)
    r2 = ej2(st)
    if w not in r2.keys():
        print("Palabra desconocida")
        return
    # P(W|c)
    c_count = r2[w][0]
    info = r2[w][3:].split("  ")
    tmp = {}
    print(info)
    for ocurrence in info:
        c_tmp, n_tmp = ocurrence.split(" ")
        tmp[c_tmp] = n_tmp
    p_w_c = int(tmp[c]) / int(c_count)

    # P(C|W)
    p_c_w = int(tmp[c]) / int(r1[c])

    return p_w_c, p_c_w


def main():
    st = "El/DT perro/N come/V carne/N de/P la/DT carnicería/N y/C de/P la/DT nevera/N y/C canta/V el/DT la/N la/N la/N ./Fp"
    print("Ej 1 \n------------------")
    r = ej1(st)
    rs = sorted(r.keys(), key=lambda x: x.lower())
    for key in rs:
        print(f"{key} {r[key]}")
    print("Ej 2 \n------------------")
    r = ej2(st)
    rs = sorted(r.keys(), key=lambda x: x.lower())
    for key in rs:
        print(f"{key} {r[key]}")
    print("Ej 3 \n------------------")
    r = ej3(st)
    for key in r:
        print(f"{key} {r[key]}")

    print(ej4(st, "la", "DT"))


if __name__ == "__main__":
    main()