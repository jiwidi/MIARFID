import re


# definir regexs
reHora = [
    "([0-9]+:[0-9]+)",
]
reWeb = [
    "(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})",
]
reUsuario_Hashtag = [
    "(@[a-zA-Z0-9_]+)",
    "(#[a-zA-Z0-9_-]+)",
]
reEmail = ["([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)"]
reAcro = ["([A-Z].\.[A-Z]+\.|(?:[a-zA-Z]\.){2,}|[A-Z]\.)"]
reNumeros = ["([0-9]+[-/,.][0-9]+)"]
reSignos = [
    '(\.{3}|[().,"?¿!¡…;:%])',
]
reFecha = [
    "(\d{1,2})(\s+)(\w{2})(\s+)(\w+)(\s+)(\w{2})(\s+)(\d{2,4})",
    "([0-9]+[-/][0-9]+[-/][0-9]+)",
]
reEmojis = ["[^\w\s,]"]
rePalabras = [
    "([a-zA-Z]+-\w+)",
    "(\w+)",
]

master_regex = (
    reFecha
    + reHora
    + reNumeros
    + reSignos
    + reWeb
    + reEmail
    + reUsuario_Hashtag
    + reAcro
    + reEmojis
    + rePalabras
)
master_regex = "|".join(master_regex)


def tokenize(sentences):
    tokenizer = re.compile(master_regex)
    output = ""
    for sentence in sentences:
        output += sentence.replace("\n", "") + "\n"  # Some lines are missing \n
        res = tokenizer.finditer(sentence)
        for i in res:
            output += "\n" + str(i.group())
        output += "\n"
    return output


if __name__ == "__main__":
    # Leemos los datos de entrada
    with open("entrada_tokenizador.txt") as f:
        data = f.readlines()

    # Guardamos los datos procesados
    result = tokenize(data)
    with open("salida_tokenizador.txt", "w+") as f:
        f.write(result)

    # Comparamos nuestros datos procesados contra la salida de test que nos dio el profesor
    with open("salida_test.txt") as f:
        test = f.readlines()

    assert "".join(test) == result

