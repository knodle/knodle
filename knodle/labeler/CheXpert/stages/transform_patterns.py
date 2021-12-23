def transform(text_file):

    file = open(text_file, "r+")

    new_file = []

    for line in file:
        lemmatized1 = "{} < {} {lemma:/" + str(line).rstrip() + "/}"
        new_file.append(lemmatized1)

        lemmatized2 = "{} > {} {lemma:/" + str(line).rstrip() + "/}"
        new_file.append(lemmatized2)

    with open(text_file, "w+") as file:
        for i in new_file:
            file.write(i + "\n")
