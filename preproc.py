from nltk.tokenize import word_tokenize

fin = open("data/test1.tsv", "rt")
lines = fin.readlines()

preprocessed_lines = []

for line in lines:
    line = line.lower()

    try:
        sent1, sent2 = line.split("\t")

        tokens_sent1 = word_tokenize(sent1)[:1000]
        tokens_sent2 = word_tokenize(sent2)[:1000]

        sent1 = " ".join(tokens_sent1)
        sent2 = " ".join(tokens_sent2)

        preprocessed_line = sent1 + "\t" + sent2 + "\t" + sent2 + "\n"
        preprocessed_lines.append(preprocessed_line)
    except:
        pass


print("Writing %d lines ..." % (len(preprocessed_lines, )))
fout = open("data/pp_test1.tsv", "wt")
fout.writelines(preprocessed_lines)
fout.close()
