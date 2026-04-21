def create_read_vector(sequence, L, Delta):
    # pad with 'N'
    sequenceStr = ToDNAString(sequence)
    sequenceStr = "N" * (L - 1) + sequenceStr + "N" * (L - 1)

    vector = []

    for i in range(L - 1, len(sequenceStr), Delta):
        upper = i
        lower = i - (L - 1)

        comp = {'A': 0, 'C': 0, 'G': 0, 'T': 0}

        for j in range(lower, upper + 1):
            base = sequenceStr[j]
            if base in comp:
                comp[base] += 1

        for j in comp:
            for x in range(comp[j]):
                    if(j == "A"):
                        vector.append([1,0,0,0])
                    elif(j == "C"):
                        vector.append([0,1,0,0])
                    elif (j == "G"):
                        vector.append([0, 0, 1, 0])
                    elif (j == "T"):
                        vector.append([0, 0, 0, 1])

    return vector

def ToDNAString(sequence):
    DNAString = ""
    for i in sequence:
        if i == [1,0,0,0]:
            DNAString += "A"
        if i == [0,1,0,0]:
            DNAString += "C"
        if i == [0, 0, 1, 0]:
            DNAString += "G"
        if i == [0, 0, 0, 1]:
            DNAString += "T"

    return DNAString