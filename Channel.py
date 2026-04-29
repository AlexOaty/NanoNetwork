from sympy import Sum
from sympy.strategies.core import switch
import numpy as np

L = 4
Delta = 1

def create_read_vector(sequence, L, Delta):
    # pad with 'N'
    #sequenceStr = ToDNAString(sequence)
    sequenceStr = "N" * (L - 1) + sequence

    mapping = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
        "N": [0, 0, 0, 0]  # Assuming N should contribute nothing to the sum
    }

    temp_list = [mapping.get(base, [0, 0, 0, 0]) for base in sequenceStr]
    seqArr = np.array(temp_list)  # This is now a (Length x 4) matrix

    vector = []
    for i in range(L - 1, len(sequenceStr), Delta):
        upper = i
        lower = i - (L - 1)

        window_sum = seqArr[lower:upper+1].sum(axis=0).tolist()
        vector.append(window_sum)

    return vector

def matcher(arr, reconstruction, L):
    match arr:
        case [1, 0, 0, 0]:
            reconstruction.append("A")
        case [0, 1, 0, 0]:
            reconstruction.append("C")
        case [0, 0, 1, 0]:
            reconstruction.append("G")
        case [0, 0, 0, 1]:
            reconstruction.append("T")
        case [0, 0, 0, 0]:
            reconstruction.append(reconstruction[len(reconstruction)-L])
    return reconstruction

def reconstruct(vector, L):
    reconstruction = []
    reconstruction = matcher(vector[0], reconstruction, L)

    for i in range(1, len(vector)):
        curr = np.array(vector[i])
        last = np.array(vector[i-1])
        C = (curr-last)
        C[C < 0] = 0
        reconstruction = matcher(C.tolist(), reconstruction, L)

    return "".join(reconstruction)


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

Nsequence = input("Enter Sequence: ")
readVector = create_read_vector(Nsequence, L, Delta)
print(readVector)
recon = reconstruct(readVector, L)
print(recon)
