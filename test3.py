from FSNeuralNetwork.startweight_utils import *

weights1 = generate_random_weights(20, 20)
weights2 = generate_centerheavy_weights(20, 20)
weights3 = generate_edgeheavy_weights(20, 20)


print(f"\nrandom: \n{weights1}\n")
print(len(weights1))
print(len(weights1[0]))
print(f"\ncenterheavy: \n{weights2}\n")
print(len(weights2))
print(len(weights2[0]))
print(f"\nedgeheavy: \n{weights3}\n")
print(len(weights3))
print(len(weights3[0]))
