from FSNeuralNetwork.neural_network import NeuralNetwork
from FSNeuralNetwork.startweight_utils import generate_random_weights
from FSNeuralNetwork.training_utils import train_NEpochs_alternately
from FSNeuralNetwork.activation_functions import activationType
from FSNeuralNetwork.loss_functions import LossType
import random
import json


def initialize():
    inputs_count = 9
    neurons_hiddenlayer1 = 24
    neurons_hiddenlayer2 = 12
    outputs_count = 9

    startweigts_h1 = generate_random_weights(neurons_hiddenlayer1, inputs_count)
    startweigts_h2 = generate_random_weights(neurons_hiddenlayer2, neurons_hiddenlayer1)
    startweigts_out = generate_random_weights(outputs_count, neurons_hiddenlayer2)

    neuralNetwork = NeuralNetwork(
        inputs_count,
        outputs_count,
        output_bias=0,
        output_activation_type=activationType.SOFTMAX,
        loss_type=LossType.CATEGORICAL_CROSS_ENTROPY,
    )

    neuralNetwork.add_hidden_layer(
        neurons_hiddenlayer1,
        weights=startweigts_h1,
        bias=0,
        activation_type=activationType.LEAKY_RELU,
    )
    neuralNetwork.add_hidden_layer(
        neurons_hiddenlayer2,
        weights=startweigts_h2,
        bias=0,
        activation_type=activationType.LEAKY_RELU,
    )

    neuralNetwork.outputLayer.set_weights(startweigts_out)

    print(neuralNetwork)

    neuralNetwork.save_neural_network("tictactoesetup24x12.json")

    return neuralNetwork


def train():
    neuralNetwork = NeuralNetwork.load_neural_network("tictactoesetup24x12.json")
    try:
        with open("tictactoe_training_data.json", "r", encoding="utf-8") as f:
            data = json.load(f)

            train_NEpochs_alternately(
                neuralNetwork=neuralNetwork, data=data, epochs=150
            )

            neuralNetwork.save_neural_network("tictactoetrained24x12.json")
    except FileNotFoundError:
        raise FileNotFoundError("Json datei wurde nicht gefunden")


def print_board(board):
    """Gibt das Spielfeld formatiert im Terminal aus."""
    symbols = {0: " ", 1: "X", -1: "O"}  # 1 = KI (X), -1 = Mensch (O)
    print("\n")
    for i in range(0, 9, 3):
        print(f" {symbols[board[i]]} | {symbols[board[i+1]]} | {symbols[board[i+2]]} ")
        if i < 6:
            print("---+---+---")
    print("\n")


def check_winner(board):
    """Prüft, ob jemand gewonnen hat oder ein Unentschieden vorliegt.

    Gibt 1 (KI gewinnt), -1 (Mensch gewinnt), 0 (Unentschieden) oder None (läuft noch) zurück.
    """
    win_lines = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],  # Horizontale Reihen
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],  # Vertikale Spalten
        [0, 4, 8],
        [2, 4, 6],  # Diagonalen
    ]

    for line in win_lines:
        if board[line[0]] == board[line[1]] == board[line[2]] != 0:
            return board[line[0]]  # Gibt 1 oder -1 zurück

    if 0 not in board:
        return 0  # Unentschieden, da kein Feld mehr frei ist

    return None  # Spiel läuft noch


def format_for_ai(board):
    """Bringt das Board in das exakte Format für Ihr Netzwerk."""
    # Ihr Netzwerk erwartet: 1 = KI, -1 = Gegner, 0 = Frei.
    # Da das interne 'board' genau so definiert ist, reicht eine Kopie der Liste.
    return list(board)


def ai_move(board, ai_model: NeuralNetwork) -> int:
    """Berechnet den KI-Zug basierend auf den Ausgaben Ihres Netzwerks."""
    if ai_model is None:
        # Fallback, falls Ihr Modell noch nicht eingefügt wurde
        print("Kein KI Modell vorhanden")
        free_indices = [i for i, x in enumerate(board) if x == 0]
        return random.choice(free_indices)

    # 1. Board ins richtige Format bringen
    inputs = format_for_ai(board)

    # 2. Vorwärtsschritt durch Ihr eigenes Netzwerk (erwartet 9 Softmax-Werte)
    # Nutzen Sie hier den exakten Methodennamen Ihrer Implementierung (z.B. predict, forward...)
    output_probabilities = ai_model.calculate_data(inputs)

    # 3. Den besten VALIDEN Zug herausfiltern
    # Wir sortieren die Indizes (0-8) absteigend nach ihrer Wahrscheinlichkeit
    sorted_indices = sorted(
        range(9), key=lambda k: output_probabilities[k], reverse=True
    )
    print(sorted_indices)
    sorted_indices_with_k = []
    for i in range(len(sorted_indices)):
        sorted_indices_with_k.append(
            [sorted_indices[i], output_probabilities[sorted_indices[i]]]
        )

    print(sorted_indices_with_k)

    valid_sorted_indices_with_k = []

    for index, k in sorted_indices_with_k:
        if board[index] == 0:  # Gültig, da das Feld frei ist
            valid_sorted_indices_with_k.append([index, k])

    print(valid_sorted_indices_with_k)

    if not (valid_sorted_indices_with_k):
        return -1

    best_idx, best_k = valid_sorted_indices_with_k[0]
    second_idx, second_k = valid_sorted_indices_with_k[1]

    diff = best_k - second_k
    if diff > 0.4:
        print("KI ist sich zeimlich sicher")
        return best_idx
    else:
        print("KI ist sich unsicher")
        return random.choices([best_idx, second_idx], weights=[best_k, second_k])[0]


def play_game(ai_model):
    """Die Hauptschleife für das Terminal-Spiel."""
    # Internes Board: 0 = frei, 1 = KI (X), -1 = Mensch (O)
    board = [0] * 9

    print("=== Willkommen zu TicTacToe gegen deine eigene KI ===")
    print("Spielfeld-Indizes sind wie folgt aufgeteilt:")
    print(" 0 | 1 | 2 \n---+---+---\n 3 | 4 | 5 \n---+---+---\n 6 | 7 | 8 ")

    # Random-Funktion entscheidet, wer anfängt
    current_player = random.choice([1, -1])

    if current_player == 1:
        print("\n🎲 Die Auslosung ergab: Die KI (X) beginnt!")
    else:
        print("\n🎲 Die Auslosung ergab: Du (O) beginnst!")

    while check_winner(board) is None:
        print_board(board)
        print(board)

        if current_player == -1:
            # --- MENSCH IST DRAN ---
            valid_move = False
            while not valid_move:
                try:
                    move = int(
                        input("Dein Zug (Wähle einen freien Index von 0 bis 8): ")
                    )
                    if board[move] == 0:
                        board[move] = -1
                        valid_move = True
                    else:
                        print("Dieses Feld ist bereits besetzt!")
                except (ValueError, IndexError):
                    print("Ungültige Eingabe. Bitte eine Zahl von 0 bis 8 eingeben.")
            current_player = 1  # Wechsel zur KI
        else:
            # --- KI IST DRAN ---
            print("KI berechnet Zug...")
            move = ai_move(board, ai_model)
            board[move] = 1
            print(f"KI setzt auf Feld: {move}")
            current_player = -1  # Wechsel zum Menschen

    # Spiel vorbei: Ergebnis anzeigen
    print_board(board)
    result = check_winner(board)

    if result == 1:
        print("🤖 Die KI hat gewonnen! Dein SGD-Training war erfolgreich.")
    elif result == -1:
        print("🎉 Du hast gewonnen! Die KI braucht wohl noch ein paar Epochen.")
    else:
        print("🤝 Unentschieden! Ein perfektes Spiel von beiden Seiten.")


def main():
    ai_model = NeuralNetwork.load_neural_network("tictactoetrained24x12.json")

    play_game(ai_model)
    # initialize()
    # train()


# Spiel starten
if __name__ == "__main__":
    main()
