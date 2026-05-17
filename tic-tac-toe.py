from FSNeuralNetwork.neural_network import NeuralNetwork
from FSNeuralNetwork.startweight_utils import generate_random_weights
from FSNeuralNetwork.training_utils import train_NEpochs_alternately
from FSNeuralNetwork.activation_functions import activationType
import random

epochs_data = [
    {
        "X": [
            [1, 1, 0, -1, 0, -1, 0, 0, 0],
            [1, 0, -1, 1, 0, 0, 0, -1, 0],
            [1, -1, 0, 0, 1, -1, 0, 0, 0],
            [0, 0, 0, -1, -1, 0, 1, 1, 0],
            [-1, -1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, -1, 1, 0, -1, 1, 0, 0],
            [0, 0, -1, 0, -1, 0, 0, 0, 1],
            [-1, 0, 1, 0, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [
                -1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [0, 0, 0, 0, 1, 0, 0, 0, -1],
            [
                0,
                -1,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
            ],
        ],
        "Y": [
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
    },
    # Fokus: Vertikale Gewinne/Blocks + neue Eröffnungen (Reihenfolge gemischt)
    {
        "X": [
            [0, 0, -1, 0, 0, -1, 0, 0, 0],  # Gegner-Spalte rechts blockieren -> Index 8
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                -1,
                0,
            ],
            [1, 0, 0, 1, -1, 0, 0, 0, -1],  # Spalte links vollenden -> Index 6
            [0, -1, 0, 0, -1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 0],  # Spalte rechts vollenden -> Index 8
            [
                -1,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
            ],
            [0, 1, 1, -1, 0, -1, 0, 0, 0],
            [
                1,
                0,
                -1,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        ],
        "Y": [
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
        ],
    },
    # Fokus: Mittlere horizontale Reihen + Diagonale von rechts oben nach links unten
    {
        "X": [
            [0, 0, 1, 0, 1, 0, 0, -1, -1],  # Diagonale vollenden -> Index 6
            [0, 0, 0, 1, 1, 0, -1, 0, -1],  # Reihe Mitte vollenden -> Index 5
            [0, 0, -1, 0, -1, 0, 0, 0, 0],  # Diagonale blockieren -> Index 6
            [
                -1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],  # Gegner blockierte Ecke -> Mitte besetzen -> Index 4
            [0, 0, 0, -1, -1, 0, 0, 1, 1],  # Reihe Mitte blockieren -> Index 5
            [0, 0, 0, 0, 1, -1, 0, 0, 0],  # Zug erzwingen -> Ecke oben links -> Index 0
        ],
        "Y": [
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
    },
    # Fokus: Fortgeschrittenes Spiel (Gegner hat fast zwei Optionen, KI muss klug setzen)
    {
        "X": [
            [1, 0, -1, -1, 1, 0, 0, 0, 0],  # Diagonale vollenden -> Index 8
            [-1, 1, 0, 0, -1, 0, 1, 0, 0],  # Gegner-Diagonale blockieren -> Index 8
            [1, -1, 1, 0, 0, 0, -1, 0, 0],  # Verteidigung -> Mitte besetzen -> Index 4
            [-1, 0, 1, 1, -1, 0, 0, 0, 0],  # Reihe links blockieren -> Index 6
            [0, 0, 1, -1, -1, 0, 0, 0, 1],  # Reihe Mitte blockieren -> Index 5
        ],
        "Y": [
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
        ],
    },
    # Fokus: Finale Züge vor dem Unentschieden / Letzte Lücken füllen
    {
        "X": [
            [
                1,
                -1,
                1,
                -1,
                1,
                -1,
                0,
                1,
                -1,
            ],  # Letztes freies Feld füllen für Gewinn -> Index 6
            [
                -1,
                1,
                -1,
                1,
                0,
                1,
                1,
                -1,
                -1,
            ],  # Letztes freies Feld blockieren -> Index 4
            [1, -1, 0, -1, 1, 1, -1, 1, -1],  # Reihe oben vollenden -> Index 2
            [-1, 1, -1, 0, 1, -1, 1, -1, 1],  # Einziger valider Zug -> Index 3
        ],
        "Y": [
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
        ],
    },
]

epochs_data_2 = [
    # EPOCHE 1 - Fokus: Eigene Chancen nutzen auf den Achsen (1, 3, 5, 7)
    {
        "X": [
            [
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
            ],  # Eigene Spalte Mitte vollenden (über 1 und 4) -> Index 7
            [
                0,
                0,
                0,
                1,
                1,
                0,
                0,
                0,
                0,
            ],  # Eigene Reihe Mitte vollenden (über 3 und 4) -> Index 5
            [
                0,
                0,
                0,
                0,
                1,
                1,
                0,
                0,
                0,
            ],  # Eigene Reihe Mitte vollenden (über 4 und 5) -> Index 3
            [
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
            ],  # Eigene Spalte Mitte vollenden (über 4 und 7) -> Index 1
            [
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
            ],  # Vertikale Lücke in der Mitte füllen (1 und 7 besetzt) -> Index 4
            [
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
            ],  # Horizontale Lücke in der Mitte füllen (3 und 5 besetzt) -> Index 4
        ],
        "Y": [
            [0, 0, 0, 0, 0, 0, 0, 1, 0],  # Setze auf Index 7
            [0, 0, 0, 0, 0, 1, 0, 0, 0],  # Setze auf Index 5
            [0, 0, 0, 1, 0, 0, 0, 0, 0],  # Setze auf Index 3
            [0, 1, 0, 0, 0, 0, 0, 0, 0],  # Setze auf Index 1
            [0, 0, 0, 0, 1, 0, 0, 0, 0],  # Setze auf Index 4
            [0, 0, 0, 0, 1, 0, 0, 0, 0],  # Setze auf Index 4
        ],
    },
    # EPOCHE 2 - Fokus: Gegner blockieren auf den Achsen (1, 3, 5, 7)
    {
        "X": [
            [
                0,
                -1,
                0,
                0,
                -1,
                0,
                0,
                0,
                0,
            ],  # Gegner Spalte Mitte blockieren (über 1 und 4) -> Index 7
            [
                0,
                0,
                0,
                -1,
                -1,
                0,
                0,
                0,
                0,
            ],  # Gegner Reihe Mitte blockieren (über 3 und 4) -> Index 5
            [
                0,
                0,
                0,
                0,
                -1,
                -1,
                0,
                0,
                0,
            ],  # Gegner Reihe Mitte blockieren (über 4 und 5) -> Index 3
            [
                0,
                0,
                0,
                0,
                -1,
                0,
                0,
                -1,
                0,
            ],  # Gegner Spalte Mitte blockieren (über 4 und 7) -> Index 1
            [
                0,
                -1,
                0,
                0,
                0,
                0,
                0,
                -1,
                0,
            ],  # Gegner-Lücke in vertikaler Mitte blockieren -> Index 4
            [
                0,
                0,
                0,
                -1,
                0,
                -1,
                0,
                0,
                0,
            ],  # Gegner-Lücke in horizontaler Mitte blockieren -> Index 4
        ],
        "Y": [
            [0, 0, 0, 0, 0, 0, 0, 1, 0],  # Setze auf Index 7
            [0, 0, 0, 0, 0, 1, 0, 0, 0],  # Setze auf Index 5
            [0, 0, 0, 1, 0, 0, 0, 0, 0],  # Setze auf Index 3
            [0, 1, 0, 0, 0, 0, 0, 0, 0],  # Setze auf Index 1
            [0, 0, 0, 0, 1, 0, 0, 0, 0],  # Setze auf Index 4
            [0, 0, 0, 0, 1, 0, 0, 0, 0],  # Setze auf Index 4
        ],
    },
    # EPOCHE 3 - Fokus: Gemischte Szenarien (Angriff & Verteidigung kombiniert über Kanten)
    {
        "X": [
            [
                -1,
                1,
                0,
                0,
                1,
                0,
                0,
                0,
                -1,
            ],  # Eigene Spalte Mitte vollenden (Gegner blockiert Ecken) -> Index 7
            [0, 0, -1, 1, 1, 0, -1, 0, 0],  # Eigene Reihe Mitte vollenden -> Index 5
            [
                0,
                -1,
                0,
                0,
                -1,
                0,
                1,
                0,
                1,
            ],  # Gegner-Spalte Mitte blockieren (KI hält untere Ecken) -> Index 7
            [1, 0, 0, -1, -1, 0, 0, 0, 1],  # Gegner-Reihe Mitte blockieren -> Index 5
        ],
        "Y": [
            [0, 0, 0, 0, 0, 0, 0, 1, 0],  # Setze auf Index 7
            [0, 0, 0, 0, 0, 1, 0, 0, 0],  # Setze auf Index 5
            [0, 0, 0, 0, 0, 0, 0, 1, 0],  # Setze auf Index 7
            [0, 0, 0, 0, 0, 1, 0, 0, 0],  # Setze auf Index 5
        ],
    },
    # EPOCHE 4 - Fokus: Komplexe Kanten-Strukturen mit Zug-Erzwingung
    {
        "X": [
            [
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
            ],  # Gabelung vorbereiten über Kanten (1 und 3 besetzt) -> Index 4
            [
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
            ],  # Gabelung vorbereiten über Kanten (5 und 7 besetzt) -> Index 4
            [
                0,
                -1,
                0,
                -1,
                1,
                0,
                0,
                0,
                0,
            ],  # Gegner-Kanten-Angriff abwehren (Mitte halten) -> Index 0
            [
                0,
                0,
                0,
                0,
                1,
                -1,
                0,
                -1,
                0,
            ],  # Gegner-Kanten-Angriff abwehren (Mitte halten) -> Index 8
        ],
        "Y": [
            [0, 0, 0, 0, 1, 0, 0, 0, 0],  # Setze auf Index 4
            [0, 0, 0, 0, 1, 0, 0, 0, 0],  # Setze auf Index 4
            [1, 0, 0, 0, 0, 0, 0, 0, 0],  # Setze auf Index 0 (Ecke blocken)
            [0, 0, 0, 0, 0, 0, 0, 0, 1],  # Setze auf Index 8 (Ecke blocken)
        ],
    },
    # EPOCHE 5 - Fokus: Ausgeglichene Restfelder auf den Achsen füllen
    {
        "X": [
            [
                -1,
                1,
                1,
                1,
                -1,
                -1,
                -1,
                0,
                1,
            ],  # Letzte logische Kante füllen vor Unentschieden -> Index 7
            [
                1,
                -1,
                -1,
                0,
                1,
                1,
                1,
                1,
                -1,
            ],  # Letzte logische Kante füllen vor Unentschieden -> Index 3
            [
                -1,
                0,
                1,
                1,
                1,
                -1,
                -1,
                1,
                -1,
            ],  # Einzige Kante füllen, um Gegner-Sieg zu verhindern -> Index 1
        ],
        "Y": [
            [0, 0, 0, 0, 0, 0, 0, 1, 0],  # Setze auf Index 7
            [0, 0, 0, 1, 0, 0, 0, 0, 0],  # Setze auf Index 3
            [0, 1, 0, 0, 0, 0, 0, 0, 0],  # Setze auf Index 1
        ],
    },
]

epochs_data_3 = [
    {
        "X": [
            [0, -1, 0, 0, -1, 0, 0, 0, 1],
            [-1, -1, 0, 0, 1, 0, 0, 0, 1],
            [-1, 0, 0, 0, -1, 0, 0, 0, 1],
            [-1, 0, -1, 0, -1, 1, 1, -1, 1],
            [-1, 0, -1, 0, 1, 0, 0, 0, 1],
            [-1, 0, -1, -1, 1, 1, 1, 0, -1],
            [0, 0, 0, 0, 1, 0, 0, -1, -1],
            [-1, 0, 0, -1, 1, 1, 1, -1, -1],
            [0, 0, 0, -1, 1, 0, -1, 0, 1],
            [-1, 0, -1, 0, 1, 0, 0, 0, 1],
        ],
        "Y": [
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
        ],
    },
    {
        "X": [
            [-1, 0, -1, -1, 1, 1, 1, 0, -1],
            [-1, 0, 0, -1, 1, 0, 0, 0, 0],
            [0, -1, -1, 0, 1, 0, 0, 0, 0],
            [-1, 0, -1, 0, 1, -1, 0, 0, 1],
            [0, 0, 0, 0, 1, -1, 0, 0, -1],
            [0, 0, 0, 0, 1, 0, 0, -1, -1],
            [-1, 0, 0, 0, 1, 0, -1, -1, 1],
            [-1, 0, 0, 0, 1, 0, 0, 0, 0],
            [-1, 0, 0, 0, 1, 0, -1, 0, 1],
        ],
        "Y": [
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
        ],
    },
]


def initialize():
    inputs_count = 9
    neurons_hiddenlayer1 = 64
    neurons_hiddenlayer2 = 32
    outputs_count = 9

    startweigts_h1 = generate_random_weights(neurons_hiddenlayer1, inputs_count)
    startweigts_h2 = generate_random_weights(neurons_hiddenlayer2, neurons_hiddenlayer1)
    startweigts_out = generate_random_weights(outputs_count, neurons_hiddenlayer2)

    neuralNetwork = NeuralNetwork(
        inputs_count,
        outputs_count,
        output_activation_type=activationType.SOFTMAX,
    )

    neuralNetwork.add_hidden_layer(
        neurons_hiddenlayer1,
        weights=startweigts_h1,
        bias=-0.1,
        activation_type=activationType.LEAKY_RELU,
    )
    neuralNetwork.add_hidden_layer(
        neurons_hiddenlayer2,
        weights=startweigts_h2,
        bias=-0.1,
        activation_type=activationType.LEAKY_RELU,
    )

    neuralNetwork.outputLayer.set_weights(startweigts_out)

    print(neuralNetwork)

    neuralNetwork.save_neural_network()

    return neuralNetwork


def train():
    neuralNetwork = NeuralNetwork.load_neural_network()

    err = train_NEpochs_alternately(
        neuralNetwork,
        epochs_of_epochs=[epochs_data, epochs_data_2, epochs_data_3],
        repetitions=1000,
        learning_rate=0.01,
    )
    print(err)

    neuralNetwork.save_neural_network()


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


def ai_move(board, ai_model: NeuralNetwork):
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

    for index in sorted_indices:
        if board[index] == 0:  # Gültig, da das Feld frei ist
            return index

    return -1


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
    ai_model = NeuralNetwork.load_neural_network()

    play_game(ai_model)


# Spiel starten
if __name__ == "__main__":
    main()
