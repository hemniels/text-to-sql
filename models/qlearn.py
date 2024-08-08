# Pseudocode für das QLearningAgent-Modell
class QLearningAgent:
    def __init__(state_size, action_size, learning_rate=0.01, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Konstruktor für das Q-Learning Modell:
        - state_size: Anzahl der möglichen Zustände.
        - action_size: Anzahl der möglichen Aktionen.
        - learning_rate: Lernrate für die Q-Tabellen-Updates.
        - gamma: Discount-Faktor, der die Bedeutung zukünftiger Belohnungen bestimmt.
        - epsilon: Startwert für die Explorationsrate.
        - epsilon_decay: Rate, mit der epsilon nach jedem Schritt verringert wird.
        - epsilon_min: Mindestwert für epsilon.
        """
        self.q_table = np.zeros((state_size, action_size))  # Q-Tabelle initialisieren.
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.learning_rate = learning_rate
    
    def choose_action(state):
        """
        Aktion auswählen basierend auf dem aktuellen Zustand:
        - state: Der aktuelle Zustand (z.B. die aktuelle SQL-Abfrage).
        - Rückgabe: Die ausgewählte Aktion (z.B. eine Modifikation der SQL-Abfrage).
        """
        if np.random.rand() <= self.epsilon:
            return np.random.choice(action_size)  # Zufällige Aktion (Exploration).
        return np.argmax(self.q_table[state])  # Beste bekannte Aktion (Exploitation).
    
    def learn(state, action, reward, next_state):
        """
        Lernen aus der Erfahrung:
        - state: Der aktuelle Zustand.
        - action: Die gewählte Aktion.
        - reward: Erhaltene Belohnung nach der Aktion.
        - next_state: Der neue Zustand nach der Aktion.
        """
        best_next_action = np.argmax(self.q_table[next_state])
        target = reward + self.gamma * self.q_table[next_state, best_next_action]
        self.q_table[state, action] += self.learning_rate * (target - self.q_table[state, action])
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(filepath):
        """
        Speichern des Modells (Q-Tabelle).
        """
        np.save(filepath, self.q_table)
    
    def load_model(filepath):
        """
        Laden des Modells (Q-Tabelle).
        """
        self.q_table = np.load(filepath)
