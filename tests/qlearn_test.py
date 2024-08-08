import numpy as np
import pytest

# Importieren Sie das Q-Learning-Modell hier
from models import QLearningAgent

# Testdaten
state_size = 10
action_size = 5

@pytest.fixture
def setup_q_learning_agent():
    agent = QLearningAgent(state_size, action_size)
    return agent

def test_choose_action_exploration(setup_q_learning_agent):
    agent = setup_q_learning_agent
    # Bei hoher Epsilon sollte zufällig gewählt werden
    agent.epsilon = 1.0
    action = agent.choose_action(0)
    assert action >= 0 and action < action_size

def test_choose_action_exploitation(setup_q_learning_agent):
    agent = setup_q_learning_agent
    # Füllen Sie die Q-Tabelle mit Werten und setzen Sie Epsilon auf 0
    agent.q_table[:, :] = 0.0
    agent.q_table[0, 1] = 1.0  # Setzen Sie einen hohen Wert für die Aktion 1
    agent.epsilon = 0.0
    action = agent.choose_action(0)
    assert action == 1

def test_learn_update_q_table(setup_q_learning_agent):
    agent = setup_q_learning_agent
    state = 0
    action = 1
    reward = 1.0
    next_state = 2
    agent.q_table[state, action] = 0.0  # Initialwert der Q-Tabelle
    agent.learn(state, action, reward, next_state)
    # Berechnen Sie den erwarteten Q-Wert
    expected_q_value = reward + agent.gamma * np.max(agent.q_table[next_state])
    assert np.isclose(agent.q_table[state, action], expected_q_value, atol=1e-2)

def test_save_load_model(setup_q_learning_agent):
    agent = setup_q_learning_agent
    agent.q_table[0, 1] = 1.0
    agent.save_model('test_q_table.npy')
    
    new_agent = QLearningAgent(state_size, action_size)
    new_agent.load_model('test_q_table.npy')
    
    assert np.array_equal(agent.q_table, new_agent.q_table)
