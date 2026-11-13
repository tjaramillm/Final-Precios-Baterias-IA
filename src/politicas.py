"""
Políticas para el MDP de la batería (implementadas por el equipo).

  - QLearningAgent            → model-free, ε-greedy (clase 5, CheeseWorld)
  - ValueIterationPolicy      → model-based: se estima T(s,a,s') y R(s,a) con los
                                datos de entrenamiento y se resuelve Bellman
                                (clase 4, horizonte finito → inducción hacia atrás)
  - FixedSchedulePolicy       → línea base ingenua: horario fijo (carga en las horas
                                más baratas del perfil promedio, descarga en las más caras)
  - RandomPolicy              → línea base aleatoria
  - OraclePolicy              → referencia / cota superior: conoce los precios del día
                                (información perfecta) y resuelve con programación dinámica

Todas exponen:
  start_day(env)       → se llama al inicio de cada episodio (día)
  act(env, state)      → retorna la acción a ejecutar
"""
import random

import numpy as np
import pandas as pd

from entorno import ACTIONS, CARGAR, DESCARGAR, ESPERAR, BatteryEnv


def _best_valid(q_values: np.ndarray, valid: list[int]) -> int:
    """argmax de Q restringido a acciones válidas (empates → al azar)."""
    max_q = max(q_values[a] for a in valid)
    best = [a for a in valid if q_values[a] == max_q]
    return random.choice(best)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Q-Learning (model-free)
# ─────────────────────────────────────────────────────────────────────────────
class QLearningAgent:
    name = "Q-learning"

    def __init__(self, env: BatteryEnv, omega=0.7, gamma=1.0, epsilon=1.0):
        self.env = env
        self.omega = omega        # α(s,a) = 1 / N(s,a)^ω  (decae con las visitas)
        self.gamma = gamma        # horizonte finito de 24 h → γ = 1
        self.epsilon = epsilon
        # Q(s,a) = 0 para todo par estado-acción: Q[hora, soc, nivel_precio, acción]
        self.Q = np.zeros((24, env.n_levels + 1, 3, len(ACTIONS)))
        # N(s,a): número de visitas, para la tasa de aprendizaje decreciente
        self.N = np.zeros_like(self.Q)

    def choose_action(self, state):
        """Selecciona una acción usando epsilon-greedy."""
        hour, soc, lvl = state
        actions = self.env.possible_actions(soc)
        if random.random() < self.epsilon:          # exploration
            return random.choice(actions)
        return _best_valid(self.Q[hour, soc, lvl], actions)   # exploitation

    def update_q(self, state, action, reward, next_state, done):
        """Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]"""
        current_q = self.Q[state][action]
        if done:
            best_next_q = 0.0
        else:
            valid = self.env.possible_actions(next_state[1])
            best_next_q = max(self.Q[next_state][a] for a in valid)

        # Tasa de aprendizaje decreciente: con α fijo la Q-table oscila por el
        # ruido de los precios; con α = 1/N^ω se promedian las experiencias.
        self.N[state][action] += 1
        alpha = 1.0 / self.N[state][action] ** self.omega

        td_target = reward + self.gamma * best_next_q
        td_error = td_target - current_q
        self.Q[state][action] = current_q + alpha * td_error

    def run_one_episode(self, day: int) -> float:
        state = self.env.reset(day)
        done, total_reward = False, 0.0
        while not done:
            action = self.choose_action(state)
            next_state, reward, done = self.env.step(action)
            self.update_q(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
        return total_reward

    def train(self, epochs=30, eps_start=1.0, eps_end=0.05, seed=42):
        """Recorre los días de entrenamiento `epochs` veces con ε decreciente."""
        random.seed(seed)
        days = list(range(self.env.n_days))
        history = []
        for epoch in range(epochs):
            self.epsilon = eps_start + (eps_end - eps_start) * epoch / max(1, epochs - 1)
            random.shuffle(days)
            rewards = [self.run_one_episode(d) for d in days]
            history.append(float(np.mean(rewards)))
        self.epsilon = 0.0
        return history

    # Interfaz de política (greedy)
    def start_day(self, env):
        pass

    def act(self, env, state):
        hour, soc, lvl = state
        return _best_valid(self.Q[hour, soc, lvl], env.possible_actions(soc))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Iteración de valor con modelo estimado (model-based)
# ─────────────────────────────────────────────────────────────────────────────
class ValueIterationPolicy:
    """
    Se estima el modelo con los datos de entrenamiento:
        P(nivel' | hora, nivel)   → transición del nivel de precio
        E[precio | hora, nivel]   → recompensa esperada R(s,a)
    y se aplica Bellman de optimalidad con horizonte finito (24 h):
        V_24(·) = 0
        V_h(soc,l) = max_a [ R(h,l,a) + Σ_l' P(l'|h,l) V_{h+1}(soc',l') ]
    """
    name = "Iteración de valor"

    def __init__(self, env: BatteryEnv):
        self.env = env
        self.mean_price = np.zeros((24, 3))
        self.P = np.full((24, 3, 3), 1 / 3)
        self.V = np.zeros((25, env.n_levels + 1, 3))
        self.Q = np.full((24, env.n_levels + 1, 3, len(ACTIONS)), -np.inf)

    def fit_model(self):
        env = self.env
        levels = np.array([[env.price_level(d, h) for h in range(24)] for d in range(env.n_days)])
        hourly_mean = env.prices.mean(axis=0)

        for h in range(24):
            for l in range(3):
                mask = levels[:, h] == l
                # Si no hay datos para (h,l) se usa el promedio de la hora
                self.mean_price[h, l] = env.prices[mask, h].mean() if mask.any() else hourly_mean[h]
                if h < 23 and mask.any():
                    counts = np.bincount(levels[mask, h + 1], minlength=3)
                    self.P[h, l] = counts / counts.sum()
        return self

    def value_iteration(self):
        env = self.env
        for h in reversed(range(24)):            # inducción hacia atrás
            for soc in range(env.n_levels + 1):
                for l in range(3):
                    for a in env.possible_actions(soc):
                        r = env.reward(self.mean_price[h, l], a)
                        soc2 = soc + {CARGAR: 1, DESCARGAR: -1, ESPERAR: 0}[a]
                        future = 0.0 if h == 23 else self.P[h, l] @ self.V[h + 1, soc2]
                        self.Q[h, soc, l, a] = r + future
                    self.V[h, soc, l] = self.Q[h, soc, l].max()
        return self

    def start_day(self, env):
        pass

    def act(self, env, state):
        hour, soc, lvl = state
        return _best_valid(self.Q[hour, soc, lvl], env.possible_actions(soc))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Líneas base y referencia
# ─────────────────────────────────────────────────────────────────────────────
class FixedSchedulePolicy:
    """Ingenua: carga en las N horas más baratas y descarga en las N más caras
    del perfil horario promedio de entrenamiento. No reacciona al precio del día."""
    name = "Horario fijo (ingenua)"

    def __init__(self, env_train: BatteryEnv):
        profile = env_train.prices.mean(axis=0)
        n = env_train.n_levels
        order = np.argsort(profile)
        self.charge_hours = set(order[:n].tolist())
        self.discharge_hours = set(order[-n:].tolist())

    def start_day(self, env):
        pass

    def act(self, env, state):
        hour, soc, _ = state
        valid = env.possible_actions(soc)
        if hour in self.charge_hours and CARGAR in valid:
            return CARGAR
        if hour in self.discharge_hours and DESCARGAR in valid:
            return DESCARGAR
        return ESPERAR


class RandomPolicy:
    name = "Aleatoria"

    def start_day(self, env):
        pass

    def act(self, env, state):
        return random.choice(env.possible_actions(state[1]))


class OraclePolicy:
    """Cota superior: conoce los 24 precios del día (no implementable en la realidad)."""
    name = "Oráculo (info. perfecta)"

    def start_day(self, env):
        n = env.n_levels
        prices = env.prices[env.day]
        V = np.zeros((25, n + 1))
        self.plan = np.zeros((24, n + 1), dtype=int)
        for h in reversed(range(24)):
            for soc in range(n + 1):
                best_a, best_v = ESPERAR, -np.inf
                for a in env.possible_actions(soc):
                    soc2 = soc + {CARGAR: 1, DESCARGAR: -1, ESPERAR: 0}[a]
                    v = env.reward(prices[h], a) + V[h + 1, soc2]
                    if v > best_v:
                        best_a, best_v = a, v
                V[h, soc], self.plan[h, soc] = best_v, best_a

    def act(self, env, state):
        hour, soc, _ = state
        return int(self.plan[hour, soc])


# ─────────────────────────────────────────────────────────────────────────────
# 4. Simulación y evaluación
# ─────────────────────────────────────────────────────────────────────────────
def simulate(policy, env: BatteryEnv) -> pd.DataFrame:
    """Ejecuta la política sobre todos los días del entorno. Retorna el detalle horario."""
    rows = []
    for d in range(env.n_days):
        state = env.reset(d)
        policy.start_day(env)
        done = False
        while not done:
            hour = env.hour
            action = policy.act(env, state)
            price = env.prices[d, hour]
            next_state, reward, done = env.step(action)
            rows.append({
                "fecha": env.dates[d], "hora": hour, "precio": price,
                "nivel_precio": state[2], "accion": ACTIONS[action],
                "soc_kwh": env.soc * env.e_step, "recompensa": reward,
            })
            state = next_state
    df = pd.DataFrame(rows)
    df["ahorro_acumulado"] = df["recompensa"].cumsum()
    return df


def summarize(df: pd.DataFrame, env: BatteryEnv) -> dict:
    n_days = df["fecha"].nunique()
    discharged = (df["accion"] == "descargar").sum() * env.e_step
    return {
        "ahorro_total_cop": float(df["recompensa"].sum()),
        "ahorro_diario_cop": float(df["recompensa"].sum() / n_days),
        "ciclos_equivalentes": float(discharged / env.capacity_kwh),
        "dias": int(n_days),
    }


def build_policies(env_train: BatteryEnv, q_epochs=200, seed=42) -> dict:
    """Entrena/ajusta todas las políticas sobre el periodo de entrenamiento."""
    # Q-learning se entrena con recompensa relativa (copia del entorno)
    env_q = BatteryEnv(
        pd.DataFrame(env_train.prices, index=env_train.dates),
        env_train.capacity_kwh, env_train.capacity_kwh / env_train.n_levels,
        env_train.efficiency, env_train.degradation, relative_reward=True,
    )
    q_agent = QLearningAgent(env_q)
    q_history = q_agent.train(epochs=q_epochs, seed=seed)
    vi = ValueIterationPolicy(env_train).fit_model().value_iteration()
    policies = {
        "q_learning": q_agent,
        "iteracion_valor": vi,
        "horario_fijo": FixedSchedulePolicy(env_train),
        "aleatoria": RandomPolicy(),
        "oraculo": OraclePolicy(),
    }
    return policies, q_history


def compare_policies(policies: dict, env_test: BatteryEnv, seed=0) -> tuple[pd.DataFrame, dict]:
    """Simula todas las políticas sobre el mismo periodo. Retorna tabla y detalle."""
    random.seed(seed)
    details, rows = {}, []
    for key, pol in policies.items():
        df = simulate(pol, env_test)
        details[key] = df
        rows.append({"politica": pol.name, "clave": key, **summarize(df, env_test)})
    table = pd.DataFrame(rows).set_index("clave")
    oracle = table.loc["oraculo", "ahorro_total_cop"] if "oraculo" in table.index else np.nan
    table["pct_del_oraculo"] = 100 * table["ahorro_total_cop"] / oracle
    return table, details
