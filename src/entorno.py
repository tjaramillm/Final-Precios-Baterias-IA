"""
Entorno MDP: batería que compra/vende energía al precio de bolsa horario (XM).

Formulación (misma notación de las clases 4 y 5):

  Estado   s = (hora, nivel_soc, nivel_precio)
             hora         ∈ {0,…,23}
             nivel_soc    ∈ {0,…,N}      (N = capacidad / energía por paso)
             nivel_precio ∈ {0,1,2}      (bajo / medio / alto respecto al
                                          promedio del día anterior → causal)
  Acciones a ∈ {descargar, esperar, cargar}
  Transición: la hora avanza 1; el SOC sube/baja un nivel; el nivel de precio
              de la hora siguiente es estocástico (viene de los datos reales).
  Recompensa (COP):
     cargar     → −precio · e_paso                     (se compra energía)
     descargar  → +precio · e_paso · η − c_deg · e_paso (se evita comprar / se vende)
     esperar    → 0
  Episodio: un día (24 pasos), inicia con la batería vacía.

La suma de recompensas es el ahorro frente a no tener batería: si el usuario
paga el precio de bolsa, cargar aumenta su compra y descargar la reduce.
"""
import numpy as np
import pandas as pd

ACTIONS = ["descargar", "esperar", "cargar"]
DESCARGAR, ESPERAR, CARGAR = 0, 1, 2
PRICE_LEVELS = ["bajo", "medio", "alto"]

# Umbrales del ratio precio / promedio del día anterior
LOW_RATIO, HIGH_RATIO = 0.85, 1.15


class BatteryEnv:
    def __init__(
        self,
        prices: pd.DataFrame,          # una fila por día, 24 columnas (horas)
        capacity_kwh: float = 10.0,
        power_kw: float = 2.5,
        efficiency: float = 0.90,      # eficiencia de ida y vuelta
        degradation_cop_kwh: float = 0.0,
        relative_reward: bool = False,
    ):
        # relative_reward=True divide la recompensa por el precio de referencia del
        # día. Se usa SOLO para entrenar Q-learning: reduce la varianza entre días
        # baratos y caros (p. ej. El Niño 2024) sin cambiar qué hora conviene más.
        self.relative_reward = relative_reward
        self.prices = prices.to_numpy(dtype=float)
        self.dates = list(prices.index)
        self.n_days = len(self.dates)

        # Discretización del SOC: un paso = lo que la batería carga en 1 hora
        self.n_levels = max(1, int(round(capacity_kwh / power_kw)))
        self.e_step = capacity_kwh / self.n_levels
        self.capacity_kwh = capacity_kwh
        self.efficiency = efficiency
        self.degradation = degradation_cop_kwh

        # Precio de referencia de cada día = promedio del día anterior (causal)
        daily_mean = self.prices.mean(axis=1)
        self.ref_price = np.concatenate([[daily_mean[0]], daily_mean[:-1]])

        self.actions = ACTIONS
        self.day, self.hour, self.soc = 0, 0, 0

    # ── Espacio de estados ────────────────────────────────────────────────
    def states(self):
        return [
            (h, soc, lvl)
            for h in range(24)
            for soc in range(self.n_levels + 1)
            for lvl in range(3)
        ]

    def price_level(self, day: int, hour: int) -> int:
        ratio = self.prices[day, hour] / self.ref_price[day]
        if ratio < LOW_RATIO:
            return 0
        if ratio > HIGH_RATIO:
            return 2
        return 1

    def possible_actions(self, soc: int):
        """Acciones válidas según el nivel de carga."""
        actions = [ESPERAR]
        if soc > 0:
            actions.append(DESCARGAR)
        if soc < self.n_levels:
            actions.append(CARGAR)
        return sorted(actions)

    # ── Dinámica ──────────────────────────────────────────────────────────
    def reward(self, price: float, action: int) -> float:
        if action == CARGAR:
            return -price * self.e_step
        if action == DESCARGAR:
            return price * self.e_step * self.efficiency - self.degradation * self.e_step
        return 0.0

    def reset(self, day: int):
        """Inicia el episodio (un día) con la batería vacía."""
        self.day, self.hour, self.soc = day, 0, 0
        return (0, 0, self.price_level(day, 0))

    def step(self, action: int):
        """Ejecuta una acción y retorna (next_state, reward, done)."""
        if action not in self.possible_actions(self.soc):
            raise ValueError(f"Acción inválida {ACTIONS[action]!r} con soc={self.soc}")

        price = self.prices[self.day, self.hour]
        reward = self.reward(price, action)
        if self.relative_reward:
            reward /= self.ref_price[self.day]
        self.soc += {CARGAR: 1, DESCARGAR: -1, ESPERAR: 0}[action]
        self.hour += 1

        done = self.hour == 24
        next_state = None if done else (self.hour, self.soc, self.price_level(self.day, self.hour))
        return next_state, reward, done
