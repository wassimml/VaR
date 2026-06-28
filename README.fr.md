# Value at Risk (VaR) — Analyse du Risque de Marché

Boîte à outils Python implémentant plusieurs méthodologies de VaR sur un portefeuille actions multi-actifs, avec backtesting et tests de résistance historiques.

---

## Portefeuille

| Ticker | Société               | Poids |
|--------|-----------------------|-------|
| AAPL   | Apple Inc.            | 25%   |
| MSFT   | Microsoft Corp.       | 20%   |
| JPM    | JPMorgan Chase        | 20%   |
| XOM    | ExxonMobil            | 20%   |
| JNJ    | Johnson & Johnson     | 15%   |

- **Valeur du portefeuille :** 1 000 000 $  
- **Période des données :** 2019-01-01 → 2024-01-01 (via Yahoo Finance)  
- **Rendements :** Log-rendements sur les cours ajustés des clôtures

---

## Méthodes implémentées

### Modèles VaR (`src/var_models.py`)

| Méthode | Description |
|---------|-------------|
| **VaR Historique** | Percentile empirique des P&L passés du portefeuille |
| **VaR Glissante** | VaR historique sur fenêtre mobile (50, 100, 252 jours) |
| **CVaR / Expected Shortfall** | Perte moyenne au-delà du seuil de VaR |
| **VaR Paramétrique** | Hypothèse de distribution normale avec moyenne et covariance du portefeuille |
| **VaR Monte Carlo** | Simulation normale multivariée (10 000 tirages par défaut) |

Toutes les méthodes supportent les niveaux de confiance α = 95% et α = 99%.

### Backtesting (`src/backtest.py`)

- Compare le P&L réel journalier à la prévision de VaR glissante du jour précédent  
- Reporte le nombre de violations et le taux observé vs. le taux attendu  
- **Test de Kupiec (ratio de vraisemblance)** pour valider la couverture statistique du modèle

### Tests de résistance (`src/stress_test.py`)

Trois scénarios de crise historiques simulés sur le portefeuille d'1 M$ :

| Scénario | Hypothèses |
|----------|------------|
| **Crise Lehman 2008** | −5 %/jour pendant 10 jours, corrélation entre les jours +0,30 |
| **COVID-19 2020** | Choc ponctuel de −30 %, puis volatilité ×3 pendant 20 jours |
| **Hausse des taux 2022** | Tech −25 %, Valeur −10 %, Énergie +15 % |

---

## Structure du projet

```
VaR/
├── src/
│   ├── data_loader.py      # Téléchargement Yahoo Finance, log-rendements, graphiques EDA
│   ├── var_models.py       # Toutes les méthodes VaR/CVaR + graphique de comparaison
│   ├── backtest.py         # Backtesting de la VaR glissante + test de Kupiec
│   ├── stress_test.py      # Simulation des trois scénarios de crise
│   ├── plot_skewness.py    # Illustration visuelle du skewness (3 distributions)
│   └── plot_kurtosis.py    # Illustration visuelle du kurtosis / fat tails
├── reports/                # Graphiques générés automatiquement (PNG)
├── requirements.txt
└── Compte_Rendu_VaR.pdf    # Compte rendu du projet
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Dépendances :** `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `yfinance`, `jupyterlab`

---

## Utilisation

Chaque module est exécutable indépendamment :

```bash
# 1. Analyse exploratoire (prix, rendements, QQ-plots, test de Jarque-Bera)
python src/data_loader.py

# 2. Calcul de tous les modèles VaR et génération des graphiques comparatifs
python src/var_models.py

# 3. Backtesting de la VaR glissante (99 %) avec test de Kupiec
python src/backtest.py

# 4. Simulation des scénarios de stress test
python src/stress_test.py

# 5. Illustrations pédagogiques (skewness et kurtosis)
python src/plot_skewness.py
python src/plot_kurtosis.py
```

Tous les graphiques sont sauvegardés automatiquement dans `reports/`.
