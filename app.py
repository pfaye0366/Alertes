import streamlit as st
import yfinance as yf
import pandas as pd
import time
from datetime import datetime, timedelta
from collections import defaultdict
import csv
from zoneinfo import ZoneInfo  # intégré à Python ≥3.9

# Configuration de la page
st.set_page_config(
    page_title="Alertes Day Trading [TT]",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    .stAlert {
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .alert-up {
        background-color: rgba(56, 239, 125, 0.2);
        border-left: 5px solid #38ef7d;
    }
    .alert-down {
        background-color: rgba(255, 106, 0, 0.2);
        border-left: 5px solid #ff6a00;
    }
    .alert-neutral {
        background-color: rgba(78, 84, 200, 0.2);
        border-left: 5px solid #4e54c8;
    }
    .alert-warning {
        background-color: rgba(255, 215, 0, 0.2);
        border-left: 5px solid #ffd700;
    }
    h1, h2, h3 {
        color: white;
    }
    .stMetric {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 10px;
    }
    .stMetric label {
        color: white !important;
    }
    .stMetric .metric-value {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

class TradingAlertSystem:
    def __init__(self):
        if 'previous_data' not in st.session_state:
            st.session_state.previous_data = {}
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
        if 'alert_count' not in st.session_state:
            st.session_state.alert_count = 0
        if 'active_alerts' not in st.session_state:
            st.session_state.active_alerts = {}
        if 'simulation_time' not in st.session_state:
            st.session_state.simulation_time = None
    
    def load_ticker_data(self, ticker, end_datetime=None, period_days=1):
        """Récupère les données avec yfinance - utilise la dernière bougie FERMÉE
        
        Args:
            ticker: Symbole de l'action
            end_datetime: Datetime de fin pour le backtesting (None = temps réel)
            period_days: Nombre de jours de données à récupérer
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Déterminer la période à récupérer
            if period_days == 1:
                period = "1d"
            elif period_days <= 5:
                period = "5d"
            elif period_days <= 30:
                period = "1mo"
            else:
                period = "1mo"  # Maximum pour données 1 minute
            
            # Récupérer les données avec intervalle de 1 minute
            hist = stock.history(period=period, interval="1m")
            
            if hist.empty or len(hist) < 3:  # Besoin d'au moins 3 bougies
                return None
            
            # Si mode simulation avec end_datetime, filtrer les données
            if end_datetime is not None:
                # Convertir end_datetime en timezone America/New_York (timezone du marché US)
                import pytz
                
                # Timezone du marché US
                ny_tz = pytz.timezone('America/New_York')
                
                # Localiser end_datetime dans la timezone NY si pas déjà fait
                if end_datetime.tzinfo is None:
                    # Supposer que l'utilisateur entre l'heure en heure de NY (marché US)
                    end_datetime_tz = ny_tz.localize(end_datetime)
                else:
                    # Convertir en timezone NY
                    end_datetime_tz = end_datetime.astimezone(ny_tz)
                
                # Si l'index a une timezone différente, convertir
                if hist.index.tzinfo is not None:
                    # S'assurer que les deux sont dans la même timezone pour la comparaison
                    if hist.index.tzinfo != end_datetime_tz.tzinfo:
                        # Convertir l'index dans la même timezone
                        hist.index = hist.index.tz_convert(ny_tz)
                
                # Filtrer pour ne garder que les données jusqu'à end_datetime_tz
                hist = hist[hist.index <= end_datetime_tz]
                
                if hist.empty or len(hist) < 3:
                    return None
            
            # IMPORTANT: Utiliser la dernière bougie FERMÉE (iloc[-2])
            # La dernière bougie (iloc[-1]) est en cours de formation
            if len(hist) >= 2:
                current_price = hist['Close'].iloc[-2]
                current_high = hist['High'].iloc[-2]
                current_low = hist['Low'].iloc[-2]
                current_open = hist['Open'].iloc[-2]
                current_volume = hist['Volume'].iloc[-2]
                
                # Timestamp de la bougie fermée
                candle_timestamp = hist.index[-2]
            else:
                # Pas assez de données
                return None
            
            # Volume: gérer les volumes à 0
            if current_volume == 0 or pd.isna(current_volume):
                recent_volumes = hist['Volume'].iloc[:-1].tail(10)  # Exclure la bougie en cours
                recent_volumes = recent_volumes[recent_volumes > 0]
                if len(recent_volumes) > 0:
                    current_volume = recent_volumes.mean()
                else:
                    current_volume = 0
            
            # Données de la bougie précédente (avant la dernière fermée)
            prev_close = hist['Close'].iloc[-3]
            
            # Variation depuis la bougie précédente (1 minute)
            price_change_1m = ((current_price - prev_close) / prev_close) * 100
            
            # Ouverture du jour (première bougie)
            day_open = hist['Open'].iloc[0]
            price_change_day = ((current_price - day_open) / day_open) * 100
            
            # VWAP calculation (exclure la bougie en cours)
            valid_hist = hist.iloc[:-1][hist.iloc[:-1]['Volume'] > 0]
            if len(valid_hist) > 0:
                vwap = (valid_hist['Volume'] * (valid_hist['High'] + valid_hist['Low'] + valid_hist['Close']) / 3).sum() / valid_hist['Volume'].sum()
            else:
                vwap = current_price
            
            # Volume moyen sur les 20 dernières minutes (exclure la bougie en cours)
            recent_volumes = hist['Volume'].iloc[:-1].tail(20)
            recent_volumes = recent_volumes[recent_volumes > 0]
            avg_volume = recent_volumes.mean() if len(recent_volumes) > 0 else 1
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Plus haut et plus bas du jour (exclure la bougie en cours)
            day_high = hist['High'].iloc[:-1].max()
            day_low = hist['Low'].iloc[:-1].min()
            
            # Clôture précédente (veille)
            info = stock.info
            previous_close = info.get('previousClose', day_open)
            
            # Calculer les ORB (exclure la bougie en cours)
            # IMPORTANT: Les ORB ne doivent être calculés que sur les bougies DU JOUR
            orb_data = {}
            
            # Déterminer la date du jour à analyser
            # Si on a une bougie fermée, utiliser sa date
            if len(hist) >= 2:
                current_day = candle_timestamp.date()
                
                # Filtrer uniquement les bougies du jour actuel
                hist_today = hist[hist.index.date == current_day]
                
                # Exclure la dernière bougie (en cours de formation)
                hist_closed_today = hist_today.iloc[:-1]
                
                for minutes in [5, 15, 30]:
                    if len(hist_closed_today) >= minutes:
                        # Prendre les X premières minutes du jour
                        orb_range = hist_closed_today.head(minutes)
                        orb_data[minutes] = {
                            'high': orb_range['High'].max(),
                            'low': orb_range['Low'].min()
                        }
            else:
                orb_data = {}
            
            data = {
                'ticker': ticker,
                'price': current_price,
                'change_1m': price_change_1m,
                'change': price_change_day,
                'volume': current_volume,
                'avg_volume': avg_volume,
                'volume_ratio': volume_ratio,
                'vwap': vwap,
                'high': day_high,
                'low': day_low,
                'open': day_open,
                'current_high': current_high,
                'current_low': current_low,
                'current_open': current_open,
                'prev_close': prev_close,
                'previous_close': previous_close,
                'orb': orb_data,
                'candle_time': candle_timestamp
            }
            
            return data
            
        except Exception as e:
            st.error(f"Erreur pour {ticker}: {str(e)}")
            return None
    
    def detect_alerts(self, data, params):
        """Détecte les différents types d'alertes avec paramètres configurables"""
        alerts = []
        ticker = data['ticker']
        
        # Initialiser le tracker d'alertes actives pour ce ticker si nécessaire
        if ticker not in st.session_state.active_alerts:
            st.session_state.active_alerts[ticker] = set()
        
        # 1. Gappers UP/DOWN (éviter les répétitions)
        if params.get('gap_enabled', True):
            gap = ((data['price'] - data['previous_close']) / data['previous_close']) * 100
            gap_threshold = params.get('gap_threshold', 3.0)
            
            gap_up_key = f"GAP_UP_{ticker}"
            gap_down_key = f"GAP_DOWN_{ticker}"
            
            if gap > gap_threshold:
                if gap_up_key not in st.session_state.active_alerts[ticker]:
                    alerts.append({
                        'type': 'Gappers UP',
                        'severity': 'up',
                        'icon': '▲',
                        'message': f"Gap de {gap:.2f}% au-dessus de la clôture précédente (seuil: {gap_threshold}%)"
                    })
                    st.session_state.active_alerts[ticker].add(gap_up_key)
                st.session_state.active_alerts[ticker].discard(gap_down_key)
                
            elif gap < -gap_threshold:
                if gap_down_key not in st.session_state.active_alerts[ticker]:
                    alerts.append({
                        'type': 'Gappers DOWN',
                        'severity': 'down',
                        'icon': '▼',
                        'message': f"Gap de {abs(gap):.2f}% en dessous de la clôture précédente (seuil: {gap_threshold}%)"
                    })
                    st.session_state.active_alerts[ticker].add(gap_down_key)
                st.session_state.active_alerts[ticker].discard(gap_up_key)
            else:
                st.session_state.active_alerts[ticker].discard(gap_up_key)
                st.session_state.active_alerts[ticker].discard(gap_down_key)
        
        # 2. High Volume (éviter les répétitions)
        if params.get('high_volume_enabled', True):
            high_volume_ratio = params.get('high_volume_ratio', 2.0)
            high_volume_key = f"HIGH_VOL_{ticker}"
            
            if data['volume_ratio'] > high_volume_ratio:
                # Vérifier si cette alerte n'était pas déjà active
                if high_volume_key not in st.session_state.active_alerts[ticker]:
                    alerts.append({
                        'type': 'High Volume Trending',
                        'severity': 'neutral',
                        'icon': '📈',
                        'message': f"Volume {data['volume_ratio']:.1f}x supérieur à la moyenne (seuil: {high_volume_ratio}x)"
                    })
                    st.session_state.active_alerts[ticker].add(high_volume_key)
            else:
                # Volume revenu à la normale, réinitialiser
                st.session_state.active_alerts[ticker].discard(high_volume_key)
        
        # 3. VWAP Breaking Out (éviter les répétitions)
        if params.get('vwap_enabled', True):
            vwap_distance = ((data['price'] - data['vwap']) / data['vwap']) * 100
            vwap_threshold = params.get('vwap_threshold', 1.0)
            
            vwap_alert_key = None
            if vwap_distance > vwap_threshold:
                vwap_alert_key = f"VWAP_UP_{ticker}"
                if vwap_alert_key not in st.session_state.active_alerts[ticker]:
                    alerts.append({
                        'type': 'VWAP Breaking Out',
                        'severity': 'up',
                        'icon': '🎯',
                        'message': f"Prix {vwap_distance:.2f}% au-dessus du VWAP (seuil: {vwap_threshold}%)"
                    })
                    st.session_state.active_alerts[ticker].add(vwap_alert_key)
            elif vwap_distance < -vwap_threshold:
                vwap_alert_key = f"VWAP_DOWN_{ticker}"
                if vwap_alert_key not in st.session_state.active_alerts[ticker]:
                    alerts.append({
                        'type': 'VWAP Breaking Out',
                        'severity': 'down',
                        'icon': '🎯',
                        'message': f"Prix {abs(vwap_distance):.2f}% en dessous du VWAP (seuil: {vwap_threshold}%)"
                    })
                    st.session_state.active_alerts[ticker].add(vwap_alert_key)
            else:
                st.session_state.active_alerts[ticker].discard(f"VWAP_UP_{ticker}")
                st.session_state.active_alerts[ticker].discard(f"VWAP_DOWN_{ticker}")
        
        # 4. Volume Breakouts (éviter les répétitions)
        if params.get('volume_breakout_enabled', True):
            volume_breakout_ratio = params.get('volume_breakout_ratio', 3.0)
            volume_breakout_key = f"VOL_BREAKOUT_{ticker}"
            
            if data['volume_ratio'] > volume_breakout_ratio:
                # Vérifier si cette alerte n'était pas déjà active
                if volume_breakout_key not in st.session_state.active_alerts[ticker]:
                    alerts.append({
                        'type': 'Volume Breakouts',
                        'severity': 'warning',
                        'icon': '💥',
                        'message': f"Volume explosif: {data['volume_ratio']:.1f}x la moyenne (seuil: {volume_breakout_ratio}x)"
                    })
                    st.session_state.active_alerts[ticker].add(volume_breakout_key)
            else:
                # Volume revenu à la normale, réinitialiser
                st.session_state.active_alerts[ticker].discard(volume_breakout_key)
        
        # 5. Turbo BreakUp/Down (basé sur la variation de 1 minute)
        if params.get('turbo_enabled', True):
            turbo_threshold = params.get('turbo_threshold', 2.0)
            
            if abs(data['change_1m']) > turbo_threshold:
                if data['change_1m'] > 0:
                    alerts.append({
                        'type': 'Turbo BreakUp',
                        'severity': 'up',
                        'icon': '🚀',
                        'message': f"Mouvement rapide sur 1min: +{data['change_1m']:.2f}% (seuil: {turbo_threshold}%)"
                    })
                else:
                    alerts.append({
                        'type': 'Turbo BreakDown',
                        'severity': 'down',
                        'icon': '⚡',
                        'message': f"Mouvement rapide sur 1min: {data['change_1m']:.2f}% (seuil: {turbo_threshold}%)"
                    })
        
        # 6. BHOD/BLOD (éviter les répétitions)
        if params.get('bhod_blod_enabled', True):
            bhod_blod_distance = params.get('bhod_blod_distance', 0.5)
            high_threshold = 1 - (bhod_blod_distance / 100)
            low_threshold = 1 + (bhod_blod_distance / 100)
            
            bhod_key = f"BHOD_{ticker}"
            blod_key = f"BLOD_{ticker}"
            
            if data['price'] >= data['high'] * high_threshold:
                if bhod_key not in st.session_state.active_alerts[ticker]:
                    alerts.append({
                        'type': 'BHOD (Break High Of Day)',
                        'severity': 'up',
                        'icon': '📊',
                        'message': f"Prix au plus haut du jour: ${data['high']:.2f} (distance: {bhod_blod_distance}%)"
                    })
                    st.session_state.active_alerts[ticker].add(bhod_key)
                st.session_state.active_alerts[ticker].discard(blod_key)
                
            elif data['price'] <= data['low'] * low_threshold:
                if blod_key not in st.session_state.active_alerts[ticker]:
                    alerts.append({
                        'type': 'BLOD (Break Low Of Day)',
                        'severity': 'down',
                        'icon': '📊',
                        'message': f"Prix au plus bas du jour: ${data['low']:.2f} (distance: {bhod_blod_distance}%)"
                    })
                    st.session_state.active_alerts[ticker].add(blod_key)
                st.session_state.active_alerts[ticker].discard(bhod_key)
            else:
                st.session_state.active_alerts[ticker].discard(bhod_key)
                st.session_state.active_alerts[ticker].discard(blod_key)
        
        # 7. Extreme Reversals
        if params.get('reversal_enabled', True) and ticker in st.session_state.previous_data:
            prev_change = st.session_state.previous_data[ticker]['change']
            reversal_threshold = params.get('reversal_threshold', 2.0)
            if (prev_change > reversal_threshold and data['change'] < -reversal_threshold) or \
               (prev_change < -reversal_threshold and data['change'] > reversal_threshold):
                alerts.append({
                    'type': 'Extreme Reversals',
                    'severity': 'warning',
                    'icon': '🔄',
                    'message': f"Retournement de {prev_change:.2f}% à {data['change']:.2f}% (seuil: {reversal_threshold}%)"
                })
        
        # 8. Opening Range Breakouts (ORB) - éviter les répétitions
        if params.get('orb_enabled', False) and 'orb' in data:
            orb_periods = params.get('orb_periods', [5, 15, 30])
            
            for period in orb_periods:
                if period in data['orb']:
                    orb_high = data['orb'][period]['high']
                    orb_low = data['orb'][period]['low']
                    
                    orb_up_key = f"ORB_{period}_UP_{ticker}"
                    orb_down_key = f"ORB_{period}_DOWN_{ticker}"
                    
                    if data['price'] > orb_high:
                        if orb_up_key not in st.session_state.active_alerts[ticker]:
                            alerts.append({
                                'type': f'ORB {period}min BreakUp',
                                'severity': 'up',
                                'icon': '⏰',
                                'message': f"Prix ${data['price']:.2f} franchit ORB high ${orb_high:.2f} ({period}min)"
                            })
                            st.session_state.active_alerts[ticker].add(orb_up_key)
                        st.session_state.active_alerts[ticker].discard(orb_down_key)
                    
                    elif data['price'] < orb_low:
                        if orb_down_key not in st.session_state.active_alerts[ticker]:
                            alerts.append({
                                'type': f'ORB {period}min BreakDown',
                                'severity': 'down',
                                'icon': '⏰',
                                'message': f"Prix ${data['price']:.2f} franchit ORB low ${orb_low:.2f} ({period}min)"
                            })
                            st.session_state.active_alerts[ticker].add(orb_down_key)
                        st.session_state.active_alerts[ticker].discard(orb_up_key)
                    
                    else:
                        st.session_state.active_alerts[ticker].discard(orb_up_key)
                        st.session_state.active_alerts[ticker].discard(orb_down_key)
        
        return alerts
    
    def format_volume(self, volume):
        """Formate le volume"""
        if volume >= 1_000_000:
            return f"{volume/1_000_000:.1f}M"
        elif volume >= 1_000:
            return f"{volume/1_000:.0f}K"
        return str(int(volume))

def load_tickers_from_csv(file):
    """Charge les tickers depuis un fichier CSV uploadé"""
    tickers = []
    content = file.getvalue().decode("utf-8")
    csv_reader = csv.reader(content.splitlines())
    
    for row in csv_reader:
        if row and row[0].strip():
            ticker = row[0].strip().upper()
            if ticker and not ticker.startswith('#'):
                tickers.append(ticker)
    
    return tickers

def main():
    # Titre principal
    st.title("📊 SYSTÈME D'ALERTES DAY TRADING [TT]")
    st.markdown("### Surveillance en temps réel avec Yahoo Finance")
    
    # Initialiser le système
    system = TradingAlertSystem()
    
    # Sidebar pour les contrôles
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Upload fichier CSV
        uploaded_file = st.file_uploader("📁 Charger fichier CSV", type=['csv', 'txt'])
        
        if uploaded_file is not None:
            tickers = load_tickers_from_csv(uploaded_file)
            st.success(f"✓ {len(tickers)} tickers chargés")
            st.session_state.tickers = tickers
        else:
            if 'tickers' not in st.session_state:
                st.session_state.tickers = ['AAPL', 'TSLA', 'NVDA', 'AMD', 'MSFT']
        
        st.info(f"**Tickers:** {', '.join(st.session_state.tickers[:5])}" + 
                (f" ... (+{len(st.session_state.tickers)-5})" if len(st.session_state.tickers) > 5 else ""))
        
        # Intervalle de rafraîchissement
        refresh_interval = st.slider("⏱️ Intervalle de rafraîchissement (secondes)", 
                                     min_value=5, max_value=60, value=10, step=5)
        
        st.markdown("---")
        st.subheader("📅 Backtesting / Simulation")
        
        # Mode backtesting
        backtest_mode = st.checkbox("Mode Simulation", value=False, key='backtest_mode')
        
        if backtest_mode:
            # Période de données à récupérer (indépendant de la date de départ)
            period_days = st.slider(
                "📦 Période de données (jours)",
                min_value=1,
                max_value=30,
                value=5,
                step=1,
                key='period_days',
                help="Nombre de jours de données historiques à télécharger depuis yfinance"
            )
            
            st.caption(f"💾 Télécharge {period_days} jour(s) de données historiques")
            
            st.markdown("---")
            
            # Date/heure de départ de la simulation
            backtest_date = st.date_input(
                "📅 Date de départ simulation",
                value=datetime.now().date(),
                key='backtest_date',
                help="Point de départ de votre simulation"
            )
            
            backtest_time = st.time_input(
                "🕐 Heure de départ simulation",
                value=datetime.now().time(),
                key='backtest_time',
                help="Heure de début de la simulation"
            )
            
            backtest_datetime = datetime.combine(backtest_date, backtest_time)
            
            # Sauvegarder immédiatement dans session_state
            st.session_state.backtest_datetime = backtest_datetime
            st.session_state.backtest_period_days = period_days
            
            # Initialiser simulation_time si elle n'existe pas
            if st.session_state.simulation_time is None:
                st.session_state.simulation_time = backtest_datetime
            
            if st.button("🔄 Réinitialiser simulation", key='reset_sim'):
                st.session_state.simulation_time = backtest_datetime
                st.session_state.alerts = []
                st.session_state.alert_count = 0
                st.session_state.active_alerts = {}
                st.session_state.previous_data = {}
                st.success("Simulation réinitialisée!")
                st.rerun()
            
            if st.session_state.simulation_time:
                time_diff = st.session_state.simulation_time - backtest_datetime
                minutes_elapsed = int(time_diff.total_seconds() / 60)
                
                st.info(f"🕐 **Temps simulation actuel:** {st.session_state.simulation_time.strftime('%Y-%m-%d %H:%M:%S')}")
                st.caption(f"⏱️ Temps écoulé depuis le début: {minutes_elapsed} minutes")
                st.caption(f"📦 Données disponibles: {period_days} jour(s) d'historique")
        else:
            st.session_state.backtest_datetime = None
            st.session_state.simulation_time = None
            st.session_state.backtest_period_days = 1
            st.caption("Mode temps réel actif")
        
        st.markdown("---")
        st.subheader("🎯 Paramètres des Indicateurs")
        
        params = {}
        
        # 1. Gappers
        with st.expander("▲▼ Gappers UP/DOWN", expanded=False):
            params['gap_enabled'] = st.checkbox("Activer Gappers", value=True, key='gap_en')
            params['gap_threshold'] = st.slider("Seuil de gap (%)", 
                                               min_value=0.5, max_value=10.0, value=3.0, step=0.5, key='gap_th')
            st.caption("Détecte les écarts significatifs par rapport à la clôture précédente")
        
        # 2. High Volume
        with st.expander("📈 High Volume Trending", expanded=False):
            params['high_volume_enabled'] = st.checkbox("Activer High Volume", value=True, key='hv_en')
            params['high_volume_ratio'] = st.slider("Ratio volume (x moyenne)", 
                                                    min_value=1.5, max_value=5.0, value=2.0, step=0.5, key='hv_ratio')
            st.caption("Détecte les volumes anormalement élevés")
        
        # 3. VWAP
        with st.expander("🎯 VWAP Breaking Out", expanded=False):
            params['vwap_enabled'] = st.checkbox("Activer VWAP", value=True, key='vwap_en')
            params['vwap_threshold'] = st.slider("Distance du VWAP (%)", 
                                                min_value=0.5, max_value=5.0, value=1.0, step=0.25, key='vwap_th')
            st.caption("Détecte quand le prix s'éloigne du VWAP")
        
        # 4. Volume Breakouts
        with st.expander("💥 Volume Breakouts", expanded=False):
            params['volume_breakout_enabled'] = st.checkbox("Activer Volume Breakouts", value=True, key='vb_en')
            params['volume_breakout_ratio'] = st.slider("Ratio volume explosif (x moyenne)", 
                                                        min_value=2.0, max_value=10.0, value=3.0, step=0.5, key='vb_ratio')
            st.caption("Détecte les volumes explosifs")
        
        # 5. Turbo Break
        with st.expander("🚀 Turbo BreakUp/Down", expanded=False):
            params['turbo_enabled'] = st.checkbox("Activer Turbo Break", value=True, key='turbo_en')
            params['turbo_threshold'] = st.slider("Seuil mouvement rapide (%)", 
                                                  min_value=0.5, max_value=5.0, value=2.0, step=0.25, key='turbo_th')
            st.caption("Détecte les mouvements de prix rapides sur 1 minute")
        
        # 6. BHOD/BLOD
        with st.expander("📊 BHOD/BLOD", expanded=False):
            params['bhod_blod_enabled'] = st.checkbox("Activer BHOD/BLOD", value=True, key='bhod_en')
            params['bhod_blod_distance'] = st.slider("Distance du high/low (%)", 
                                                     min_value=0.1, max_value=2.0, value=0.5, step=0.1, key='bhod_dist')
            st.caption("Détecte les cassures du plus haut/bas du jour")
        
        # 7. Extreme Reversals
        with st.expander("🔄 Extreme Reversals", expanded=False):
            params['reversal_enabled'] = st.checkbox("Activer Reversals", value=True, key='rev_en')
            params['reversal_threshold'] = st.slider("Seuil de retournement (%)", 
                                                     min_value=1.0, max_value=5.0, value=2.0, step=0.5, key='rev_th')
            st.caption("Détecte les retournements brusques de tendance")
        
        # 8. Opening Range Breakouts (ORB)
        with st.expander("⏰ Opening Range Breakouts (ORB)", expanded=False):
            params['orb_enabled'] = st.checkbox("Activer ORB", value=True, key='orb_en')
            
            orb_options = st.multiselect(
                "Périodes ORB (minutes)",
                options=[5, 15, 30],
                default=[5, 15, 30],
                key='orb_periods'
            )
            params['orb_periods'] = orb_options if orb_options else [5, 15, 30]
            
            st.caption("Détecte les cassures de la range d'ouverture (vert=hausse, orange=baisse)")
        
        st.session_state.params = params
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start", width="stretch"):
                st.session_state.running = True
                # Réinitialiser first_run pour que le premier scan soit immédiat
                if 'first_run' in st.session_state:
                    del st.session_state.first_run
                
                # IMPORTANT: Initialiser simulation_time si en mode simulation
                if st.session_state.get('backtest_datetime') is not None:
                    st.session_state.simulation_time = st.session_state.backtest_datetime
                
                st.rerun()
        
        with col2:
            if st.button("⏹️ Stop", width="stretch"):
                st.session_state.running = False
                st.rerun()
        
        if st.button("🗑️ Effacer les alertes", width="stretch"):
            st.session_state.alerts = []
            st.session_state.alert_count = 0
            st.rerun()
        
        st.markdown("---")
        if 'running' in st.session_state and st.session_state.running:
            st.success("🟢 **ACTIF**")
        else:
            st.error("🔴 **ARRÊTÉ**")
    
    # Zone principale - Statistiques
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Alertes générées", st.session_state.alert_count)
    
    with col2:
        st.metric("Tickers surveillés", len(st.session_state.tickers))
    
    with col3:
        status = "ACTIF" if ('running' in st.session_state and st.session_state.running) else "ARRÊTÉ"
        st.metric("Statut", status)
    
    st.markdown("---")
    
    # Afficher les alertes sous forme de tableau
    if len(st.session_state.alerts) == 0:
        st.info("En attente d'alertes... Cliquez sur 'Start' pour commencer la surveillance.")
    else:
        df_alerts = pd.DataFrame(st.session_state.alerts[:1000])  # Augmenté à 1000 alertes
        
        df_alerts['timestamp'] = (pd.to_datetime(df_alerts['timestamp']).dt.tz_localize('UTC').dt.tz_convert('America/Toronto'))
        df_display = pd.DataFrame({
            'Heure': df_alerts['timestamp'],
            'Ticker': df_alerts['ticker'],
            'Type': df_alerts['icon'] + ' ' + df_alerts['type'],
            'Prix': df_alerts['price'].apply(lambda x: f"${x:.2f}"),
            'Var%': df_alerts['change'].apply(lambda x: f"{x:+.2f}%"),
            'Volume': df_alerts['volume'],
            'Vol Moy': df_alerts['avg_volume'],
            'VWAP': df_alerts['vwap'].apply(lambda x: f"${x:.2f}"),
            'Message': df_alerts['message']
        })
        
        def highlight_severity(row):
            severity = df_alerts.loc[row.name, 'severity']
            if severity == 'up':
                return ['background-color: rgba(56, 239, 125, 0.15)'] * len(row)
            elif severity == 'down':
                return ['background-color: rgba(255, 106, 0, 0.15)'] * len(row)
            elif severity == 'warning':
                return ['background-color: rgba(255, 215, 0, 0.15)'] * len(row)
            else:
                return ['background-color: rgba(78, 84, 200, 0.15)'] * len(row)
        
        styled_df = df_display.style.apply(highlight_severity, axis=1)
        
        st.dataframe(
            styled_df,
            width="stretch",
            height=800,
            hide_index=True
        )
    
    # Si le système est en cours d'exécution, scanner les tickers
    if 'running' in st.session_state and st.session_state.running:
        # Attendre avant le scan SAUF au premier démarrage
        if 'first_run' not in st.session_state:
            # Premier scan, pas d'attente
            st.session_state.first_run = False
        else:
            # Scans suivants, attendre l'intervalle
            time.sleep(refresh_interval)
            
            # Si mode simulation, avancer le temps de simulation
            if st.session_state.simulation_time is not None:
                st.session_state.simulation_time = st.session_state.simulation_time + timedelta(seconds=refresh_interval)
        
        status_placeholder = st.empty()
        
        params = st.session_state.params if 'params' in st.session_state else {}
        
        # Déterminer quelle date/heure utiliser
        if st.session_state.simulation_time is not None:
            end_datetime = st.session_state.simulation_time
            period_days = st.session_state.get('backtest_period_days', 5)
            status_placeholder.info(f"🔄 Scan simulation... ({len(st.session_state.tickers)} tickers) - Temps: {end_datetime.strftime('%H:%M:%S')}")
        else:
            end_datetime = None
            period_days = 1
            status_placeholder.info(f"🔄 Scan en cours... ({len(st.session_state.tickers)} tickers)")
        
        # Scanner tous les tickers
        for i, ticker in enumerate(st.session_state.tickers):
            if end_datetime:
                status_placeholder.info(f"🔄 Scan simulation... {ticker} ({i+1}/{len(st.session_state.tickers)}) - {end_datetime.strftime('%H:%M:%S')}")
            else:
                status_placeholder.info(f"🔄 Scan en cours... {ticker} ({i+1}/{len(st.session_state.tickers)})")
            
            data = system.load_ticker_data(ticker, end_datetime, period_days)
            
            if data:
                alerts = system.detect_alerts(data, params)
                
                for alert in alerts:
                    # Format de date complet: DDMMYYYY HH:MI
                    candle_datetime = data['candle_time'].strftime("%d%m%Y %H:%M")
                    message_with_time = f"{alert['message']} [Bougie: {candle_datetime}]"
                    
                    alert_info = {
                        'timestamp': datetime.now().strftime("%H:%M:%S"),
                        'ticker': ticker,
                        'type': alert['type'],
                        'severity': alert['severity'],
                        'icon': alert['icon'],
                        'message': message_with_time,
                        'price': data['price'],
                        'change': data['change'],
                        'volume': system.format_volume(data['volume']),
                        'avg_volume': system.format_volume(data['avg_volume']),
                        'vwap': data['vwap'],
                        'high': data['high'],
                        'low': data['low']
                    }
                    st.session_state.alerts.insert(0, alert_info)
                    st.session_state.alert_count += 1
                
                st.session_state.previous_data[ticker] = data
        
        if end_datetime:
            status_placeholder.success(f"✅ Scan terminé! Temps simulation: {end_datetime.strftime('%H:%M:%S')} - Prochain dans {refresh_interval}s")
        else:
            status_placeholder.success(f"✅ Scan terminé! Prochain scan dans {refresh_interval} secondes...")
        
        st.rerun()

if __name__ == "__main__":
    main()