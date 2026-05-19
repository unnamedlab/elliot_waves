# Elliot Waves Backtester (S&P 500)

Aplicación en Python para backtestear una estrategia **inspirada en ondas de Elliott** (sin lookahead), y compararla contra:

- **Buy & Hold**
- **DCA (Dollar Cost Averaging)**

## Principio clave: sin usar información del futuro

La detección de pivotes usa confirmación causal con retardo `k`:

- En el día `t`, solo se confirma un pivote en `t-k` cuando ya existen suficientes velas posteriores.
- Las decisiones de compra/venta se toman únicamente cuando el pivote queda confirmado.

Esto evita “pintar” ondas con datos que todavía no existían en tiempo real.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecución

```bash
python app.py --start 2000-01-01 --cash 10000 --pivot-k 3 --out backtest_results.csv
```

Opciones:

- `--start`: fecha inicio (default `2000-01-01`)
- `--end`: fecha fin (opcional)
- `--cash`: capital inicial
- `--pivot-k`: retardo de confirmación de pivotes
- `--out`: CSV de salida con equity curves

## Qué hace la estrategia Elliott-like

1. Detecta pivotes locales confirmados (mínimos/máximos causales).
2. Busca secuencias de 8 pivotes que aproximen un ciclo `1-2-3-4-5-A-B-C` alcista.
3. Aplica validaciones tipo Elliott (estructura y retroceso aproximado Fibonacci 35%-68%).
4. Entra largo tras confirmación de corrección ABC.
5. Sale si el precio rompe el último pivote mínimo confirmado.

> Nota: es una aproximación cuantitativa/algorítmica de una teoría subjetiva. No es asesoría financiera.

## Salida

- Resumen en consola (valor final, retorno total, CAGR, volatilidad, Sharpe, max drawdown).
- CSV con columnas:
  - `Date`, `Close`
  - `equity_elliott`
  - `equity_buy_hold`
  - `equity_dca`
