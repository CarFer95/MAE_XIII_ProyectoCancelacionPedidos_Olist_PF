# src/bronze/targets.py

import numpy as np
import pandas as pd


def add_targets(orders: pd.DataFrame) -> pd.DataFrame:
    """
    BLOQUE 3:
    - Crea target estricto is_canceled_strict.
    - Crea target extendido order_canceled_extended.
    Devuelve el DataFrame orders con las dos columnas agregadas.
    """
    orders = orders.copy()

    # 1) Target estricto: solo estado 'canceled'
    orders["is_canceled_strict"] = (orders["order_status"] == "canceled").astype(int)

    # 2.1 Cancelado explícito
    cond_canceled = orders["order_status"] == "canceled"

    # 2.2 Unavailable (pedido no completado por stock/pago)
    cond_unavailable = orders["order_status"] == "unavailable"

    # 2.3 Pedidos "colgados": created / processing sin entrega después de 30 días
    last_purchase = orders["order_purchase_timestamp"].max()
    umbral_fecha = last_purchase - pd.Timedelta(days=30)

    cond_pending_old = (
        orders["order_status"].isin(["created", "processing"])
        & orders["order_delivered_customer_date"].isna()
        & (orders["order_purchase_timestamp"] < umbral_fecha)
    )

    # 3) Target compuesto
    orders["order_canceled_extended"] = np.where(
        cond_canceled | cond_unavailable | cond_pending_old,
        1,
        0,
    )

    # 4) Resumen de tasas
    print("Tasa estricta (solo 'canceled'):", round(orders["is_canceled_strict"].mean(), 4))
    print("Tasa extendida:", round(orders["order_canceled_extended"].mean(), 4))

    print("\nConteo target extendido:")
    print(orders["order_canceled_extended"].value_counts())

    print("\nCruce order_status vs target extendido:")
    print(pd.crosstab(orders["order_status"], orders["order_canceled_extended"]))

    return orders
