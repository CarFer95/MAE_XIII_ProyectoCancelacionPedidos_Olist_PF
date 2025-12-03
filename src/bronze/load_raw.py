# src/bronze/load_raw.py

import os
import pandas as pd
from typing import Dict
from src.config.config import Settings


EXPECTED_FILES = [
    "olist_customers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
]


def validate_and_get_files(cfg: Settings) -> Dict[str, str]:
    """
    BLOQUE 1:
    - Lista archivos en RUTA_BASE.
    - Valida que existan todos los datasets esperados.
    - Devuelve un diccionario FILES con las rutas completas.
    """
    ruta_base = cfg.RUTA_BASE_BRONZE

    print("Archivos encontrados en la carpeta base:")
    found_files = os.listdir(ruta_base)
    print(found_files)

    missing = [f for f in  EXPECTED_FILES if f not in found_files]

    if len(missing) == 0:
        print("\nTodos los datasets requeridos están en la carpeta base.")
    else:
        print("\nFaltan estos archivos en la carpeta base:")
        for f in missing:
            print(" -", f)

    files = {f: os.path.join(ruta_base, f) for f in EXPECTED_FILES}

    print("\nDiccionario FILES creado. Ejemplo:")
    for k in list(files.keys())[:3]:
        print(k, "->", files[k])

    return files






def load_raw_tables(files: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """
    BLOQUE 2:
    - Carga todas las tablas desde FILES.
    - Parsea fechas en orders.
    - Devuelve un diccionario con todos los DataFrames.
    """
    customers = pd.read_csv(files["olist_customers_dataset.csv"])
    geo = pd.read_csv(files["olist_geolocation_dataset.csv"])
    order_items = pd.read_csv(files["olist_order_items_dataset.csv"])
    order_payments = pd.read_csv(files["olist_order_payments_dataset.csv"])
    reviews = pd.read_csv(files["olist_order_reviews_dataset.csv"])
    products = pd.read_csv(files["olist_products_dataset.csv"])
    sellers = pd.read_csv(files["olist_sellers_dataset.csv"])
    cat_trad = pd.read_csv(files["product_category_name_translation.csv"])

    orders = pd.read_csv(
        files["olist_orders_dataset.csv"],
        parse_dates=[
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
        dayfirst=False,
        infer_datetime_format=True,
    )

    print("Shapes de tablas cargadas:")
    print("- customers:", customers.shape)
    print("- geo:", geo.shape)
    print("- order_items:", order_items.shape)
    print("- order_payments:", order_payments.shape)
    print("- reviews:", reviews.shape)
    print("- orders:", orders.shape)
    print("- products:", products.shape)
    print("- sellers:", sellers.shape)
    print("- cat_trad:", cat_trad.shape)

    print("\nColumnas de orders:")
    print(orders.columns.tolist())

    print("\nEstado de las fechas en orders:")
    print(
        orders[
            [
                "order_purchase_timestamp",
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
            ]
        ].dtypes
    )

    return {
        "customers": customers,
        "geo": geo,
        "order_items": order_items,
        "order_payments": order_payments,
        "reviews": reviews,
        "orders": orders,
        "products": products,
        "sellers": sellers,
        "cat_trad": cat_trad,
    }
