# src/bronze/master_table.py

import pandas as pd


def build_master_table(
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    order_items: pd.DataFrame,
    order_payments: pd.DataFrame,
    reviews: pd.DataFrame,
    products: pd.DataFrame,
    sellers: pd.DataFrame,
    cat_trad: pd.DataFrame,
) -> pd.DataFrame:
    """
    BLOQUE 4:
    - Valida duplicados en claves.
    - Realiza el merge principal para construir la Master Table.
    """
    print("Iniciando construcción de la Master Table...\n")

    # 1) Validar duplicados en claves principales
    print("Duplicados en order_id (orders):", orders["order_id"].duplicated().sum())
    print("Duplicados en customer_id (customers):", customers["customer_id"].duplicated().sum())
    print("Duplicados en seller_id (sellers):", sellers["seller_id"].duplicated().sum())
    print("Duplicados en product_id (products):", products["product_id"].duplicated().sum())

    # 2) MERGE PRINCIPAL (orden lógico)
    df = (
        orders
        .merge(customers, on="customer_id", how="left")
        .merge(order_items, on="order_id", how="left")
        .merge(products, on="product_id", how="left")
        .merge(order_payments, on="order_id", how="left")
        .merge(reviews, on="order_id", how="left")
        .merge(sellers, on="seller_id", how="left")
        .merge(cat_trad, on="product_category_name", how="left")
    )

    print("\nShape tras merge:", df.shape)

    # 3) Validación básica post-merge
    print("\nProporción de nulos por tabla clave:")
    cols_check = ["customer_id", "seller_id", "product_id", "payment_type", "review_score"]
    print(df[cols_check].isna().mean().round(3))

    # 4) Vista rápida
    print("\nVista rápida de la tabla unificada:")
    print(df.head())

    return df
