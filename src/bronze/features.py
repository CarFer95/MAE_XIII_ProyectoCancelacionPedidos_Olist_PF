# src/bronze/features.py

import numpy as np
import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def create_features(
    df: pd.DataFrame,
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    geo: pd.DataFrame,
    customers: pd.DataFrame,
    sellers: pd.DataFrame,
) -> pd.DataFrame:
    """
    BLOQUE 5:
    - Crea todas las features definidas en tu notebook original.
    - Devuelve df enriquecido.
    """
    df = df.copy()
    print("Creando features...")

    # ---------- 5.1 FEATURES TEMPORALES ----------
    df["purchase_day"] = df["order_purchase_timestamp"].dt.day
    df["purchase_weekday"] = df["order_purchase_timestamp"].dt.weekday
    df["purchase_week"] = df["order_purchase_timestamp"].dt.isocalendar().week.astype(int)
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_year"] = df["order_purchase_timestamp"].dt.year
    df["is_weekend_purchase"] = df["purchase_weekday"].isin([5, 6]).astype(int)

    df["days_between_approved_and_purchase"] = (
        df["order_approved_at"] - df["order_purchase_timestamp"]
    ).dt.days

    df["days_between_purchase_and_delivery_estimate"] = (
        df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
    ).dt.days

    df["days_between_purchase_and_review"] = (
        pd.to_datetime(df["review_creation_date"], errors="coerce")
        - df["order_purchase_timestamp"]
    ).dt.days

    # ---------- 5.2 FEATURES DEL PEDIDO (AGREGADAS POR order_id) ----------
    order_agg = df.groupby("order_id").agg(
        num_items_in_order=("order_item_id", "count"),
        total_price_order=("price", "sum"),
        total_freight_order=("freight_value", "sum"),
        avg_price_item=("price", "mean"),
        max_price_item=("price", "max"),
        min_price_item=("price", "min"),
        std_price_item=("price", "std"),
        num_sellers_in_order=("seller_id", "nunique"),
        num_products_in_order=("product_id", "nunique"),
    ).reset_index()

    df = df.merge(order_agg, on="order_id", how="left")

    df["freight_rate"] = df["total_freight_order"] / df["total_price_order"]
    df["order_total_value"] = df["total_price_order"] + df["total_freight_order"]
    df["avg_freight_per_item"] = df["total_freight_order"] / df["num_items_in_order"]

    # ---------- 5.3 FEATURES DE PRODUCTO ----------
    df["product_volume"] = (
        df["product_length_cm"] * df["product_height_cm"] * df["product_width_cm"]
    )

    df["is_heavy_order"] = (df["product_weight_g"] > df["product_weight_g"].median()).astype(int)
    df["is_large_volume"] = (df["product_volume"] > df["product_volume"].median()).astype(int)

    cat_agg = df.groupby("product_category_name").agg(
        avg_category_weight=("product_weight_g", "mean"),
        std_category_weight=("product_weight_g", "std"),
        avg_category_volume=("product_volume", "mean"),
        std_category_volume=("product_volume", "std"),
        avg_category_price=("price", "mean"),
        std_category_price=("price", "std"),
    ).reset_index()

    df = df.merge(cat_agg, on="product_category_name", how="left")

    df["product_above_avg_weight"] = (
        df["product_weight_g"] > df["avg_category_weight"]
    ).astype(int)
    df["product_above_avg_volume"] = (
        df["product_volume"] > df["avg_category_volume"]
    ).astype(int)
    df["product_price_relative_to_category_avg"] = df["price"] / df["avg_category_price"]

    # ---------- 5.4 FEATURES DE PAGO (AGREGADAS POR order_id) ----------
    pay_agg = df.groupby("order_id").agg(
        num_payments=("payment_sequential", "max"),
        payment_value_total=("payment_value", "sum"),
        payment_installments_total=("payment_installments", "sum"),
        main_payment_type=(
            "payment_type",
            lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0],
        ),
    ).reset_index()

    df = df.merge(pay_agg, on="order_id", how="left")

    df["avg_installment_value"] = df["payment_value_total"] / df["payment_installments_total"]
    df["is_multiple_payment_types"] = (
        df.groupby("order_id")["payment_type"].transform("nunique") > 1
    ).astype(int)

    df["main_payment_type_credit_card"] = (df["main_payment_type"] == "credit_card").astype(int)
    df["main_payment_type_voucher"] = (df["main_payment_type"] == "voucher").astype(int)
    df["main_payment_type_debit_card"] = (df["main_payment_type"] == "debit_card").astype(int)
    df["main_payment_type_boleto"] = (df["main_payment_type"] == "boleto").astype(int)

    # ---------- 5.5 FEATURES DE REVIEWS ----------
    df["is_low_review"] = (df["review_score"] <= 2).astype(int)

    df["review_creation_delay_days"] = (
        pd.to_datetime(df["review_creation_date"], errors="coerce")
        - df["order_purchase_timestamp"]
    ).dt.days

    df["review_answer_delay_days"] = (
        pd.to_datetime(df["review_answer_timestamp"], errors="coerce")
        - pd.to_datetime(df["review_creation_date"], errors="coerce")
    ).dt.days

    df["has_review_comment"] = df["review_comment_message"].notna().astype(int)
    df["review_comment_length"] = df["review_comment_message"].astype(str).apply(len)

    # ---------- 5.6 FEATURES DE VENDEDOR ----------
    seller_metrics = df.groupby("seller_id").agg(
        seller_rating_avg=("review_score", "mean"),
        seller_review_variability=("review_score", "std"),
        seller_cancel_rate_avg=("order_canceled_extended", "mean"),
        seller_total_sales=("order_id", "count"),
    ).reset_index()

    df = df.merge(seller_metrics, on="seller_id", how="left")

    order_items = order_items.copy()
    order_items["shipping_limit_date"] = pd.to_datetime(
        order_items["shipping_limit_date"],
        errors="coerce",
    )

    seller_exp = (
        order_items.groupby("seller_id")["shipping_limit_date"].max()
        - order_items.groupby("seller_id")["shipping_limit_date"].min()
    ).dt.days.reset_index().rename(
        columns={"shipping_limit_date": "seller_experience_days"}
    )

    df = df.merge(seller_exp, on="seller_id", how="left")

    # ---------- 5.7 FEATURES DE CLIENTE ----------
    cust_hist = orders.groupby("customer_id").agg(
        customer_order_history_count=("order_id", "count"),
        customer_cancellation_history=("order_canceled_extended", "sum"),
    ).reset_index()

    df = df.merge(cust_hist, on="customer_id", how="left")

    df["customer_avg_ticket"] = df.groupby("customer_id")["total_price_order"].transform("mean")
    df["customer_avg_freight_paid"] = df.groupby("customer_id")["total_freight_order"].transform("mean")

    df["customer_days_since_last_order"] = (
        df.groupby("customer_id")["order_purchase_timestamp"].transform("max")
        - df["order_purchase_timestamp"]
    ).dt.days

    df["customer_rfm_recency"] = df["customer_days_since_last_order"]
    df["customer_rfm_frequency"] = df.groupby("customer_id")["order_id"].transform("count")
    df["customer_rfm_monetary"] = df.groupby("customer_id")["total_price_order"].transform("sum")

    # ---------- 5.8 FEATURES DE DISTANCIA (geo) ----------
    geo_group = geo.groupby("geolocation_zip_code_prefix")[
        ["geolocation_lat", "geolocation_lng"]
    ].mean().reset_index()

    customers_geo = customers.merge(
        geo_group,
        left_on="customer_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left",
    ).rename(
        columns={
            "geolocation_lat": "geolocation_lat_customer",
            "geolocation_lng": "geolocation_lng_customer",
        }
    )

    sellers_geo = sellers.merge(
        geo_group,
        left_on="seller_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left",
    ).rename(
        columns={
            "geolocation_lat": "geolocation_lat_seller",
            "geolocation_lng": "geolocation_lng_seller",
        }
    )

    df = df.merge(
        customers_geo[["customer_id", "geolocation_lat_customer", "geolocation_lng_customer"]],
        on="customer_id",
        how="left",
    ).merge(
        sellers_geo[["seller_id", "geolocation_lat_seller", "geolocation_lng_seller"]],
        on="seller_id",
        how="left",
    )

    df["distance_seller_customer_km"] = haversine_km(
        df["geolocation_lat_customer"],
        df["geolocation_lng_customer"],
        df["geolocation_lat_seller"],
        df["geolocation_lng_seller"],
    )

    # ---------- 5.9 VARIABLES DE RIESGO DERIVADAS ----------
    df["risk_heavy_package"] = df["is_heavy_order"]
    df["risk_multiple_sellers"] = (df["num_sellers_in_order"] > 1).astype(int)
    df["risk_long_distance"] = (
        df["distance_seller_customer_km"] > df["distance_seller_customer_km"].median()
    ).astype(int)
    df["risk_low_review"] = df["is_low_review"]
    df["risk_many_installments"] = (df["payment_installments_total"] > 5).astype(int)
    df["risk_weekend_purchase"] = df["is_weekend_purchase"]
    df["risk_high_freight_ratio"] = (
        df["freight_rate"] > df["freight_rate"].median()
    ).astype(int)

    print("\nFeatures creadas. Shape final:", df.shape)

    num_vars = df.select_dtypes(include=["int64", "float64"]).shape[1]
    print("Número de variables numéricas actuales:", num_vars)

    return df
