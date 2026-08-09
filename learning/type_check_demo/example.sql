with enriched as (
    select
        order_id,
        date_trunc('day', order_ts)                  as order_day,
        amount_cents / 100.0                         as amount,
        upper(status_code)                           as status,
        datediff('day', order_ts, current_timestamp) as age_days,
        list_extract(split(promo_csv, ','), 1)       as first_promo
    from orders
),
flagged as (
    select
        *,
        amount > 100              as is_large,
        round(status, 2)          as bogus,
        zeroifnull(amount)        as safe_amount,
        list_extract(first_promo) as broken
    from enriched
)
select order_id, order_day, amount, is_large
from flagged
where age_days < 30
