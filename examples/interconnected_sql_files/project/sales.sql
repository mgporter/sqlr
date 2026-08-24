with online_orders as (
  select
    order_id,
    customer_id,
    employee_id,
    order_date,
    amount,
    'online' as channel
  from mydatabase.raw.online_order
  where amount > 0
),
instore_orders as (
  select
    order_id,
    customer_id,
    employee_id,
    order_date,
    amount,
    'in_store' as channel
  from mydatabase.raw.instore_order
  where amount > 0
),
all_orders as (
  select order_id, customer_id, employee_id, order_date, amount, channel
  from online_orders
  union all
  select order_id, customer_id, employee_id, order_date, amount, channel
  from instore_orders
),
active_customer_orders as (
  select o.*
  from all_orders o
  where exists (
    select 1
    from mydatabase.raw.customer c
    where c.customer_id = o.customer_id
      and c.is_active = true
  )
),
distinct_order_customers as (
  select customer_id, employee_id from online_orders
  union
  select customer_id, employee_id from instore_orders
),
customer_counts as (
  select
    employee_id,
    count(*) as distinct_customer_count
  from distinct_order_customers
  group by employee_id
),
order_summary as (
  select
    o.employee_id,
    o.order_date,
    o.channel,
    sum(o.amount) as daily_total,
    count(*) as order_count,
    avg(o.amount) as avg_order_amount,
    row_number() over (partition by o.employee_id order by o.order_date desc) as recency_rank,
    sum(o.amount) over (partition by o.employee_id order by o.order_date) as running_total,
    c.distinct_customer_count
  from active_customer_orders o
  left join customer_counts c
    on o.employee_id = c.employee_id
  group by o.employee_id, o.order_date, o.channel, c.distinct_customer_count
)
select * from order_summary
