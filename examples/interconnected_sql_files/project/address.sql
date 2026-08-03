with ranked as (
  select
    *,
    row_number() over (partition by person_id order by updated_at desc) as rn
  from mydatabase.myschema.raw_address
  where country = 'US'
),
current_address as (
  select
    person_id,
    street,
    city,
    state,
    zip,
    street || ', ' || city || ', ' || state || ' ' || zip as full_address
  from ranked
  where rn = 1
)
select * from current_address
