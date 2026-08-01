with dedupped as (
  select 
    *,
    row_number() over (partition by id order by modified_at desc) as rn
  from mydatabase.myschema.person
  where age > 20
),
projected as (
  select
    id,
    name,
    age,
    email,
    modified_at
  from dedupped
  where rn = 1
)
select * from projected