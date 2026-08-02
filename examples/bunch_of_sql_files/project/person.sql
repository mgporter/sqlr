with dedupped as (
  select 
    *,
    row_number() over (partition by id order by modified_at desc) as rn
  from mydatabase.myschema.person
  where age > 20
),
projected as (
  select
    b.id,
    b.name,
    b.age,
    b.email,
    b.modified_at,
    a.street,
    a.city
  from dedupped b
  left join mydatabase.myschema.address a on b.id = a.person_id
  where b.rn = 1
)
select * from projected